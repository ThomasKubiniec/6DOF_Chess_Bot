"""
Training script for the TD3 Robot Arm IK Agent — vectorised K-environment loop.

────────────────────────────────────────────────────────────────────────────
HOW TO RUN THIS PROGRAM
────────────────────────────────────────────────────────────────────────────

1. HYPERPARAMETER TUNING  (run this first)
   ----------------------------------------
   Runs N Optuna trials, each a full 10k-episode training run.
   Saves the best hyperparameters to best_params.json when done.

     python train.py --mode tune --n_trials 50

   To watch the live Optuna dashboard during tuning, open a second
   terminal and run:

     python train.py --mode dashboard

   The dashboard will be at http://localhost:8080 in your browser.
   It shows optimization history, parameter importance, parallel
   coordinates, and convergence plots.

   To tune on CPU with parallel workers instead of sequential GPU:

     python train.py --mode tune --parallel_mode cpu --n_jobs 4

   To fix specific parameters and only tune the rest:

     python train.py --mode tune --no_tune_hidden_width --no_tune_hidden_depth

2. TRAINING WITH TUNED PARAMETERS
   ---------------------------------
   After tuning, load the saved best parameters for the final training run:

     python train.py --mode train --load_params best_params.json

   To train with default hyperparameters (no tuning file):

     python train.py --mode train

   To enable self-collision checking during training (slower but physically
   accurate — recommended for phase 2 / fine-tuning):

     python train.py --mode train --load_params best_params.json --check_collisions

3. DASHBOARD (view a previous or ongoing study)
   -----------------------------------------------
     python train.py --mode dashboard
     python train.py --mode dashboard --storage_path my_study.db --dashboard_port 8080

4. FULL OPTION REFERENCE
   -----------------------
   --mode              train | tune | dashboard        (default: train)
   --load_params       path to JSON file from tuning   (default: none)
   --check_collisions  flag: enable capsule collision   (default: off)
   --n_episodes        number of training episodes      (default: 10000)
   --max_steps         max timesteps per episode        (default: 200)
   --updates_per_episode TD3 gradient steps per episode (default: 50)
   --batch_size        replay buffer sample size        (default: 2048)
   --n_envs            parallel environments K          (default: 8)
   --warmup_transitions transitions to collect before training (default: 500000)
   --decay_expl_noise  flag: decay sigma over training  (default: off)
   --n_trials          Optuna trials                    (default: 50)
   --storage_path      SQLite file for Optuna study     (default: optuna_study.db)
   --dashboard_port    port for Optuna dashboard        (default: 8080)
   --parallel_mode     gpu | cpu  (for tuning)          (default: gpu)
   --n_jobs            CPU workers for parallel tuning  (default: 4)
   --no_tune_X         fix parameter X at its default   (e.g. --no_tune_hidden_width)

   Tuneable parameters: gamma, tau, actor_lr, critic_lr, hidden_width,
   hidden_depth, n_envs, pos_w, rot_w, vel_lambda, acc_lambda,
   her_ratio_start, her_decay_steps, expl_sigma, expl_sigma_end, target_sigma
"""

import argparse
import copy
import json
import time
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np
import torch
import optuna
import optuna_dashboard

from forward_kinematics import Robot_math
from rewards_math       import Reward_Math
from rot_math           import to_6D_R_batch
from replay_buffer      import Replay_Buffer
from td3                import TD3


# ---------------------------------------------------------------------------
# Robot definition
# ---------------------------------------------------------------------------

def make_robot(device) -> Robot_math:
    my_a     = [0.0, 7.375, 0.0,  0.0,  0.0,  0.0]
    my_alpha = [np.deg2rad(90),  np.deg2rad(180), np.deg2rad(90),
                np.deg2rad(90),  np.deg2rad(-90),  np.deg2rad(0)]
    my_d     = [-3.5, 0.0, 0.0, 8.25, 0.0, 5.1875]
    my_theta = [np.deg2rad(0),   np.deg2rad(0),   np.deg2rad(90),
                np.deg2rad(180), np.deg2rad(0),   np.deg2rad(-90)]
    my_bounds = [
        (np.deg2rad(-90),  np.deg2rad(90)),
        (np.deg2rad(-180), np.deg2rad(0)),
        (np.deg2rad(-90),  np.deg2rad(90)),
        (np.deg2rad(-90),  np.deg2rad(90)),
        (np.deg2rad(-90),  np.deg2rad(90)),
        (np.deg2rad(-90),  np.deg2rad(90)),
    ]
    robot = Robot_math(
        a=my_a, alpha=my_alpha, d=my_d, theta=my_theta,
        joint_type=["r"] * 6, bounds=my_bounds,
        fail_dist=[0.1] * 6,
        device=device,
    )
    robot.WT = robot.make_homogenous_transformation(
        yaw=0, pitch=0, roll=180, x=0, y=0, z=0)
    return robot


# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

@dataclass
class HParams:
    # Training loop
    n_episodes:          int   = 10_000
    max_steps:           int   = 200
    updates_per_episode: int   = 50
    batch_size:          int   = 2048
    n_envs:              int   = 8
    # Number of transitions to collect with uniform random actions before
    # any TD3 training begins.  HER ratio is held at her_ratio_start during
    # this phase and begins decaying only once training starts.
    # Rule of thumb: fill roughly half the buffer for good initial coverage.
    # At K=8 envs, ~180 steps/ep, 1.8x HER: ~500k transitions ≈ 350 episodes.
    warmup_transitions:  int   = 500_000

    # TD3
    gamma:               float = 0.99
    tau:                 float = 0.005
    actor_lr:            float = 1e-4
    critic_lr:           float = 1e-3
    policy_delay:        int   = 2
    hidden_width:        int   = 256
    hidden_depth:        int   = 2

    # Exploration noise
    expl_sigma:          float = 0.1
    expl_sigma_end:      float = 0.02
    expl_clip:           float = 0.5
    decay_expl_noise:    bool  = False

    # Target policy smoothing
    target_sigma:        float = 0.2
    target_clip:         float = 0.5

    # Reward weights
    pos_w:               float = 1.0
    rot_w:               float = 0.5
    crash_w:             float = 2.0
    dist_w:              float = 0.1
    vel_lambda:          float = 0.1
    acc_lambda:          float = 0.1

    # HER
    her_ratio_start:     float = 0.80
    her_ratio_end:       float = 0.20
    her_decay_steps:     int   = 1_000

    # Replay buffer
    buffer_size:         int   = 1_000_000


def save_params(hp: HParams, path: str):
    with open(path, "w") as f:
        json.dump(asdict(hp), f, indent=2)
    print(f"[Params] Saved to {path}")


def load_params(path: str, base: HParams) -> HParams:
    with open(path) as f:
        d = json.load(f)
    hp = copy.deepcopy(base)
    for k, v in d.items():
        if hasattr(hp, k):
            setattr(hp, k, v)
    print(f"[Params] Loaded from {path}")
    return hp


@dataclass
class TuneConfig:
    gamma:           bool = True
    tau:             bool = True
    actor_lr:        bool = True
    critic_lr:       bool = True
    hidden_width:    bool = False
    hidden_depth:    bool = False
    n_envs:          bool = False
    pos_w:           bool = True
    rot_w:           bool = True
    vel_lambda:      bool = True
    acc_lambda:      bool = True
    her_ratio_start: bool = True
    her_decay_steps: bool = True
    expl_sigma:      bool = True
    expl_sigma_end:  bool = True
    target_sigma:    bool = True


# ---------------------------------------------------------------------------
# Pose sampling
# ---------------------------------------------------------------------------

def sample_random_q(robot: Robot_math, K: int) -> torch.Tensor:
    low  = robot.low_bounds.unsqueeze(0)
    high = robot.high_bounds.unsqueeze(0)
    return low + (high - low) * torch.rand(
        K, len(robot.a), dtype=torch.float64, device=robot.device)


def sample_random_q_collision_free(robot: Robot_math, K: int) -> torch.Tensor:
    results = []
    while len(results) < K:
        q = (robot.low_bounds + (robot.high_bounds - robot.low_bounds)
             * torch.rand(len(robot.a), dtype=torch.float64, device=robot.device))
        robot.q_vect = q
        crash, _ = robot.do_fk_and_check_crash()
        if not crash:
            results.append(q)
    return torch.stack(results)


def batch_ee_pose(robot: Robot_math, q_batch: torch.Tensor):
    return robot.give_ds_batch(q_batch), robot.give_Rs_batch(q_batch)


# ---------------------------------------------------------------------------
# Vectorised environment state
# ---------------------------------------------------------------------------

class VecEnvState:
    def __init__(self, K: int, n: int, device: torch.device):
        self.K, self.n, self.device = K, n, device
        self.q_curr       = torch.zeros(K, n,    dtype=torch.float64, device=device)
        self.pos_curr     = torch.zeros(K, 3,    dtype=torch.float64, device=device)
        self.R_curr       = torch.zeros(K, 3, 3, dtype=torch.float64, device=device)
        self.pos_goal     = torch.zeros(K, 3,    dtype=torch.float64, device=device)
        self.R_goal       = torch.zeros(K, 3, 3, dtype=torch.float64, device=device)
        self.delta_q_prev = torch.zeros(K, n,    dtype=torch.float64, device=device)
        self.steps        = torch.zeros(K,       dtype=torch.long,    device=device)
        self.ep_reward    = torch.zeros(K,       dtype=torch.float64, device=device)
        # Per-env episode caches committed to the buffer at episode end
        self.caches: list[list[dict]] = [[] for _ in range(K)]


# ---------------------------------------------------------------------------
# Vectorised step
# ---------------------------------------------------------------------------

def vectorised_step(env:              VecEnvState,
                    robot:            Robot_math,
                    reward_math:      Reward_Math,
                    td3:              TD3,
                    buffer:           Replay_Buffer,
                    hp:               HParams,
                    total_steps:      int,
                    update_step:      int,
                    check_collisions: bool,
                    current_sigma:    float,
                    episode_log:      list,
                    uniform_random:   bool = False,
                    do_updates:       bool = True) -> tuple[int, int]:
    """
    Advance all K environments by one timestep.

    uniform_random : if True, bypass actor and sample actions uniformly.
                     Used during the dedicated warmup phase.
    do_updates     : if False, skip TD3 gradient updates this step.
                     Set False during warmup so the buffer fills first.

    Returns updated (total_steps, update_step).
    """
    K      = env.K
    n      = env.n
    device = robot.device

    # ── 1. Build batched state (K, S) using reward_math / rot_math methods ──
    # norm_dist: (K, 3) — normalised displacement vector to goal
    norm_dist = reward_math.get_normal_dist_to_goal(env.pos_curr,
                                                     env.pos_goal)          # (K,3)
    # rot_6d: (K, 6) — first two columns of each goal rotation matrix
    rot_6d    = to_6D_R_batch(env.R_goal)                                   # (K,6)
    # prev_vel: (K, n) — previous action in normalised velocity space
    prev_vel  = reward_math.get_normal_joint_vel(env.delta_q_prev)         # (K,n)
    state_f64 = torch.cat([norm_dist, rot_6d, prev_vel], dim=1)            # (K,S)
    state_f32 = state_f64.to(dtype=torch.float32, device=td3.device)

    # ── 2. Select actions ─────────────────────────────────────────────────
    if uniform_random:
        a_norm_f32 = torch.rand(K, n, dtype=torch.float32, device=td3.device) * 2 - 1
    else:
        with torch.no_grad():
            a_norm_f32 = td3.actor(state_f32)
        noise      = (torch.randn_like(a_norm_f32) * current_sigma
                      ).clamp(-hp.expl_clip, hp.expl_clip)
        a_norm_f32 = (a_norm_f32 + noise).clamp(-1.0, 1.0)

    a_norm_f64 = a_norm_f32.to(dtype=torch.float64, device=device)

    # ── 3. Apply actions using reward_math normalisation methods ────────────
    # Normalise current q, add normalised delta, clamp to [-1,1], denormalise
    q_norm     = reward_math.get_normal_joint_value(env.q_curr)            # (K,n)
    q_new_norm = (q_norm + a_norm_f64).clamp(-1.0, 1.0)                   # (K,n)
    q_new      = reward_math.get_original_joint_value(q_new_norm)         # (K,n)
    # Denormalise action to real joint velocity units for reward/buffer
    vel_low    = robot.low_bounds  - robot.high_bounds                     # (n,)
    vel_high   = robot.high_bounds - robot.low_bounds                      # (n,)
    delta_q    = reward_math.denormalize_from_range(
        a_norm_f64,
        vel_low.unsqueeze(0).expand(K, -1),
        vel_high.unsqueeze(0).expand(K, -1))                               # (K,n)

    # ── 4. Batched FK ─────────────────────────────────────────────────────
    pos_next, R_next = batch_ee_pose(robot, q_new)                         # (K,3),(K,3,3)

    # ── 5. Batched reward ─────────────────────────────────────────────────
    r_batch, _ = reward_math.reward_batch(
        pos_curr_batch     = env.pos_curr,
        R_curr_SO3_batch   = env.R_curr,
        delta_q_new_batch  = delta_q,
        delta_q_prev_batch = env.delta_q_prev,
        pos_goal_batch     = env.pos_goal,
        R_goal_SO3_batch   = env.R_goal,
        use_focal          = True,
    )                                                                       # (K,)

    # ── 6. Optional collision check (sequential) ──────────────────────────
    crashed_k = torch.zeros(K, dtype=torch.bool, device=device)
    if check_collisions:
        for k in range(K):
            robot.q_vect = q_new[k]
            crash, _ = robot.do_fk_and_check_crash()
            if crash:
                crashed_k[k] = True
                r_batch[k]  -= hp.crash_w

    # ── 7. Done flags ─────────────────────────────────────────────────────
    # Use reward_math methods for consistent normalisation
    eps_norm  = reward_math.get_normal_dist_to_goal(pos_next,
                                                     env.pos_goal)         # (K,3)
    e_pos_raw = torch.linalg.vector_norm(eps_norm, dim=1)                  # (K,)
    e_ori_raw = torch.linalg.matrix_norm(env.R_goal - R_next)              # (K,)
    success_k = ((reward_math.pos_w * e_pos_raw <= reward_math._good_pos_thresh) &
                 (reward_math.rot_w * e_ori_raw <= reward_math._good_ori_thresh))
    env.steps += 1
    timeout_k = (env.steps >= hp.max_steps)
    done_k    = success_k | timeout_k | crashed_k
    env.ep_reward += r_batch

    # ── 8. Cache steps; commit and reset finished environments ────────────
    for k in range(K):
        done = done_k[k].item()
        env.caches[k].append({
            "q_new":        q_new[k].cpu(),
            "delta_q_new":  delta_q[k].cpu(),
            "delta_q_prev": env.delta_q_prev[k].cpu(),
            "pos_curr":     env.pos_curr[k].cpu(),
            "R_curr":       env.R_curr[k].cpu(),
            "pos_next":     pos_next[k].cpu(),
            "R_next":       R_next[k].cpu(),
            "a_norm":       a_norm_f64[k].cpu(),
            "done":         float(done),
        })

        if done:
            buffer._episode_cache = env.caches[k]
            buffer.finish_episode(
                pos_goal   = env.pos_goal[k].cpu(),
                R_goal_SO3 = env.R_goal[k].cpu(),
            )
            grade = ("GOOD" if success_k[k] else
                     "CRASH" if crashed_k[k] else "FAIL")
            # Final normalised EE distance and orientation error at episode end
            # Used as the Optuna MAE objective — independent of reward weights.
            final_pos_err = e_pos_raw[k].item()          # normalised [0,1]
            final_ori_err = e_ori_raw[k].item()          # Frobenius norm
            episode_log.append({
                "total_reward":  env.ep_reward[k].item(),
                "steps":         env.steps[k].item(),
                "success":       bool(success_k[k].item()),
                "grade":         grade,
                "crashed":       bool(crashed_k[k].item()),
                "final_pos_err": final_pos_err,
                "final_ori_err": final_ori_err,
            })
            # Reset env k
            env.caches[k]      = []
            env.ep_reward[k]   = 0.0
            env.steps[k]       = 0
            if check_collisions:
                q_s = sample_random_q_collision_free(robot, 1)[0]
                q_g = sample_random_q_collision_free(robot, 1)[0]
            else:
                q_s = sample_random_q(robot, 1)[0]
                q_g = sample_random_q(robot, 1)[0]
            p_s, R_s = batch_ee_pose(robot, q_s.unsqueeze(0))
            p_g, R_g = batch_ee_pose(robot, q_g.unsqueeze(0))
            env.q_curr[k]       = q_s
            env.pos_curr[k]     = p_s[0]
            env.R_curr[k]       = R_s[0]
            env.pos_goal[k]     = p_g[0]
            env.R_goal[k]       = R_g[0]
            env.delta_q_prev[k] = torch.zeros(n, dtype=torch.float64, device=device)

    # ── 9. Advance non-done environments ──────────────────────────────────
    alive = ~done_k
    if alive.any():
        env.q_curr[alive]       = q_new[alive]
        env.pos_curr[alive]     = pos_next[alive]
        env.R_curr[alive]       = R_next[alive]
        env.delta_q_prev[alive] = delta_q[alive]

    total_steps += K

    # ── 10. TD3 updates (amortised) ───────────────────────────────────────
    updates_this_step = max(1, int(hp.updates_per_episode * K / hp.max_steps))
    if do_updates and len(buffer) >= hp.batch_size:
        for _ in range(updates_this_step):
            batch = buffer.sample(hp.batch_size)
            if batch is None:
                break
            td3.train_step(
                batch        = batch,
                gamma        = hp.gamma,
                tau          = hp.tau,
                policy_delay = hp.policy_delay,
                step         = update_step,
                target_sigma = hp.target_sigma,
                target_clip  = hp.target_clip,
            )
            update_step += 1

    return total_steps, update_step


# ---------------------------------------------------------------------------
# Full training run
# ---------------------------------------------------------------------------

def train(hp:               HParams,
          robot:            Robot_math,
          device:           torch.device,
          check_collisions: bool = False,
          trial:            Optional[optuna.Trial] = None,
          verbose:          bool = True) -> float:

    K         = hp.n_envs
    n_joints  = len(robot.a)
    state_dim = 3 + 6 + n_joints

    reward_math = Reward_Math(
        my_robot   = robot,
        pos_w      = hp.pos_w,
        rot_w      = hp.rot_w,
        crash_w    = hp.crash_w,
        dist_w     = hp.dist_w,
        vel_lambda = [hp.vel_lambda] * n_joints,
        acc_lambda = [hp.acc_lambda] * n_joints,
    )
    buffer = Replay_Buffer(
        buffer_size     = hp.buffer_size,
        reward_math     = reward_math,
        device          = device,
        state_dim       = state_dim,
        action_dim      = n_joints,
        her_ratio_start = hp.her_ratio_start,
        her_ratio_end   = hp.her_ratio_end,
        her_decay_steps = hp.her_decay_steps,
    )
    td3 = TD3(
        state_dim    = state_dim,
        action_dim   = n_joints,
        hidden_width = hp.hidden_width,
        hidden_depth = hp.hidden_depth,
        actor_lr     = hp.actor_lr,
        critic_lr    = hp.critic_lr,
        dtype        = torch.float32,
    )

    env = VecEnvState(K=K, n=n_joints, device=robot.device)
    if check_collisions:
        q_s = sample_random_q_collision_free(robot, K)
        q_g = sample_random_q_collision_free(robot, K)
    else:
        q_s = sample_random_q(robot, K)
        q_g = sample_random_q(robot, K)

    env.q_curr   = q_s
    p_s, R_s     = batch_ee_pose(robot, q_s)
    p_g, R_g     = batch_ee_pose(robot, q_g)
    env.pos_curr = p_s;  env.R_curr = R_s
    env.pos_goal = p_g;  env.R_goal = R_g

    episode_log       = []
    total_steps       = 0
    update_step       = 0
    eval_start_n      = int(hp.n_episodes * 0.90)
    total_step_budget = hp.n_episodes * hp.max_steps
    log_interval      = max(1, hp.n_episodes // 100)
    t0                = time.time()

    # ── Dedicated warmup phase ────────────────────────────────────────────
    # Run uniform random actions across all K environments until the replay
    # buffer holds warmup_transitions entries.  No TD3 updates happen here.
    # HER ratio is frozen at her_ratio_start for the entire warmup so the
    # episode counter (which drives HER decay) doesn't start until training.
    warmup_log = []   # warmup episodes tracked separately, not in episode_log
    if hp.warmup_transitions > 0:
        print(f"[Warmup] Filling buffer to {hp.warmup_transitions:,} transitions "
              f"with uniform random actions across {K} envs...", flush=True)
        warmup_steps = 0
        while len(buffer) < hp.warmup_transitions:
            total_steps, update_step = vectorised_step(
                env=env, robot=robot, reward_math=reward_math,
                td3=td3, buffer=buffer, hp=hp,
                total_steps=total_steps, update_step=update_step,
                check_collisions=check_collisions,
                current_sigma=hp.expl_sigma,
                episode_log=warmup_log,
                uniform_random=True,
                do_updates=False,      # no gradient updates during warmup
            )
            warmup_steps += K
            # Overwrite-in-place progress line
            print(f"\r[Warmup] {len(buffer):>9,} / {hp.warmup_transitions:,} transitions | "
                  f"{len(warmup_log)} episodes | {warmup_steps} env-steps",
                  end="", flush=True)
        print(f"\n[Warmup] Complete — {len(buffer):,} transitions, "
              f"{len(warmup_log)} episodes.", flush=True)

    # ── Main training loop ────────────────────────────────────────────────
    while len(episode_log) < hp.n_episodes:
        if hp.decay_expl_noise:
            frac  = min(total_steps / max(total_step_budget, 1), 1.0)
            sigma = hp.expl_sigma + frac * (hp.expl_sigma_end - hp.expl_sigma)
        else:
            sigma = hp.expl_sigma

        total_steps, update_step = vectorised_step(
            env=env, robot=robot, reward_math=reward_math,
            td3=td3, buffer=buffer, hp=hp,
            total_steps=total_steps, update_step=update_step,
            check_collisions=check_collisions,
            current_sigma=sigma, episode_log=episode_log,
            uniform_random=False,
            do_updates=True,
        )

        n_ep = len(episode_log)
        if verbose and n_ep > 0 and n_ep % log_interval == 0:
            window    = episode_log[max(0, n_ep - log_interval):]
            sr        = sum(e["success"] for e in window) / len(window) * 100
            cr        = sum(e["crashed"] for e in window) / len(window) * 100
            avg_r     = sum(e["total_reward"] for e in window) / len(window)
            avg_steps = sum(e["steps"] for e in window) / len(window)
            elapsed   = time.time() - t0
            warmup    = ""   # warmup is a separate phase now
            # \r overwrites the current line — 100 updates stay as one line.
            # end="" suppresses newline; flush=True forces immediate display.
            # A newline is printed only at the final episode so the last
            # result is preserved in the terminal after training ends.
            avg_pos = sum(e["final_pos_err"] for e in window) / len(window)
            avg_ori = sum(e["final_ori_err"] for e in window) / len(window)
            mae     = avg_pos + 0.1 * avg_ori
            final   = (n_ep >= hp.n_episodes)
            print(f"\r  ep {n_ep:6d}/{hp.n_episodes} | "
                  f"SR {sr:5.1f}% | MAE {mae:.4f} | AvgR {avg_r:7.2f} | "
                  f"Steps {avg_steps:5.1f} | sigma {sigma:.3f} | "
                  f"HER {buffer.her_ratio:.2f} | Buf {len(buffer):7d} | "
                  f"{elapsed:.0f}s{warmup}",
                  end="\n" if final else "", flush=True)

        if trial is not None and n_ep >= eval_start_n and n_ep % log_interval == 0:
            eval_window  = episode_log[eval_start_n:]
            n_eval       = max(len(eval_window), 1)
            # MAE objective: mean(pos_err + ori_scale * ori_err) over last 10%.
            # ori_scale=0.1 balances Frobenius norm units against normalised
            # position error units. Both are minimised — smaller is better.
            curr_mae = (sum(e["final_pos_err"] + 0.1 * e["final_ori_err"]
                            for e in eval_window) / n_eval)
            trial.report(curr_mae, step=n_ep)
            if trial.should_prune():
                raise optuna.TrialPruned()

    eval_log    = episode_log[eval_start_n:]
    n_eval      = max(len(eval_log), 1)
    final_mae   = (sum(e["final_pos_err"] + 0.1 * e["final_ori_err"]
                       for e in eval_log) / n_eval)
    final_avg_r = sum(e["total_reward"] for e in eval_log) / n_eval
    final_sr    = sum(e["success"]      for e in eval_log) / n_eval
    if verbose:
        print(f"\n  MAE (last 10%): {final_mae:.4f} | "
              f"Avg reward: {final_avg_r:.2f} | "
              f"SR: {final_sr*100:.2f}%")
    return final_mae   # Optuna objective: minimise MAE (pos + ori error)


# ---------------------------------------------------------------------------
# Optuna
# ---------------------------------------------------------------------------

def suggest_hparams(trial: optuna.Trial,
                    base:  HParams,
                    flags: TuneConfig) -> HParams:
    hp = copy.deepcopy(base)
    if flags.gamma:           hp.gamma           = trial.suggest_float("gamma", 0.95, 0.999)
    if flags.tau:             hp.tau             = trial.suggest_float("tau", 1e-3, 1e-1, log=True)
    if flags.actor_lr:        hp.actor_lr        = trial.suggest_float("actor_lr", 1e-5, 1e-3, log=True)
    if flags.critic_lr:       hp.critic_lr       = trial.suggest_float("critic_lr", 1e-4, 1e-2, log=True)
    if flags.hidden_width:    hp.hidden_width     = trial.suggest_categorical("hidden_width", [128, 256, 512])
    if flags.hidden_depth:    hp.hidden_depth     = trial.suggest_int("hidden_depth", 1, 4)
    if flags.n_envs:          hp.n_envs           = trial.suggest_categorical("n_envs", [4, 8, 16, 32])
    if flags.pos_w:           hp.pos_w            = trial.suggest_float("pos_w", 1.0, 100.0, log=True)
    if flags.rot_w:           hp.rot_w            = trial.suggest_float("rot_w", 1.0, 10.0, log=True)
    if flags.vel_lambda:      hp.vel_lambda       = trial.suggest_float("vel_lambda", 0.01, 0.5, log=True)
    if flags.acc_lambda:      hp.acc_lambda       = trial.suggest_float("acc_lambda", 0.01, 0.5, log=True)
    if flags.her_ratio_start: hp.her_ratio_start  = trial.suggest_float("her_ratio_start", 0.4, 0.95)
    if flags.her_decay_steps: hp.her_decay_steps  = trial.suggest_int("her_decay_steps", 200, 5_000, log=True)
    if flags.expl_sigma:      hp.expl_sigma       = trial.suggest_float("expl_sigma", 0.05, 0.3)
    if flags.expl_sigma_end:
        hp.expl_sigma_end = trial.suggest_float("expl_sigma_end", 0.005, 0.1, log=True)
        hp.expl_sigma_end = min(hp.expl_sigma_end, hp.expl_sigma * 0.9)
    if flags.target_sigma:    hp.target_sigma     = trial.suggest_float("target_sigma", 0.1, 0.4)
    return hp


def run_tuning(base_hp:          HParams,
               tune_flags:       TuneConfig,
               n_trials:         int  = 50,
               parallel_mode:    str  = "gpu",
               n_jobs:           int  = 1,
               check_collisions: bool = False,
               storage_path:     str  = "optuna_study.db",
               params_out:       str  = "best_params.json"):

    storage_url = f"sqlite:///{storage_path}"
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials = 5,
        n_warmup_steps   = int(base_hp.n_episodes * 0.5),
        interval_steps   = max(1, base_hp.n_episodes // 100),
    )
    study = optuna.create_study(
        study_name="td3_ik_tuning", direction="minimize",
        storage=storage_url, load_if_exists=True, pruner=pruner,
    )
    device   = (torch.device("cuda" if torch.cuda.is_available() else "cpu")
                if parallel_mode == "gpu" else torch.device("cpu"))
    eff_jobs = 1 if parallel_mode == "gpu" else n_jobs
    print(f"[Optuna] mode={parallel_mode}  device={device}  n_jobs={eff_jobs}")

    def objective(trial):
        hp    = suggest_hparams(trial, base_hp, tune_flags)
        robot = make_robot(device)
        print(f"\n[Trial {trial.number}] "
              + "  ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
                          for k, v in trial.params.items()))
        return train(hp=hp, robot=robot, device=device,
                     check_collisions=check_collisions, trial=trial, verbose=True)

    study.optimize(objective, n_trials=n_trials, n_jobs=eff_jobs)

    print("\n" + "=" * 60)
    best = study.best_trial
    print(f"Best MAE (last 10%): {best.value:.4f}")
    for k, v in best.params.items():
        print(f"  {k:20s} = {v}")

    # Reconstruct best HParams and save to JSON
    best_hp = suggest_hparams(best, base_hp, tune_flags)
    save_params(best_hp, params_out)
    print(f"\nDashboard: optuna-dashboard {storage_url}")
    return study


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

def launch_dashboard(storage_path: str = "optuna_study.db", port: int = 8080):
    storage_url = f"sqlite:///{storage_path}"
    print(f"[Dashboard] http://localhost:{port}  (Ctrl-C to stop)")
    optuna_dashboard.run_server(storage=storage_url, port=port)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="TD3 IK Agent",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--mode", choices=["train", "tune", "dashboard"], default="train")
    p.add_argument("--load_params",      type=str, default=None,
                   help="JSON file of saved hyperparameters (from tuning)")
    p.add_argument("--params_out",       type=str, default="best_params.json",
                   help="Where to save best params after tuning")
    p.add_argument("--parallel_mode",    choices=["gpu", "cpu"], default="gpu")
    p.add_argument("--n_jobs",           type=int, default=4)
    p.add_argument("--check_collisions", action="store_true")
    p.add_argument("--n_episodes",          type=int,   default=10_000)
    p.add_argument("--max_steps",           type=int,   default=200)
    p.add_argument("--updates_per_episode", type=int,   default=50)
    p.add_argument("--batch_size",          type=int,   default=2048)
    p.add_argument("--n_envs",              type=int,   default=8)
    p.add_argument("--warmup_transitions",  type=int,   default=500_000,
                   help="Fill buffer to this many transitions before training")
    p.add_argument("--decay_expl_noise",    action="store_true")
    p.add_argument("--n_trials",       type=int, default=50)
    p.add_argument("--storage_path",   type=str, default="optuna_study.db")
    p.add_argument("--dashboard_port", type=int, default=8080)

    tuneable = ["gamma", "tau", "actor_lr", "critic_lr", "hidden_width",
                "hidden_depth", "n_envs", "pos_w", "rot_w", "vel_lambda",
                "acc_lambda", "her_ratio_start", "her_decay_steps",
                "expl_sigma", "expl_sigma_end", "target_sigma"]
    for name in tuneable:
        p.add_argument(f"--no_tune_{name}", action="store_true",
                       help=f"Fix {name} at its HParams default")
    return p.parse_args()


def main():
    args = parse_args()

    # Build base HParams from CLI args
    hp = HParams(
        n_episodes          = args.n_episodes,
        max_steps           = args.max_steps,
        updates_per_episode = args.updates_per_episode,
        batch_size          = args.batch_size,
        n_envs              = args.n_envs,
        warmup_transitions  = args.warmup_transitions,
        decay_expl_noise    = args.decay_expl_noise,
    )

    # Override with saved params if --load_params provided
    if args.load_params:
        hp = load_params(args.load_params, hp)

    # Build TuneConfig
    tc = TuneConfig()
    tuneable = ["gamma", "tau", "actor_lr", "critic_lr", "hidden_width",
                "hidden_depth", "n_envs", "pos_w", "rot_w", "vel_lambda",
                "acc_lambda", "her_ratio_start", "her_decay_steps",
                "expl_sigma", "expl_sigma_end", "target_sigma"]
    for name in tuneable:
        if getattr(args, f"no_tune_{name}", False):
            setattr(tc, name, False)

    if args.mode == "dashboard":
        launch_dashboard(args.storage_path, args.dashboard_port)

    elif args.mode == "train":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Train] device={device}  n_envs={hp.n_envs}  "
              f"check_collisions={args.check_collisions}")
        robot = make_robot(device)
        train(hp=hp, robot=robot, device=device,
              check_collisions=args.check_collisions, verbose=True)

    elif args.mode == "tune":
        run_tuning(
            base_hp          = hp,
            tune_flags       = tc,
            n_trials         = args.n_trials,
            parallel_mode    = args.parallel_mode,
            n_jobs           = args.n_jobs,
            check_collisions = args.check_collisions,
            storage_path     = args.storage_path,
            params_out       = args.params_out,
        )


if __name__ == "__main__":
    main()