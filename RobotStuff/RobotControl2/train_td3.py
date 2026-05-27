"""
Training script for the TD3 Robot Arm IK Agent — vectorised K-environment loop.

────────────────────────────────────────────────────────────────────────────
Usage
────────────────────────────────────────────────────────────────────────────

  # Plain training run:
  python train.py --mode train

  # Optuna hyperparameter search (sequential GPU trials):
  python train.py --mode tune --parallel_mode gpu

  # Optuna search (parallel CPU trials):
  python train.py --mode tune --parallel_mode cpu --n_jobs 4

  # Launch the Optuna dashboard:
  python train.py --mode dashboard

────────────────────────────────────────────────────────────────────────────
Vectorised environment design
────────────────────────────────────────────────────────────────────────────

Instead of running one episode at a time, K environments run in lockstep.
Every "step" of the outer loop:

  1. One batched actor forward pass over K states  →  (K, n) actions.
  2. One batched FK call  →  (K, 3) pos_next, (K, 3, 3) R_next.
  3. Batched reward computation using reward_batch().
  4. Per-environment done check (success / crash / timeout).
  5. Any finished environment immediately resets:
       - its episode cache is committed to the buffer via finish_episode()
       - new collision-free start and goal are sampled
  6. After every step, run updates_per_episode/K TD3 gradient updates
     (amortised so total updates per "episode equivalent" stays constant).

This turns K sequential FK calls into one batched bmm, K sequential
actor forward passes into one GPU matrix multiply, etc.  The speedup
is roughly K× for FK and actor inference.

Collision checking (--check_collisions flag)
  When enabled, start/goal sampling and per-step crash detection use
  do_fk_and_check_crash() sequentially (one Robot_math instance, one
  q_vect at a time).  This is the only part that stays sequential.
  When disabled (default), sampling uses uniform random joint angles
  without collision validation — appropriate when self-collisions are
  geometrically unlikely with your robot parameters.

float32 / float64 boundary
  FK and reward math use float64 for numerical precision.
  Network inputs/outputs and buffer storage are float32 for GPU speed.
  Conversion happens in vectorised_step() at the FK→network boundary.
"""

import argparse
import copy
import time
from dataclasses import dataclass
from typing import Optional

import torch
import optuna
import optuna_dashboard

from forward_kinematics import Robot_math
from rewards_math       import Reward_Math
from replay_buffer      import Replay_Buffer
from td3                import TD3

import numpy as np

# ---------------------------------------------------------------------------
# Robot definition  -- edit this block for your specific arm
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
    n_envs:              int   = 8       # parallel environments
    start_steps:         int   = 2_000   # uniform random steps before actor

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


@dataclass
class TuneConfig:
    gamma:           bool = True
    tau:             bool = True
    actor_lr:        bool = True
    critic_lr:       bool = True
    hidden_width:    bool = False
    hidden_depth:    bool = False
    n_envs:          bool = False   # architecture-level; off by default
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
# Pose sampling helpers
# ---------------------------------------------------------------------------

def sample_random_q(robot: Robot_math, K: int) -> torch.Tensor:
    """
    Sample K joint configurations uniformly within joint limits.
    No collision checking — pure GPU tensor op.
    Returns (K, n) float64 on robot.device.
    """
    low  = robot.low_bounds.unsqueeze(0)    # (1, n)
    high = robot.high_bounds.unsqueeze(0)   # (1, n)
    return low + (high - low) * torch.rand(
        K, len(robot.a), dtype=torch.float64, device=robot.device)


def sample_random_q_collision_free(robot: Robot_math,
                                    K: int,
                                    max_attempts: int = 200) -> torch.Tensor:
    """
    Sample K collision-free joint configurations sequentially.
    Used only when --check_collisions is set.
    Returns (K, n) float64 on robot.device.
    """
    results = []
    while len(results) < K:
        q = (robot.low_bounds
             + (robot.high_bounds - robot.low_bounds)
             * torch.rand(len(robot.a), dtype=torch.float64, device=robot.device))
        robot.q_vect = q
        crash, _ = robot.do_fk_and_check_crash()
        if not crash:
            results.append(q)
    return torch.stack(results)   # (K, n)


def batch_ee_pose(robot: Robot_math, q_batch: torch.Tensor):
    """
    Batched EE pose from joint configs.
    q_batch : (K, n)
    returns : pos (K, 3) float64,  R (K, 3, 3) float64
    """
    pos = robot.give_ds_batch(q_batch)    # (K, 3)
    R   = robot.give_Rs_batch(q_batch)    # (K, 3, 3)
    return pos, R


# ---------------------------------------------------------------------------
# Vectorised environment state container
# ---------------------------------------------------------------------------

class VecEnvState:
    """
    Holds the mutable per-environment state for K parallel environments.
    All tensors are float64 on robot.device (FK precision).
    """
    def __init__(self, K: int, n: int, device: torch.device):
        self.K      = K
        self.n      = n
        self.device = device

        # Current joint config, EE pose
        self.q_curr    = torch.zeros(K, n,    dtype=torch.float64, device=device)
        self.pos_curr  = torch.zeros(K, 3,    dtype=torch.float64, device=device)
        self.R_curr    = torch.zeros(K, 3, 3, dtype=torch.float64, device=device)

        # Goal pose
        self.pos_goal  = torch.zeros(K, 3,    dtype=torch.float64, device=device)
        self.R_goal    = torch.zeros(K, 3, 3, dtype=torch.float64, device=device)

        # Previous action (for acceleration reward)
        self.delta_q_prev = torch.zeros(K, n, dtype=torch.float64, device=device)

        # Per-environment step counter
        self.steps     = torch.zeros(K, dtype=torch.long, device=device)

        # Per-environment episode caches (list of K lists)
        self.caches: list[list[dict]] = [[] for _ in range(K)]

        # Logging accumulators (reset per episode, tracked per env)
        self.ep_reward  = torch.zeros(K, dtype=torch.float64, device=device)
        self.ep_success = [False] * K
        self.ep_grade   = ["FAIL"] * K
        self.ep_crashed = [False] * K


# ---------------------------------------------------------------------------
# Vectorised step
# ---------------------------------------------------------------------------

def vectorised_step(
        env:          VecEnvState,
        robot:        Robot_math,
        reward_math:  Reward_Math,
        td3:          TD3,
        buffer:       Replay_Buffer,
        hp:           HParams,
        total_steps:  int,
        update_step:  int,
        check_collisions: bool,
        current_sigma: float,
        episode_log:  list,
) -> tuple[int, int]:
    """
    Advance all K environments by one step.

    Returns (total_steps, update_step) updated.
    """
    K = env.K
    n = env.n
    device = robot.device

    # ------------------------------------------------------------------
    # 1. Build batched state  (K, state_dim)  float64 -> float32 for net
    # ------------------------------------------------------------------
    # Normalised displacement to goal per env: (K, 3)
    norm_dist = (env.pos_goal - env.pos_curr) / reward_math.rob.max_reach

    # 6D rotation representation of goal per env: (K, 6)
    rot_6d = torch.cat([env.R_goal[:, :, 0],
                         env.R_goal[:, :, 1]], dim=1)          # (K, 6)

    # Previous normalised velocity: (K, n)
    vel_low  = robot.low_bounds  - robot.high_bounds            # (n,)
    vel_high = robot.high_bounds - robot.low_bounds             # (n,)
    denom    = (vel_high - vel_low).clamp(min=1e-8)
    prev_vel_norm = 2.0 * ((env.delta_q_prev - vel_low) / denom) - 1.0  # (K, n)

    state_f64 = torch.cat([norm_dist, rot_6d, prev_vel_norm], dim=1)    # (K, S)
    state_f32 = state_f64.to(dtype=torch.float32, device=td3.device)    # GPU f32

    # ------------------------------------------------------------------
    # 2. Select actions  (K, n)
    # ------------------------------------------------------------------
    uniform_random = (total_steps < hp.start_steps)

    if uniform_random:
        a_norm_f32 = torch.rand(K, n, dtype=torch.float32, device=td3.device) * 2 - 1
    else:
        with torch.no_grad():
            a_norm_f32 = td3.actor(state_f32)                  # (K, n)
        # Add clipped Gaussian exploration noise
        noise = torch.randn_like(a_norm_f32) * current_sigma
        noise = noise.clamp(-hp.expl_clip, hp.expl_clip)
        a_norm_f32 = (a_norm_f32 + noise).clamp(-1.0, 1.0)

    a_norm_f64 = a_norm_f32.to(dtype=torch.float64, device=device)

    # ------------------------------------------------------------------
    # 3. Apply actions to get new joint configs  (K, n)
    # ------------------------------------------------------------------
    # Normalise current q to [-1, 1], add normalised delta, clamp, denorm
    q_range = robot.high_bounds - robot.low_bounds              # (n,)
    q_norm  = 2.0 * ((env.q_curr - robot.low_bounds) / q_range.clamp(min=1e-8)) - 1.0
    q_new_norm = (q_norm + a_norm_f64).clamp(-1.0, 1.0)
    q_new  = robot.low_bounds + (q_new_norm + 1.0) * q_range / 2.0  # (K, n)

    # Denormalise action to real velocity units (for reward/buffer)
    delta_q_new = reward_math.denormalize_from_range(
        a_norm_f64, vel_low.unsqueeze(0).expand(K, -1),
        vel_high.unsqueeze(0).expand(K, -1))                    # (K, n)

    # ------------------------------------------------------------------
    # 4. Batched FK  →  next EE pose
    # ------------------------------------------------------------------
    pos_next, R_next = batch_ee_pose(robot, q_new)              # (K,3), (K,3,3)

    # ------------------------------------------------------------------
    # 5. Batched reward
    # ------------------------------------------------------------------
    r_batch, _ = reward_math.reward_batch(
        pos_curr_batch     = env.pos_curr,
        R_curr_SO3_batch   = env.R_curr,
        delta_q_new_batch  = delta_q_new,
        delta_q_prev_batch = env.delta_q_prev,
        pos_goal_batch     = env.pos_goal,
        R_goal_SO3_batch   = env.R_goal,
        use_focal          = True,
    )                                                            # (K,) float64

    # ------------------------------------------------------------------
    # 6. Collision check (sequential, optional)
    # ------------------------------------------------------------------
    crashed_k = [False] * K
    if check_collisions:
        for k in range(K):
            robot.q_vect = q_new[k]
            crash, _ = robot.do_fk_and_check_crash()
            if crash:
                crashed_k[k] = True
                r_batch[k]  -= hp.crash_w

    # ------------------------------------------------------------------
    # 7. Done flags  (success / crash / timeout)
    # ------------------------------------------------------------------
    # Recompute per-env errors to classify grade
    eps_norm  = (env.pos_goal - pos_next) / reward_math.rob.max_reach   # (K,3)
    e_pos_raw = torch.linalg.vector_norm(eps_norm, dim=1)               # (K,)
    ori_diff  = env.R_goal - R_next
    e_ori_raw = torch.linalg.matrix_norm(ori_diff)                      # (K,)

    good_pos  = reward_math._good_pos_thresh
    good_ori  = reward_math._good_ori_thresh
    success_k = ((reward_math.pos_w * e_pos_raw <= good_pos) &
                 (reward_math.rot_w * e_ori_raw <= good_ori))           # (K,) bool

    env.steps += 1
    timeout_k = (env.steps >= hp.max_steps)
    done_k    = success_k | timeout_k | torch.tensor(
                    crashed_k, dtype=torch.bool, device=device)

    env.ep_reward += r_batch

    # ------------------------------------------------------------------
    # 8. Cache steps and commit finished episodes
    # ------------------------------------------------------------------
    for k in range(K):
        done = done_k[k].item()

        env.caches[k].append({
            "q_new":        q_new[k].cpu(),
            "delta_q_new":  delta_q_new[k].cpu(),
            "delta_q_prev": env.delta_q_prev[k].cpu(),
            "pos_curr":     env.pos_curr[k].cpu(),
            "R_curr":       env.R_curr[k].cpu(),
            "pos_next":     pos_next[k].cpu(),
            "R_next":       R_next[k].cpu(),
            "a_norm":       a_norm_f64[k].cpu(),
            "done":         bool(done),
        })

        if done:
            # Commit episode to buffer
            buffer._episode_cache = env.caches[k]
            buffer.finish_episode(
                pos_goal   = env.pos_goal[k].cpu(),
                R_goal_SO3 = env.R_goal[k].cpu(),
            )

            # Log episode
            grade = "GOOD" if success_k[k] else ("CRASH" if crashed_k[k] else "FAIL")
            episode_log.append({
                "total_reward": env.ep_reward[k].item(),
                "steps":        env.steps[k].item(),
                "success":      success_k[k].item(),
                "grade":        grade,
                "crashed":      crashed_k[k],
            })

            # Reset this environment
            env.caches[k]     = []
            env.ep_reward[k]  = 0.0
            env.steps[k]      = 0
            env.ep_success[k] = False
            env.ep_grade[k]   = "FAIL"
            env.ep_crashed[k] = False

            # Sample new start and goal
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

    # ------------------------------------------------------------------
    # 9. Advance state for non-done environments
    # ------------------------------------------------------------------
    alive = ~done_k                                             # (K,) bool
    if alive.any():
        env.q_curr[alive]       = q_new[alive]
        env.pos_curr[alive]     = pos_next[alive]
        env.R_curr[alive]       = R_next[alive]
        env.delta_q_prev[alive] = delta_q_new[alive]

    total_steps += K

    # ------------------------------------------------------------------
    # 10. TD3 updates (amortised per step)
    # ------------------------------------------------------------------
    # We want updates_per_episode updates per "episode equivalent".
    # An episode equivalent = max_steps steps.
    # Per step we do updates_per_episode / max_steps updates on average.
    # We accumulate a fractional counter and do integer updates.
    updates_this_step = int(hp.updates_per_episode * K / hp.max_steps)
    if updates_this_step < 1 and (total_steps % max(1, hp.max_steps // hp.updates_per_episode)) == 0:
        updates_this_step = 1

    if total_steps >= hp.start_steps and len(buffer) >= hp.batch_size:
        for _ in range(updates_this_step):
            batch = buffer.sample(hp.batch_size)
            if batch is None:
                break
            td3.train_step(
                batch        = batch,    # already float32 on device from buffer
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
    """
    Full training run.  Returns success rate over the last 10% of episodes.
    """
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

    # Initialise vectorised environment state
    env = VecEnvState(K=K, n=n_joints, device=robot.device)

    if check_collisions:
        q_starts = sample_random_q_collision_free(robot, K)
        q_goals  = sample_random_q_collision_free(robot, K)
    else:
        q_starts = sample_random_q(robot, K)
        q_goals  = sample_random_q(robot, K)

    env.q_curr   = q_starts
    p_s, R_s     = batch_ee_pose(robot, q_starts)
    p_g, R_g     = batch_ee_pose(robot, q_goals)
    env.pos_curr = p_s
    env.R_curr   = R_s
    env.pos_goal = p_g
    env.R_goal   = R_g

    episode_log  = []
    total_steps  = 0
    update_step  = 0
    eval_start_n = int(hp.n_episodes * 0.90)
    t0           = time.time()

    # Total steps to run ≈ n_episodes × max_steps (each env contributes)
    total_step_budget = hp.n_episodes * hp.max_steps

    log_interval = max(1, hp.n_episodes // 100)   # log ~100 times per run

    while len(episode_log) < hp.n_episodes:

        # Current exploration sigma
        if hp.decay_expl_noise and total_steps >= hp.start_steps:
            frac = min(total_steps / max(total_step_budget, 1), 1.0)
            current_sigma = hp.expl_sigma + frac * (hp.expl_sigma_end - hp.expl_sigma)
        else:
            current_sigma = hp.expl_sigma

        total_steps, update_step = vectorised_step(
            env=env, robot=robot, reward_math=reward_math,
            td3=td3, buffer=buffer, hp=hp,
            total_steps=total_steps, update_step=update_step,
            check_collisions=check_collisions,
            current_sigma=current_sigma,
            episode_log=episode_log,
        )

        n_ep = len(episode_log)

        # Logging
        if verbose and n_ep > 0 and n_ep % log_interval == 0:
            window    = episode_log[max(0, n_ep - log_interval):]
            sr        = sum(e["success"] for e in window) / len(window) * 100
            cr        = sum(e["crashed"] for e in window) / len(window) * 100
            avg_r     = sum(e["total_reward"] for e in window) / len(window)
            avg_steps = sum(e["steps"] for e in window) / len(window)
            elapsed   = time.time() - t0
            warmup    = " [warmup]" if total_steps < hp.start_steps else ""
            print(f"  ep {n_ep:6d}/{hp.n_episodes} | "
                  f"SR {sr:5.1f}% | CR {cr:4.1f}% | "
                  f"AvgR {avg_r:7.2f} | Steps {avg_steps:5.1f} | "
                  f"sigma {current_sigma:.3f} | "
                  f"HER {buffer.her_ratio:.2f} | Buf {len(buffer):7d} | "
                  f"{elapsed:.0f}s{warmup}")

        # Optuna pruning
        if trial is not None and n_ep >= eval_start_n and n_ep % log_interval == 0:
            n_eval   = n_ep - eval_start_n
            curr_sr  = sum(e["success"] for e in episode_log[eval_start_n:]) / max(n_eval, 1)
            trial.report(curr_sr, step=n_ep)
            if trial.should_prune():
                raise optuna.TrialPruned()

    eval_log = episode_log[eval_start_n:]
    final_sr = sum(e["success"] for e in eval_log) / max(len(eval_log), 1)
    if verbose:
        print(f"\n  Final success rate (last 10%): {final_sr * 100:.2f}%")
    return final_sr


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
    if flags.pos_w:           hp.pos_w            = trial.suggest_float("pos_w", 0.5, 3.0)
    if flags.rot_w:           hp.rot_w            = trial.suggest_float("rot_w", 0.1, 2.0)
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


def run_tuning(base_hp:       HParams,
               tune_flags:    TuneConfig,
               n_trials:      int = 50,
               parallel_mode: str = "gpu",
               n_jobs:        int = 1,
               check_collisions: bool = False,
               storage_path:  str = "optuna_study.db"):
    storage_url = f"sqlite:///{storage_path}"
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials = 5,
        n_warmup_steps   = int(base_hp.n_episodes * 0.5),
        interval_steps   = max(1, base_hp.n_episodes // 100),
    )
    study = optuna.create_study(
        study_name="td3_ik_tuning", direction="maximize",
        storage=storage_url, load_if_exists=True, pruner=pruner,
    )
    if parallel_mode == "gpu":
        device, eff_jobs = torch.device("cuda" if torch.cuda.is_available() else "cpu"), 1
        print(f"[Optuna] Sequential GPU trials  device={device}")
    else:
        device, eff_jobs = torch.device("cpu"), n_jobs
        print(f"[Optuna] Parallel CPU trials  n_jobs={eff_jobs}")

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
    print(f"Best SR: {best.value*100:.2f}%")
    for k, v in best.params.items():
        print(f"  {k:20s} = {v}")
    print(f"\nDashboard: optuna-dashboard {storage_url}")
    return study


def launch_dashboard(storage_path: str = "optuna_study.db", port: int = 8080):
    storage_url = f"sqlite:///{storage_path}"
    print(f"[Dashboard] http://localhost:{port}  (Ctrl-C to stop)")
    optuna_dashboard.run_server(storage=storage_url, port=port)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="TD3 IK — vectorised training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--mode", choices=["train", "tune", "dashboard"], default="train")
    p.add_argument("--parallel_mode", choices=["gpu", "cpu"], default="gpu")
    p.add_argument("--n_jobs",             type=int,  default=4)
    p.add_argument("--check_collisions",   action="store_true",
                   help="Enable sequential self-collision checking (slower)")
    p.add_argument("--n_episodes",          type=int,   default=10_000)
    p.add_argument("--max_steps",           type=int,   default=200)
    p.add_argument("--updates_per_episode", type=int,   default=50)
    p.add_argument("--batch_size",          type=int,   default=2048)
    p.add_argument("--n_envs",              type=int,   default=8)
    p.add_argument("--start_steps",         type=int,   default=2_000)
    p.add_argument("--decay_expl_noise",    action="store_true")
    p.add_argument("--n_trials",       type=int, default=50)
    p.add_argument("--storage_path",   type=str, default="optuna_study.db")
    p.add_argument("--dashboard_port", type=int, default=8080)

    tuneable = ["gamma", "tau", "actor_lr", "critic_lr", "hidden_width",
                "hidden_depth", "n_envs", "pos_w", "rot_w", "vel_lambda",
                "acc_lambda", "her_ratio_start", "her_decay_steps",
                "expl_sigma", "expl_sigma_end", "target_sigma"]
    for name in tuneable:
        p.add_argument(f"--no_tune_{name}", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    hp = HParams(
        n_episodes          = args.n_episodes,
        max_steps           = args.max_steps,
        updates_per_episode = args.updates_per_episode,
        batch_size          = args.batch_size,
        n_envs              = args.n_envs,
        start_steps         = args.start_steps,
        decay_expl_noise    = args.decay_expl_noise,
    )
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
        print(f"[Train] device={device}  n_envs={hp.n_envs}  check_collisions={args.check_collisions}")
        robot = make_robot(device)
        train(hp=hp, robot=robot, device=device,
              check_collisions=args.check_collisions, verbose=True)
    elif args.mode == "tune":
        run_tuning(base_hp=hp, tune_flags=tc, n_trials=args.n_trials,
                   parallel_mode=args.parallel_mode, n_jobs=args.n_jobs,
                   check_collisions=args.check_collisions,
                   storage_path=args.storage_path)


if __name__ == "__main__":
    main()