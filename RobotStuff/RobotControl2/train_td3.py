"""
Training script for the TD3 Robot Arm IK Agent.

────────────────────────────────────────────────────────────────────────────
Usage
────────────────────────────────────────────────────────────────────────────

  # Plain training run (fixed hyperparameters):
  python train.py --mode train

  # Optuna hyperparameter search (sequential GPU trials):
  python train.py --mode tune --parallel_mode gpu

  # Optuna search (parallel CPU trials):
  python train.py --mode tune --parallel_mode cpu --n_jobs 4

  # Launch the Optuna dashboard after / during a study:
  python train.py --mode dashboard

────────────────────────────────────────────────────────────────────────────
Design notes
────────────────────────────────────────────────────────────────────────────

Episode flow
  1. Sample a collision-free random start pose.
  2. Sample a collision-free random goal pose (FK gives the goal EE pose).
  3. Step the actor until timeout / crash / GOOD-grade success.
  4. Call buffer.finish_episode() -- commits real + HER transitions.
  5. Run updates_per_episode TD3 train steps (once buffer is warmed up).

Optuna objective
  Success rate over the last 10% of episodes in a trial.
  Each trial is a full independent training run from scratch.

float32 / float64 boundary
  Robot FK and reward math use float64 for numerical precision.
  All network inputs / outputs are cast to float32 for GPU efficiency.
  Cast back to float64 before passing to Reward_Math or Robot_math.
"""

import argparse
import copy
import time
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import optuna
import optuna_dashboard

from forward_kinematics import Robot_math
from rewards_math       import Reward_Math
from replay_buffer      import Replay_Buffer
from td3                import TD3


# ---------------------------------------------------------------------------
# Robot definition  -- edit this block for your specific arm
# ---------------------------------------------------------------------------

def make_robot(device) -> Robot_math:
    """
    Instantiate your Robot_math here.
    Edit the DH parameters, bounds, and capsule radii to match your arm.
    """
    a     = [0,    0.425, 0.3922, 0,      0,      0     ]
    alpha = [1.5708, 0,   0,      1.5708, -1.5708, 0     ]
    d     = [0.1625, 0,   0,      0.1333, 0.0997,  0.0996]
    theta = [0,    0,     0,      0,      0,       0     ]

    bounds = [
        (-6.2832, 6.2832),
        (-6.2832, 6.2832),
        (-3.1416, 3.1416),
        (-6.2832, 6.2832),
        (-6.2832, 6.2832),
        (-6.2832, 6.2832),
    ]

    fail_dist = [0.05] * 6
    pad_dist  = [0.08] * 6

    return Robot_math(
        a=a, alpha=alpha, d=d, theta=theta,
        joint_type=["r"] * 6,
        bounds=bounds,
        fail_dist=fail_dist,
        pad_dist=pad_dist,
        device=device,
    )


# ---------------------------------------------------------------------------
# Hyperparameter dataclass
# ---------------------------------------------------------------------------

@dataclass
class HParams:
    """
    All tuneable and fixed hyperparameters in one place.
    To fix a parameter (exclude from Optuna), set its flag to False
    in TuneConfig, or pass --no_tune_<name> on the CLI.
    """
    # Training loop
    n_episodes:          int   = 10_000
    max_steps:           int   = 200
    updates_per_episode: int   = 50
    batch_size:          int   = 256

    # Uniform random warmup: for this many total environment steps the actor
    # is bypassed and actions are sampled uniformly from [-1, 1].  TD3 updates
    # only begin after this threshold is crossed.
    # Rule of thumb: ~10 x max_steps (roughly 10 full random episodes).
    start_steps:         int   = 2_000

    # TD3
    gamma:               float = 0.99
    tau:                 float = 0.005
    actor_lr:            float = 1e-4
    critic_lr:           float = 1e-3
    policy_delay:        int   = 2
    hidden_width:        int   = 256
    hidden_depth:        int   = 2

    # Exploration noise
    expl_sigma:          float = 0.1    # initial sigma (and fixed value if decay is off)
    expl_sigma_end:      float = 0.02   # floor sigma reached at end of training
    expl_clip:           float = 0.5
    decay_expl_noise:    bool  = False  # True = linearly decay sigma over training

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
    """
    Controls which hyperparameters Optuna is allowed to tune.
    Set a flag to False to fix that parameter at its HParams default.
    Architecture flags (hidden_width, hidden_depth) are off by default
    since architecture search is expensive relative to its benefit.
    """
    gamma:           bool = True
    tau:             bool = True
    actor_lr:        bool = True
    critic_lr:       bool = True
    hidden_width:    bool = False
    hidden_depth:    bool = False
    pos_w:           bool = True
    rot_w:           bool = True
    vel_lambda:      bool = True
    acc_lambda:      bool = True
    her_ratio_start:  bool = True
    her_decay_steps:  bool = True
    expl_sigma:       bool = True
    expl_sigma_end:   bool = True   # only used when decay_expl_noise=True
    target_sigma:     bool = True


# ---------------------------------------------------------------------------
# Sampling helpers
# ---------------------------------------------------------------------------

def sample_collision_free_pose(robot: Robot_math,
                                max_attempts: int = 100) -> Optional[torch.Tensor]:
    """
    Sample a uniformly random joint configuration that is self-collision free.
    Returns the joint angle tensor (n,) float64, or None if max_attempts exhausted.
    """
    for _ in range(max_attempts):
        q = (robot.low_bounds
             + (robot.high_bounds - robot.low_bounds)
             * torch.rand(robot.low_bounds.shape,
                          dtype=torch.float64, device=robot.device))
        robot.q_vect = q
        crash, _ = robot.do_fk_and_check_crash()
        if not crash:
            return q
    return None


def get_ee_pose(robot: Robot_math):
    """Return (pos (3,) float64, R (3,3) float64) of the end-effector."""
    pos = robot.give_ds()[-1]
    R   = robot.give_Rs()[-1]
    return pos, R


# ---------------------------------------------------------------------------
# Action selection
# ---------------------------------------------------------------------------

def select_and_apply_action(td3:           TD3,
                             reward_math:   Reward_Math,
                             robot:         Robot_math,
                             state:         torch.Tensor,
                             expl_sigma:    float,
                             expl_clip:     float,
                             uniform_random: bool = False,
                             training:      bool = True):
    """
    Select an action, denormalise it to a real joint delta, and clamp.

    uniform_random=True  : bypass the actor entirely and sample a_norm
                           uniformly from [-1, 1]^n (warmup phase).
    uniform_random=False : use the actor with exploration noise (training)
                           or deterministically (evaluation).

    float32/float64 boundary:
      state (float64) -> float32 for the network -> float64 for robot math.

    Returns
    -------
    a_norm    : (n,) float64  normalised action in [-1, 1]
    delta_q   : (n,) float64  real joint velocity
    q_new     : (n,) float64  new joint angles, clamped to limits
    """
    n = len(robot.a)

    if uniform_random:
        # Truly uniform random action -- bypasses the actor entirely
        a_norm = (torch.rand(n, dtype=torch.float64, device=robot.device)
                  * 2.0 - 1.0)
    else:
        state_f32 = state.to(dtype=torch.float32, device=td3.device)
        if training:
            a_norm_f32 = td3.select_action(state_f32,
                                            expl_sigma=expl_sigma,
                                            expl_clip=expl_clip)
        else:
            with torch.no_grad():
                a_norm_f32 = td3.actor(state_f32)
        a_norm = a_norm_f32.to(dtype=torch.float64, device=robot.device)

    # Denormalise: [-1,1] -> joint velocity units
    vel_low  = robot.low_bounds  - robot.high_bounds
    vel_high = robot.high_bounds - robot.low_bounds
    delta_q  = reward_math.denormalize_from_range(a_norm, vel_low, vel_high)

    # Apply delta in normalised space, clamp, then recover real angles
    q_curr_norm = reward_math.get_normal_joint_value(robot.q_vect)
    q_new_norm  = (q_curr_norm + a_norm).clamp(-1.0, 1.0)
    q_new       = reward_math.get_original_joint_value(q_new_norm)

    return a_norm, delta_q, q_new


# ---------------------------------------------------------------------------
# Single episode
# ---------------------------------------------------------------------------

def run_episode(td3:            TD3,
                reward_math:    Reward_Math,
                robot:          Robot_math,
                buffer:         Replay_Buffer,
                hp:             HParams,
                pos_goal:       torch.Tensor,
                R_goal:         torch.Tensor,
                q_start:        torch.Tensor,
                uniform_random: bool  = False,
                current_sigma:  float = None,
                training:       bool  = True) -> dict:
    """
    Run one full episode and return a logging dict:
        total_reward, steps, success, grade, crashed

    uniform_random : if True all actions are sampled uniformly (warmup phase)
    current_sigma  : current exploration sigma (post-decay); uses hp.expl_sigma
                     if None
    """
    robot.q_vect = q_start.clone()
    pos_curr, R_curr = get_ee_pose(robot)

    n            = len(robot.a)
    delta_q_prev = torch.zeros(n, dtype=torch.float64, device=robot.device)

    reward_math.reset_episode()
    total_reward = 0.0
    success      = False
    grade        = "FAIL"
    crashed      = False

    sigma = current_sigma if current_sigma is not None else hp.expl_sigma

    for step in range(hp.max_steps):

        state = reward_math.build_state(
            pos_curr     = pos_curr,
            pos_goal     = pos_goal,
            R_goal_SO3   = R_goal,
            delta_q_prev = delta_q_prev,
        )

        a_norm, delta_q_new, q_new = select_and_apply_action(
            td3=td3, reward_math=reward_math, robot=robot,
            state=state, expl_sigma=sigma, expl_clip=hp.expl_clip,
            uniform_random=uniform_random, training=training,
        )

        robot.q_vect = q_new
        pos_next, R_next = get_ee_pose(robot)

        reward_math.reset_episode()
        r, info = reward_math.reward(
            q_new        = q_new,
            delta_q_new  = delta_q_new,
            delta_q_prev = delta_q_prev,
            pos_goal     = pos_goal,
            R_goal_SO3   = R_goal,
        )
        total_reward += r.item()

        crashed = info["crashed"]
        grade   = info["grade"]
        success = (grade == "GOOD")
        done    = success or crashed or (step == hp.max_steps - 1)

        if training:
            buffer.add_step(
                q_new        = q_new,
                delta_q_new  = delta_q_new,
                delta_q_prev = delta_q_prev,
                pos_curr     = pos_curr,
                R_curr       = R_curr,
                pos_next     = pos_next,
                R_next       = R_next,
                a_norm       = a_norm,
                done         = done,
            )

        pos_curr     = pos_next
        R_curr       = R_next
        delta_q_prev = delta_q_new

        if done:
            break

    if training:
        buffer.finish_episode(pos_goal=pos_goal, R_goal_SO3=R_goal)

    return {
        "total_reward": total_reward,
        "steps":        step + 1,
        "success":      success,
        "grade":        grade,
        "crashed":      crashed,
    }


# ---------------------------------------------------------------------------
# Full training run
# ---------------------------------------------------------------------------

def train(hp:      HParams,
          robot:   Robot_math,
          device:  torch.device,
          trial:   Optional[optuna.Trial] = None,
          verbose: bool = True) -> float:
    """
    Run a full training session.

    Returns the success rate over the last 10% of episodes.
    This is used as the Optuna objective when trial is not None.
    """
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

    log        = []
    eval_start = int(hp.n_episodes * 0.90)
    t0         = time.time()
    total_steps = 0   # global environment step counter across all episodes

    for ep in range(hp.n_episodes):

        # Sample collision-free start and goal poses
        q_start = sample_collision_free_pose(robot)
        if q_start is None:
            if verbose:
                print(f"  [ep {ep}] No collision-free start found -- skipping.")
            continue

        q_goal = sample_collision_free_pose(robot)
        if q_goal is None:
            if verbose:
                print(f"  [ep {ep}] No collision-free goal found -- skipping.")
            continue

        robot.q_vect = q_goal
        pos_goal, R_goal = get_ee_pose(robot)

        # Uniform random warmup: bypass actor for first start_steps steps
        uniform_random = (total_steps < hp.start_steps)

        # Exploration sigma: linear decay from expl_sigma -> expl_sigma_end
        # over the full training run, or fixed if decay is disabled.
        if hp.decay_expl_noise and not uniform_random:
            decay_frac    = min(total_steps / max(hp.n_episodes * hp.max_steps, 1), 1.0)
            current_sigma = (hp.expl_sigma
                             + decay_frac * (hp.expl_sigma_end - hp.expl_sigma))
        else:
            current_sigma = hp.expl_sigma

        ep_info = run_episode(
            td3=td3, reward_math=reward_math, robot=robot,
            buffer=buffer, hp=hp,
            pos_goal=pos_goal, R_goal=R_goal, q_start=q_start,
            uniform_random=uniform_random,
            current_sigma=current_sigma,
            training=True,
        )
        log.append(ep_info)
        total_steps += ep_info["steps"]

        # TD3 updates: only after warmup buffer is filled
        if total_steps >= hp.start_steps and len(buffer) >= hp.batch_size:
            for update_i in range(hp.updates_per_episode):
                batch = buffer.sample(hp.batch_size)
                if batch is None:
                    break
                batch_f32 = {k: v.to(dtype=torch.float32, device=device)
                             for k, v in batch.items()}
                td3.train_step(
                    batch        = batch_f32,
                    gamma        = hp.gamma,
                    tau          = hp.tau,
                    policy_delay = hp.policy_delay,
                    step         = ep * hp.updates_per_episode + update_i,
                    target_sigma = hp.target_sigma,
                    target_clip  = hp.target_clip,
                )

        if verbose and (ep % 100 == 0 or ep == hp.n_episodes - 1):
            window    = log[max(0, ep - 99):]
            sr        = sum(e["success"] for e in window) / len(window) * 100
            cr        = sum(e["crashed"] for e in window) / len(window) * 100
            avg_r     = sum(e["total_reward"] for e in window) / len(window)
            avg_steps = sum(e["steps"] for e in window) / len(window)
            elapsed   = time.time() - t0
            warmup_tag = " [warmup]" if total_steps < hp.start_steps else ""
            print(f"  ep {ep:6d}/{hp.n_episodes} | "
                  f"SR {sr:5.1f}% | CR {cr:4.1f}% | "
                  f"AvgR {avg_r:7.2f} | Steps {avg_steps:5.1f} | "
                  f"sigma {current_sigma:.3f} | "
                  f"HER {buffer.her_ratio:.2f} | Buf {len(buffer):7d} | "
                  f"{elapsed:.0f}s{warmup_tag}")

        # Optuna: report intermediate value and check for pruning
        if trial is not None and ep >= eval_start:
            n_eval   = len(log) - eval_start
            curr_sr  = sum(e["success"] for e in log[eval_start:]) / max(n_eval, 1)
            trial.report(curr_sr, step=ep)
            if trial.should_prune():
                raise optuna.TrialPruned()

    eval_log = log[eval_start:]
    final_sr = sum(e["success"] for e in eval_log) / max(len(eval_log), 1)

    if verbose:
        print(f"\n  Final success rate (last 10%): {final_sr * 100:.2f}%")

    return final_sr


# ---------------------------------------------------------------------------
# Optuna hyperparameter suggestion
# ---------------------------------------------------------------------------

def suggest_hparams(trial:  optuna.Trial,
                    base:   HParams,
                    flags:  TuneConfig) -> HParams:
    """
    Build an HParams for this trial.
    Flagged parameters are sampled by Optuna; unflagged keep their base default.
    """
    hp = copy.deepcopy(base)

    if flags.gamma:
        hp.gamma = trial.suggest_float("gamma", 0.95, 0.999)
    if flags.tau:
        hp.tau = trial.suggest_float("tau", 1e-3, 1e-1, log=True)
    if flags.actor_lr:
        hp.actor_lr = trial.suggest_float("actor_lr", 1e-5, 1e-3, log=True)
    if flags.critic_lr:
        hp.critic_lr = trial.suggest_float("critic_lr", 1e-4, 1e-2, log=True)
    if flags.hidden_width:
        hp.hidden_width = trial.suggest_categorical("hidden_width", [128, 256, 512])
    if flags.hidden_depth:
        hp.hidden_depth = trial.suggest_int("hidden_depth", 1, 4)
    if flags.pos_w:
        hp.pos_w = trial.suggest_float("pos_w", 0.5, 3.0)
    if flags.rot_w:
        hp.rot_w = trial.suggest_float("rot_w", 0.1, 2.0)
    if flags.vel_lambda:
        hp.vel_lambda = trial.suggest_float("vel_lambda", 0.01, 0.5, log=True)
    if flags.acc_lambda:
        hp.acc_lambda = trial.suggest_float("acc_lambda", 0.01, 0.5, log=True)
    if flags.her_ratio_start:
        hp.her_ratio_start = trial.suggest_float("her_ratio_start", 0.4, 0.95)
    if flags.her_decay_steps:
        hp.her_decay_steps = trial.suggest_int("her_decay_steps", 200, 5_000, log=True)
    if flags.expl_sigma:
        hp.expl_sigma = trial.suggest_float("expl_sigma", 0.05, 0.3)
    if flags.expl_sigma_end:
        # Only meaningful when decay is on; Optuna tunes it regardless so
        # the study can discover whether decay helps at all.
        hp.expl_sigma_end = trial.suggest_float("expl_sigma_end", 0.005, 0.1, log=True)
        # If the suggested end is higher than start, clamp it below start
        hp.expl_sigma_end = min(hp.expl_sigma_end, hp.expl_sigma * 0.9)
    if flags.target_sigma:
        hp.target_sigma = trial.suggest_float("target_sigma", 0.1, 0.4)

    return hp


# ---------------------------------------------------------------------------
# Optuna study runner
# ---------------------------------------------------------------------------

def run_tuning(base_hp:       HParams,
               tune_flags:    TuneConfig,
               n_trials:      int = 50,
               parallel_mode: str = "gpu",
               n_jobs:        int = 1,
               storage_path:  str = "optuna_study.db"):
    """
    Run the Optuna hyperparameter search.

    parallel_mode="gpu"  -- trials run sequentially, each on CUDA
    parallel_mode="cpu"  -- trials run in parallel on CPU (n_jobs workers)
    """
    storage_url = f"sqlite:///{storage_path}"

    pruner = optuna.pruners.MedianPruner(
        n_startup_trials = 5,
        n_warmup_steps   = int(base_hp.n_episodes * 0.5),
        interval_steps   = 100,
    )

    study = optuna.create_study(
        study_name     = "td3_ik_tuning",
        direction      = "maximize",
        storage        = storage_url,
        load_if_exists = True,
        pruner         = pruner,
    )

    if parallel_mode == "gpu":
        device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        eff_jobs = 1
        print(f"[Optuna] Sequential GPU trials  device={device}")
    else:
        device   = torch.device("cpu")
        eff_jobs = n_jobs
        print(f"[Optuna] Parallel CPU trials  n_jobs={eff_jobs}")

    def objective(trial: optuna.Trial) -> float:
        hp    = suggest_hparams(trial, base_hp, tune_flags)
        robot = make_robot(device)
        print(f"\n[Trial {trial.number}] "
              + "  ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
                          for k, v in trial.params.items()))
        return train(hp=hp, robot=robot, device=device, trial=trial, verbose=True)

    study.optimize(objective, n_trials=n_trials, n_jobs=eff_jobs)

    print("\n" + "=" * 60)
    print("Best trial:")
    best = study.best_trial
    print(f"  Success rate : {best.value * 100:.2f}%")
    print("  Params:")
    for k, v in best.params.items():
        print(f"    {k:20s} = {v}")
    print("=" * 60)
    print(f"\nTo open the dashboard: optuna-dashboard {storage_url}")
    return study


# ---------------------------------------------------------------------------
# Dashboard launcher
# ---------------------------------------------------------------------------

def launch_dashboard(storage_path: str = "optuna_study.db", port: int = 8080):
    """Launch the Optuna web dashboard against the persistent SQLite study."""
    storage_url = f"sqlite:///{storage_path}"
    print(f"[Dashboard] http://localhost:{port}  (Ctrl-C to stop)")
    optuna_dashboard.run_server(storage=storage_url, port=port)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="TD3 IK Agent -- train / tune / dashboard",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--mode", choices=["train", "tune", "dashboard"],
                   default="train")

    # Parallel mode
    p.add_argument("--parallel_mode", choices=["gpu", "cpu"], default="gpu",
                   help="gpu: sequential CUDA | cpu: parallel CPU workers")
    p.add_argument("--n_jobs", type=int, default=4,
                   help="Parallel workers (cpu mode only)")

    # Training loop
    p.add_argument("--n_episodes",          type=int,   default=10_000)
    p.add_argument("--max_steps",           type=int,   default=200)
    p.add_argument("--updates_per_episode", type=int,   default=50)
    p.add_argument("--batch_size",          type=int,   default=256)
    p.add_argument("--start_steps",         type=int,   default=2_000,
                   help="Uniform random steps before actor is used")
    p.add_argument("--decay_expl_noise",    action="store_true",
                   help="Linearly decay exploration sigma over training")

    # Optuna
    p.add_argument("--n_trials",       type=int, default=50)
    p.add_argument("--storage_path",   type=str, default="optuna_study.db")
    p.add_argument("--dashboard_port", type=int, default=8080)

    # Per-parameter tune flags  (--no_tune_X fixes X at its default)
    tuneable = ["gamma", "tau", "actor_lr", "critic_lr",
                "hidden_width", "hidden_depth", "pos_w", "rot_w",
                "vel_lambda", "acc_lambda", "her_ratio_start",
                "her_decay_steps", "expl_sigma", "expl_sigma_end", "target_sigma"]
    for name in tuneable:
        p.add_argument(f"--no_tune_{name}", action="store_true",
                       help=f"Fix {name} at its HParams default")

    return p.parse_args()


def main():
    args = parse_args()

    hp = HParams(
        n_episodes          = args.n_episodes,
        max_steps           = args.max_steps,
        updates_per_episode = args.updates_per_episode,
        batch_size          = args.batch_size,
        start_steps         = args.start_steps,
        decay_expl_noise    = args.decay_expl_noise,
    )

    tc = TuneConfig()
    for name in ["gamma", "tau", "actor_lr", "critic_lr",
                 "hidden_width", "hidden_depth", "pos_w", "rot_w",
                 "vel_lambda", "acc_lambda", "her_ratio_start",
                 "her_decay_steps", "expl_sigma", "expl_sigma_end", "target_sigma"]:
        if getattr(args, f"no_tune_{name}", False):
            setattr(tc, name, False)

    if args.mode == "dashboard":
        launch_dashboard(args.storage_path, args.dashboard_port)

    elif args.mode == "train":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Train] device={device}")
        robot = make_robot(device)
        train(hp=hp, robot=robot, device=device, verbose=True)

    elif args.mode == "tune":
        run_tuning(
            base_hp       = hp,
            tune_flags    = tc,
            n_trials      = args.n_trials,
            parallel_mode = args.parallel_mode,
            n_jobs        = args.n_jobs,
            storage_path  = args.storage_path,
        )


if __name__ == "__main__":
    main()