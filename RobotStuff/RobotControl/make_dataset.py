"""
Dataset generation with multiprocessing.

Architecture
------------
                  ┌─────────────┐
                  │  main proc  │  owns GPU (training) + result deque
                  └──────┬──────┘
                         │ drains
                  ┌──────▼──────┐
                  │    Queue    │  mp.Queue — completed traj frame batches
                  └──────▲──────┘
        ┌─────────────┬──┴──┬─────────────┐
        │  worker 0   │ ... │  worker N-1 │  CPU-only Oracle solvers
        └─────────────┘     └─────────────┘

Each worker owns its own Robot_math + Oracle + PathPlannerMath instance
(all on CPU).  Workers run forever, pushing completed trajectory frame
batches into the shared queue.  The main process drains the queue into
the dataset deque.

Why CPU for workers
-------------------
scipy L-BFGS-B is single-threaded and cannot use the GPU.  Running the
GPU inside each worker's backprop call would compete with every other
worker over the same CUDA context, adding synchronisation overhead that
wipes out any gain.  Keeping workers on CPU lets every core work
independently with zero contention.  The GPU stays free for training.

Usage
-----
    python make_dataset.py                         # auto worker count
    python make_dataset.py --workers 8             # explicit worker count
    python make_dataset.py --load myfile           # resume from saved file
    python make_dataset.py --frames 500000         # custom frame target
"""

import torch
import numpy as np
import pickle
import random
import argparse
import os
import time
from collections import deque

import multiprocessing as mp
from multiprocessing import Process, Queue, Event

from forward_kinematics import Robot_math
from path_planning_math import PathPlannerMath
from expert_inv_kin import Oracle
from rot_math import YPR_SO3, to_SO3, to_6D_R


# ===========================================================================
# Robot / solver config — edit once here, shared by main and all workers
# ===========================================================================
def make_robot_config() -> dict:
    return dict(
        a          = [0.0, 7.375, 0.0,  0.0,  0.0,  0.0],
        alpha      = [np.deg2rad(90),  np.deg2rad(180), np.deg2rad(90),
                      np.deg2rad(90),  np.deg2rad(-90), np.deg2rad(0)],
        d          = [-3.5, 0.0, 0.0, 8.25, 0.0, 5.1875],
        theta      = [np.deg2rad(0),   np.deg2rad(0),   np.deg2rad(90),
                      np.deg2rad(180), np.deg2rad(0),   np.deg2rad(-90)],
        joint_type = ["r"] * 6,
        bounds     = [
            (np.deg2rad(-90),  np.deg2rad(90)),
            (np.deg2rad(-180), np.deg2rad(0)),
            (np.deg2rad(-90),  np.deg2rad(90)),
            (np.deg2rad(-90),  np.deg2rad(90)),
            (np.deg2rad(-90),  np.deg2rad(90)),
            (np.deg2rad(-90),  np.deg2rad(90)),
        ],
        fail_dist  = [0.1] * 6,
    )


def make_oracle_config() -> dict:
    return dict(
        pos_w      = 100,
        rot_w      = 5,
        crash_w    = 2,
        dist_w     = 0.5,
        vel_lambda = [0.1] * 6,
        acc_lambda = [0.1] * 6,
    )


def make_generator_config() -> dict:
    return dict(
        low_frac  = 1 / 2,
        high_frac = 5 / 6,
        std_low   = 0.4,
        std_high  = 0.4,
        p_low     = 0.75,
    )


# ===========================================================================
# Build a fully independent CPU stack (called inside each worker)
# ===========================================================================
def _build_cpu_stack() -> "data_generator":
    cpu = torch.device("cpu")

    robot    = Robot_math(**make_robot_config(), device=cpu)
    robot.WT = robot.make_homogenous_transformation(yaw=0, pitch=0, roll=180, x=0, y=0, z=0)

    solver = Oracle(robot_class=robot, **make_oracle_config())
    solver.L.max_good_err_pos = 0.1
    solver.L.max_good_err_deg = 2.0
    solver.L.max_ok_err_pos   = 0.25
    solver.L.max_ok_err_deg   = 6.0

    path_planner = PathPlannerMath(my_robot=robot)

    return data_generator(robot=robot, solver=solver,
                          path_planner=path_planner,
                          **make_generator_config())


# ===========================================================================
# Worker process entry point
# ===========================================================================
# def _worker_fn(worker_id: int, result_queue: Queue, stop_event: Event):
def _worker_fn(worker_id: int, result_queue: Queue, stop_event): # removed "Event" to stop pylance error
    """
    Infinite generation loop.  Each completed trajectory is serialised to
    plain Python lists (tensors don't pickle reliably across spawn boundaries)
    and pushed to result_queue.  Exits when stop_event is set.
    """
    # Independent RNG per worker
    torch.manual_seed(os.getpid() + worker_id * 1000)
    np.random.seed(os.getpid() + worker_id * 1000)

    gen = _build_cpu_stack()

    while not stop_event.is_set():
        result = gen.make_random_traj()
        if result is None:
            continue  # crashed or invalid — retry

        inputs, outputs, target_ws = result

        # Serialise tensors → nested Python lists for safe pickling
        serialised = []
        for inp, out, tw in zip(inputs, outputs, target_ws):
            delta_q_N, q_N, dist_N, rot_6D = inp
            serialised.append((
                (delta_q_N.tolist(),
                 q_N.tolist(),
                 dist_N.tolist(),
                 rot_6D.tolist()),
                out.tolist(),
                float(tw),
            ))

        # block=True with timeout so stop_event can still be noticed
        try:
            result_queue.put(serialised, block=True, timeout=5)
        except Exception:
            pass  # queue full or closing — drop and loop


# ===========================================================================
# data_generator  (unchanged public API)
# ===========================================================================
class data_generator:
    def __init__(self,
                 robot: Robot_math,
                 solver: Oracle,
                 path_planner: PathPlannerMath,
                 low_frac, high_frac,
                 std_low, std_high, p_low,
                 frame_low_n=20, frame_high_n=100):

        self.robot        = robot
        self.solver       = solver
        self.path_planner = path_planner
        self.device       = robot.device

        self.mean_low  = self.robot.max_reach * low_frac
        self.std_low   = std_low
        self.mean_high = self.robot.max_reach * high_frac
        self.std_high  = std_high
        self.p_low     = p_low

        self.frame_low_n = frame_low_n
        self.frame_high_n = frame_high_n

    def _kw(self):
        return dict(dtype=torch.float64, device=self.device)

    def make_random_unit_vect(self) -> torch.Tensor:
        rand_vect = torch.randn(3, **self._kw())
        return torch.nn.functional.normalize(rand_vect, p=2, dim=0)

    def make_random_radius(self, rad_mean, rad_std) -> torch.Tensor:
        max_r  = float(self.robot.max_reach)
        rand_r = torch.nn.init.trunc_normal_(
            tensor=torch.empty(1),
            mean=float(rad_mean), std=rad_std,
            a=0.0, b=max_r,
        )
        return rand_r.to(**self._kw())

    def make_rand_xyz(self) -> torch.Tensor:
        p      = torch.rand(1).item()
        rand_u = self.make_random_unit_vect()
        if p <= self.p_low:
            rand_r = self.make_random_radius(self.mean_low,  self.std_low)
        else:
            rand_r = self.make_random_radius(self.mean_high, self.std_high)
        return rand_u * rand_r

    def make_random_YPR(self) -> torch.Tensor:
        sample = torch.distributions.Uniform(-180.0, 180.0).sample((3,))
        return YPR_SO3(yaw_deg=sample[0].item(),
                       pitch_deg=sample[1].item(),
                       roll_deg=sample[2].item(),
                       device=self.device)

    def make_random_q_vect(self) -> torch.Tensor:
        """Sample random joint angles that reach a non-crashing pose."""
        while True:
            goal_xyz = self.make_rand_xyz()
            goal_ori_SO3 = self.make_random_YPR()
            q_vect = self.solver.get_IK_traditional(
                Goal_Posi=goal_xyz, Goal_ori_SO3=goal_ori_SO3)
            if not self.solver.L.crashed:
                return q_vect

    def normalize_inputs(self,
                         delta_q_prev: torch.Tensor,
                         q_vect: torch.Tensor,
                         target_xyz: torch.Tensor,
                         target_ori: torch.Tensor):
        delta_q_N = self.solver.L.get_normal_joint_vel(delta_q=delta_q_prev)
        q_vect_N  = self.solver.L.get_normal_joint_value(q_vect=q_vect)

        self.robot.q_vect = q_vect
        current_position  = self.robot.give_ds()[-1]
        dist_to_targ_N    = self.solver.L.get_normal_dist_to_goal(
            pos_curr=current_position, pos_G_ws=target_xyz)

        current_ori_SO3  = self.robot.give_Rs()[-1]
        targ_ori_SO3     = to_SO3(target_ori)
        rot_to_targ_SO3  = targ_ori_SO3.T @ current_ori_SO3
        rot_to_targ_6D_R = to_6D_R(rot_to_targ_SO3)

        return (delta_q_N, q_vect_N, dist_to_targ_N, rot_to_targ_6D_R)


    def make_random_traj(self):
        """
        Generate one random trajectory.
        Returns (inputs, outputs, target_weights) or None if the robot crashed.
        """
        q1 = self.make_random_q_vect()
        q2 = self.make_random_q_vect()

        # frame_low    = 20
        # frame_high   = 100
        frames_range = torch.linspace(self.frame_low_n, self.frame_high_n,
                                      steps= (self.frame_high_n - self.frame_low_n + 1))
        frames = int(frames_range[
            torch.randint(0, len(frames_range), (1,)).item()].item())

        workspace_traj, time_delay, initial_pose = self.path_planner.MoveL(
            tot_time=1, frames=frames, q_init=q1, q_end=q2)

        self.solver.follow_trajectory(
            traj_t_delay_start_q=(workspace_traj, time_delay, initial_pose),
            Print_Bool=False)

        if self.solver.L.crashed:
            return None

        inputs  = []
        outputs = []

        # append the first trajectory frame (0 initial joint velocity)
        inputs.append(self.normalize_inputs(
            delta_q_prev=torch.zeros(len(self.robot.a), **self._kw()),
            q_vect=initial_pose,
            target_xyz=workspace_traj[0][:3],
            target_ori=workspace_traj[0][3:],
        ))
        outputs.append(self.solver.L.get_normal_joint_vel(self.solver.current_trajectory[0][0]))

        # append the rest of the trajectory frames
        for i, ws_traj in enumerate(workspace_traj[1:]):
            inputs.append(self.normalize_inputs(
                delta_q_prev=self.solver.current_trajectory[i - 1][0],
                q_vect=self.solver.current_trajectory[i - 1][1],
                target_xyz=ws_traj[:3],
                target_ori=ws_traj[3:],
            ))
            outputs.append(self.solver.L.get_normal_joint_vel(self.solver.current_trajectory[i][0]))

        return inputs, outputs, self.solver.current_traj_grade.copy()


# ===========================================================================
# my_dataframe — parallel make_dataset() + original single-process fallback
# ===========================================================================
class my_dataframe:
    def __init__(self, my_data_gen: data_generator, datapoints_goal, dataset_filename):
        self.my_data_gen      = my_data_gen
        self.datapoints_goal  = int(datapoints_goal)
        self.my_dataset       = deque(maxlen=round(self.datapoints_goal * 1.05))
        self.dataset_filename = dataset_filename

    # ------------------------------------------------------------------
    # Parallel generation  (default)
    # ------------------------------------------------------------------
    def make_dataset(self, num_workers: int = None):
        """
        Spawn `num_workers` CPU processes, each running an independent
        Oracle solver.  The main process collects frames until the dataset
        target is reached, then shuts down all workers cleanly.

        Args:
            num_workers: parallel worker count.
                         Defaults to (cpu_count - 1), minimum 1.

        Measures total time, and estimates remaining time throughout
        """
        if num_workers is None:
            num_workers = max(1, (os.cpu_count() or 2) - 1)

        print(f"Parallel generation | workers={num_workers} | "
              f"target={self.datapoints_goal} frames")

        # Bounded queue: prevents workers from flooding RAM if main is slow
        result_queue = Queue(maxsize=num_workers * 4)
        stop_event   = mp.Event()

        workers = [
            Process(target=_worker_fn,
                    args=(i, result_queue, stop_event),
                    daemon=True,
                    name=f"oracle-worker-{i}")
            for i in range(num_workers)
        ]
        for w in workers:
            w.start()
        print(f"Workers started: {[w.pid for w in workers]}")

        traj_collected = 0
        t0 = time.time()

        try:
            while len(self.my_dataset) < self.datapoints_goal:
                try:
                    serialised_frames = result_queue.get(timeout=30)
                except Exception:
                    print("Warning: no data received for 30 s — "
                          "check that workers are alive.")
                    # Check if any workers died unexpectedly
                    alive = [w for w in workers if w.is_alive()]
                    if not alive:
                        print("All workers have exited — aborting.")
                        break
                    continue

                # Deserialise back to tensors (CPU; training will move to GPU)
                for (inp_lists, out_list, tw) in serialised_frames:
                    frame = (
                        (torch.tensor(inp_lists[0], dtype=torch.float64),
                         torch.tensor(inp_lists[1], dtype=torch.float64),
                         torch.tensor(inp_lists[2], dtype=torch.float64),
                         torch.tensor(inp_lists[3], dtype=torch.float64)),
                        torch.tensor(out_list, dtype=torch.float64),
                        tw,
                    )
                    self.my_dataset.append(frame)

                traj_collected += 1
                elapsed = time.time() - t0
                rate    = traj_collected / elapsed if elapsed > 0 else 1e-9
                frames_left = self.datapoints_goal - len(self.my_dataset)
                eta_min     = (frames_left / max(rate * 60, 1e-9))
                print(f"  traj {traj_collected:6d} | "
                      f"frames {len(self.my_dataset):7d}/{self.datapoints_goal} | "
                      f"{rate:.2f} traj/s | "
                      f"ETA {eta_min:.1f} min",
                      end="\r", flush=True)

        except KeyboardInterrupt:
            print("\nInterrupted by user.")

        finally:
            print()  # newline after \r progress line
            stop_event.set()
            for w in workers:
                w.join(timeout=5)
                if w.is_alive():
                    w.terminate()
            print("Workers shut down.")

        elapsed = time.time() - t0
        print(f"Done. {len(self.my_dataset)} frames | "
              f"{traj_collected} trajectories | "
              f"{elapsed/60:.1f} min")

    # ------------------------------------------------------------------
    # Single-process fallback
    # ------------------------------------------------------------------
    def make_dataset_single(self):
        traj_n = 0
        while len(self.my_dataset) < self.datapoints_goal:
            result = self.my_data_gen.make_random_traj()
            if result is None:
                continue
            inputs, outputs, weights = result
            for frame in zip(inputs, outputs, weights):
                self.my_dataset.append(frame)
            traj_n += 1
            print(f"[single] traj {traj_n} | "
                  f"frames {len(self.my_dataset)}/{self.datapoints_goal}")

    # ------------------------------------------------------------------
    # Sampling / persistence
    # ------------------------------------------------------------------
    def sample(self, batchsize=512):
        if len(self.my_dataset) < batchsize:
            return None
        return random.sample(list(self.my_dataset), batchsize)

    def save_dataset(self):
        path = f"{self.dataset_filename}.pkl"
        with open(path, "wb") as f:
            pickle.dump(self.my_dataset, f)
        print(f"Saved {len(self.my_dataset)} frames → {path}")

    def load_dataset(self, filename):
        path = f"{filename}.pkl"
        with open(path, "rb") as f:
            self.my_dataset = pickle.load(f)
        print(f"Loaded {len(self.my_dataset)} frames ← {path}")


# ===========================================================================
# Entry point
# ===========================================================================
if __name__ == "__main__":

    print(f'cpu count = {os.cpu_count()}')
    # Required on Windows / macOS (spawn start method).
    # Safe no-op on Linux (fork), but always good practice.
    mp.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser(description="Generate IK training dataset")
    parser.add_argument("--workers",  type=int, default=None,
                        help="Worker process count (default: cpu_count - 1)")
    parser.add_argument("--frames",   type=int, default=200_000,
                        help="Target frame count (default: 200000)")
    parser.add_argument("--load",     type=str, default=None,
                        help="Resume from an existing .pkl file (no extension)")
    parser.add_argument("--filename", type=str, default="training_data",
                        help="Output .pkl filename without extension")
    parser.add_argument("--frame_low_n",   type=int, default=20,
                        help="lowest traj frame number (default: 20)")
    parser.add_argument("--frame_high_n",   type=int, default=100,
                        help="highest traj frame number (default: 100)")
    args = parser.parse_args()

    # Main process runs its stack on CPU too — GPU stays free for training.
    cpu = torch.device("cpu")
    robot    = Robot_math(**make_robot_config(), device=cpu)
    robot.WT = robot.make_homogenous_transformation(yaw=0, pitch=0, roll=180, x=0, y=0, z=0)

    solver = Oracle(robot_class=robot, **make_oracle_config())
    solver.L.max_good_err_pos = 0.1
    solver.L.max_good_err_deg = 2.0
    solver.L.max_ok_err_pos   = 0.25
    solver.L.max_ok_err_deg   = 6.0

    path_planner = PathPlannerMath(my_robot=robot)
    data_gen     = data_generator(robot=robot, solver=solver,
                                  path_planner=path_planner,
                                  frame_low_n= args.frame_low_n, frame_high_n= args.frame_high_n,
                                  **make_generator_config())

    dataset = my_dataframe(
        my_data_gen=data_gen,
        datapoints_goal=args.frames,
        dataset_filename=args.filename,
    )

    if args.load:
        dataset.load_dataset(args.load)
        print(f"Resuming from {len(dataset.my_dataset)} existing frames.")

    dataset.make_dataset(num_workers=args.workers)
    dataset.save_dataset()


# To change dataset name:
# python make_dataset.py --filename my_robot_dataset

# Change other arguments:
# python make_dataset.py --filename my_robot_dataset --frames 500000 --workers 10