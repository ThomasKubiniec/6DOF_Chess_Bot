"""
The goal is to make approximately 100k total samples of good and okay frames
(trajectory points), discarding bad or crashing frames.

Good points have a target weight of 1.0.  Okay has a target weight of 0.3.
Loss is multiplied by the target_weight.

GPU-aware: all tensor construction happens on robot.device.
The only CPU boundary is scipy inside the Oracle (handled there).
"""

import torch
import numpy as np
import pickle
from collections import deque
import random

from forward_kinematics import Robot_math
from path_planning_math import PathPlannerMath
from expert_inv_kin import Oracle
from rot_math import YPR_SO3, to_SO3, to_6D_R


class data_generator:
    def __init__(self,
                 robot: Robot_math,
                 solver: Oracle,
                 path_planner: PathPlannerMath,
                 low_frac,
                 high_frac,
                 std_low,
                 std_high,
                 p_low):

        self.robot        = robot
        self.solver       = solver
        self.path_planner = path_planner
        self.device       = robot.device

        self.mean_low  = self.robot.max_reach * low_frac
        self.std_low   = std_low
        self.mean_high = self.robot.max_reach * high_frac
        self.std_high  = std_high
        self.p_low     = p_low

    # ------------------------------------------------------------------
    def _kw(self):
        return dict(dtype=torch.float64, device=self.device)

    def make_random_unit_vect(self) -> torch.Tensor:
        rand_vect = torch.randn(3, **self._kw())
        return torch.nn.functional.normalize(rand_vect, p=2, dim=0)

    def make_random_radius(self, rad_mean, rad_std) -> torch.Tensor:
        min_r  = 0.0
        max_r  = float(self.robot.max_reach)
        # trunc_normal_ works in-place on a CPU tensor, then move to device
        rand_r = torch.nn.init.trunc_normal_(
            tensor=torch.empty(1),
            mean=float(rad_mean) if not isinstance(rad_mean, float) else rad_mean,
            std=rad_std,
            a=min_r,
            b=max_r,
        )
        return rand_r.to(**self._kw())

    def make_rand_xyz(self) -> torch.Tensor:
        """
        Bimodal distribution of radii within the workspace.
        """
        p      = torch.rand(1).item()
        rand_u = self.make_random_unit_vect()

        if p <= self.p_low:
            rand_r = self.make_random_radius(rad_mean=self.mean_low, rad_std=self.std_low)
        else:
            rand_r = self.make_random_radius(rad_mean=self.mean_high, rad_std=self.std_high)

        return rand_u * rand_r

    def make_random_YPR(self) -> torch.Tensor:
        dist   = torch.distributions.Uniform(low=-180.0, high=180.0)
        sample = dist.sample((3,))
        return YPR_SO3(yaw_deg=sample[0].item(),
                       pitch_deg=sample[1].item(),
                       roll_deg=sample[2].item(),
                       device=self.device)

    def make_random_q_vect(self) -> torch.Tensor:
        """Sample random joint angles that reach a non-crashing pose."""
        while True:
            goal_xyz     = self.make_rand_xyz()
            goal_ori_SO3 = self.make_random_YPR()
            q_vect       = self.solver.get_IK_traditional(
                Goal_Posi=goal_xyz, Goal_ori_SO3=goal_ori_SO3)
            if not self.solver.L.crashed:
                return q_vect

    def normalize_inputs(self,
                         delta_q_prev: torch.Tensor,
                         q_vect: torch.Tensor,
                         target_xyz: torch.Tensor,
                         target_ori: torch.Tensor):
        """
        Normalise raw inputs for the network:
            delta_q_prev  → normalised joint velocity
            q_vect        → normalised joint angles
            target_xyz    → normalised distance to goal
            target_ori    → relative rotation to goal in 6D
        """
        delta_q_N = self.solver.L.get_normal_joint_vel(delta_q=delta_q_prev)
        q_vect_N  = self.solver.L.get_normal_joint_value(q_vect=q_vect)

        self.robot.q_vect    = q_vect
        current_position     = self.robot.give_ds()[-1]
        dist_to_targ_N       = self.solver.L.get_normal_dist_to_goal(
            pos_curr=current_position, pos_G_ws=target_xyz)

        current_ori_SO3  = self.robot.give_Rs()[-1]
        targ_ori_SO3     = to_SO3(target_ori)
        rot_to_targ_SO3  = targ_ori_SO3.T @ current_ori_SO3
        rot_to_targ_6D_R = to_6D_R(rot_to_targ_SO3)

        return (delta_q_N, q_vect_N, dist_to_targ_N, rot_to_targ_6D_R)

    def make_random_traj(self):
        """
        Make a random trajectory between two random workspace coordinates.
        Returns (inputs, outputs, target_weights) or None on a crashed traj.
        """
        q1 = self.make_random_q_vect()
        q2 = self.make_random_q_vect()

        frame_low  = 20
        frame_high = 100
        frames_range = torch.linspace(frame_low, frame_high,
                                      steps=frame_high - frame_low + 1)
        random_idx = torch.randint(0, len(frames_range), (1,)).item()
        frames     = int(frames_range[random_idx].item())

        workspace_traj, time_delay, initial_pose = self.path_planner.MoveL(
            tot_time=1, frames=frames, q_init=q1, q_end=q2)

        self.solver.follow_trajectory(
            traj_t_delay_start_q=(workspace_traj, time_delay, initial_pose),
            Print_Bool=False)

        if self.solver.L.crashed:
            return None

        inputs    = []
        outputs   = []
        target_ws = []

        # First frame
        inputs.append(self.normalize_inputs(
            delta_q_prev=torch.zeros(len(self.robot.a), **self._kw()),
            q_vect=initial_pose,
            target_xyz=workspace_traj[0][:3],
            target_ori=workspace_traj[0][3:],
        ))
        outputs.append(self.solver.current_trajectory[0][0])

        for i, ws_traj in enumerate(workspace_traj[1:]):
            inputs.append(self.normalize_inputs(
                delta_q_prev=self.solver.current_trajectory[i - 1][0],
                q_vect=self.solver.current_trajectory[i - 1][1],
                target_xyz=ws_traj[:3],
                target_ori=ws_traj[3:],
            ))
            outputs.append(self.solver.current_trajectory[i][0])

        target_ws = self.solver.current_traj_grade.copy()
        return inputs, outputs, target_ws


class my_dataframe:
    def __init__(self, my_data_gen: data_generator, datapoints_goal, dataset_filename):
        self.my_data_gen    = my_data_gen
        self.datapoints_goal = datapoints_goal
        self.my_dataset      = deque(maxlen=round(self.datapoints_goal * 1.05))
        self.dataset_filename = dataset_filename

    def make_dataset(self):
        trajectory_collected = 0
        while len(self.my_dataset) < self.datapoints_goal:
            result = self.my_data_gen.make_random_traj()
            if result is None:
                continue   # crashed trajectory — skip
            inputs, outputs, weights = result
            for frame in zip(inputs, outputs, weights):
                self.my_dataset.append(frame)

            trajectory_collected += 1
            print(f'collected {trajectory_collected} trajectories | '
                  f'dataset size: {len(self.my_dataset)}/{int(self.datapoints_goal)}')

    def sample(self, batchsize=512):
        if len(self.my_dataset) < batchsize:
            return None
        return random.sample(list(self.my_dataset), batchsize)

    def save_dataset(self):
        with open(f'{self.dataset_filename}.pkl', 'wb') as f:
            pickle.dump(self.my_dataset, f)

    def load_dataset(self, filename):
        with open(f'{filename}.pkl', 'rb') as f:
            self.my_dataset = pickle.load(f)


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {device}")

    my_a     = [0.0, 7.375, 0.0,  0.0,  0.0,  0.0]
    my_alpha = [np.deg2rad(90),  np.deg2rad(180), np.deg2rad(90),
                np.deg2rad(90),  np.deg2rad(-90), np.deg2rad(0)]
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

    solver = Oracle(robot_class=robot,
                    pos_w=100, rot_w=5,
                    crash_w=2, dist_w=0.5,
                    vel_lambda=[0.1] * 6,
                    acc_lambda=[0.1] * 6)

    solver.L.max_good_err_pos = 0.1
    solver.L.max_good_err_deg = 2.0
    solver.L.max_ok_err_pos   = 0.25
    solver.L.max_ok_err_deg   = 6.0

    path_planner = PathPlannerMath(my_robot=robot)

    data_gen = data_generator(
        robot=robot, solver=solver, path_planner=path_planner,
        low_frac=1/2, high_frac=5/6,
        std_low=0.4,  std_high=0.4,
        p_low=0.75,
    )

    My_Dataset = my_dataframe(
        my_data_gen=data_gen,
        datapoints_goal=2e5,
        dataset_filename='training_data_5_6_26',
    )

    My_Dataset.make_dataset()