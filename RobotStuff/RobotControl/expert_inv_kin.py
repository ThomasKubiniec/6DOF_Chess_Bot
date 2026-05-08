"""
L-BFGS-B Inverse Kinematics solver.
Acts as an "Oracle" — checks whether a trajectory is reachable without
self-collision, and produces ground-truth joint deltas for imitation learning.
Pure PyTorch implementation.

GPU-aware: all tensors live on robot.device.
scipy.optimize requires CPU numpy at its interface; tensors are detached and
moved to CPU only at those call boundaries.
"""
import torch
import numpy as np
from scipy.optimize import minimize, Bounds

from forward_kinematics import Robot_math
from loss_math import Loss_Math
from rot_math import YPR_SO3, to_6D_R, to_SO3
from path_planning_math import PathPlannerMath


class Oracle:
    def __init__(self,
                 robot_class: Robot_math,
                 pos_w=None,
                 rot_w=None,
                 crash_w=None,
                 dist_w=None,
                 vel_lambda=None,
                 acc_lambda=None):

        self.rob    = robot_class
        self.device = robot_class.device   # inherit device from robot

        self.L = Loss_Math(my_robot=self.rob,
                           pos_w=pos_w, rot_w=rot_w,
                           crash_w=crash_w, dist_w=dist_w,
                           vel_lambda=vel_lambda, acc_lambda=acc_lambda)

        n = len(self.rob.a)
        self.delta_q_prev = torch.zeros(n, dtype=torch.float64, device=self.device)
        self.delta_q_next = torch.zeros(n, dtype=torch.float64, device=self.device)
        self.q_curr       = self.rob.q_vect.clone()

        self.current_XYZ_targ = torch.zeros(3, dtype=torch.float64, device=self.device)
        self.current_YPR_targ = torch.zeros(3, dtype=torch.float64, device=self.device)

        self.current_trajectory = []
        self.current_time_delay = 0
        self.current_traj_grade = []

        self.loss_curr = torch.tensor([0.0], requires_grad=True, device=self.device)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _to_tensor(self, x) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            return x.to(dtype=torch.float64, device=self.device)
        return torch.tensor(x, dtype=torch.float64, device=self.device)

    def get_delta_q(self, new_delta_q):
        """Shift velocity history forward one step."""
        self.delta_q_prev = self._to_tensor(self.delta_q_next)
        self.delta_q_next = self._to_tensor(new_delta_q)

    def reset_vars(self):
        n = len(self.rob.a)
        self.delta_q_prev = torch.zeros(n, dtype=torch.float64, device=self.device)
        self.delta_q_next = torch.zeros(n, dtype=torch.float64, device=self.device)
        self.L.reset_checks()
        self.current_traj_grade = []

    # ------------------------------------------------------------------
    # scipy objective with gradient via backprop
    # NOTE: scipy lives on CPU — tensors are moved to CPU for the grad
    #       numpy array, then immediately moved back to device for the
    #       next forward pass.
    # ------------------------------------------------------------------
    def _cost_and_grad(self, x_np: np.ndarray,
                       Goal_Posi: torch.Tensor,
                       Goal_Ori_6D: torch.Tensor):
        # numpy → GPU tensor with grad
        x = torch.tensor(x_np, dtype=torch.float64,
                         device=self.device, requires_grad=True)

        Goal_Ori_SO3 = to_SO3(Goal_Ori_6D)

        loss = self.L.Loss(
            delta_q_prev=self.delta_q_prev,
            q_curr=self.q_curr,
            delta_q_next=x,
            pos_G_workspace=Goal_Posi,
            ori_G_SO3=Goal_Ori_SO3,
        )

        loss.backward()

        # grad must be a plain CPU float64 numpy array for scipy
        grad = x.grad.detach().cpu().numpy().astype(np.float64)
        return float(loss.detach()), grad

    def _fun_and_jac(self, x, Goal_Posi, Goal_Ori_6D):
        return self._cost_and_grad(x, Goal_Posi, Goal_Ori_6D)

    # ------------------------------------------------------------------
    # Single-step IK
    # ------------------------------------------------------------------
    def get_IK(self,
               Goal_Posi: torch.Tensor,
               Goal_Ori_6D: torch.Tensor,
               start_q=None) -> torch.Tensor:
        """
        Find delta_q that moves the robot toward Goal_Posi / Goal_Ori_6D
        while minimising velocity, acceleration, and self-collision penalty.
        """
        if start_q is not None:
            self.rob.q_vect = self._to_tensor(start_q)
        self.q_curr = self.rob.q_vect.clone()

        Goal_Posi   = self._to_tensor(Goal_Posi)
        Goal_Ori_6D = self._to_tensor(Goal_Ori_6D)

        # delta_q_bounds returns a CPU tensor — scipy needs numpy
        moving_bounds = self.rob.delta_q_bounds(q_curr=self.q_curr).numpy()
        bounds_obj    = Bounds(moving_bounds[:, 0], moving_bounds[:, 1])

        # x0 must be CPU numpy for scipy
        x0 = np.zeros(len(self.q_curr))

        res = minimize(
            fun=lambda x: self._fun_and_jac(x, Goal_Posi, Goal_Ori_6D),
            x0=x0,
            method="L-BFGS-B",
            bounds=bounds_obj,
            jac=True,
        )

        self.get_delta_q(new_delta_q=res.x)
        self.rob.q_vect = self.q_curr + self._to_tensor(res.x)

        return torch.tensor(res.x, dtype=torch.float64, device=self.device)

    # ------------------------------------------------------------------
    # Single-step IK with internally held target
    # ------------------------------------------------------------------
    def get_IK_given_targ(self, start_q=None) -> torch.Tensor:
        if start_q is not None:
            self.rob.q_vect = self._to_tensor(start_q)
        self.q_curr = self.rob.q_vect.clone()

        Goal_Posi   = self.current_XYZ_targ
        Goal_Ori_6D = to_6D_R(YPR_SO3(yaw_deg=self.current_YPR_targ[0].item(),
                                       pitch_deg=self.current_YPR_targ[1].item(),
                                       roll_deg=self.current_YPR_targ[2].item(),
                                       device=self.device))

        moving_bounds = self.rob.delta_q_bounds(q_curr=self.q_curr).numpy()
        bounds_obj    = Bounds(moving_bounds[:, 0], moving_bounds[:, 1])

        x0 = np.zeros(len(self.q_curr))

        res = minimize(
            fun=lambda x: self._fun_and_jac(x, Goal_Posi, Goal_Ori_6D),
            x0=x0,
            method="L-BFGS-B",
            bounds=bounds_obj,
            jac=True,
        )

        self.get_delta_q(new_delta_q=res.x)
        self.rob.q_vect = self.q_curr + self._to_tensor(res.x)

        return torch.tensor(res.x, dtype=torch.float64, device=self.device)

    # ------------------------------------------------------------------
    # Trajectory following
    # ------------------------------------------------------------------
    def follow_trajectory(self, traj_t_delay_start_q: tuple, Print_Bool=True) -> None:
        """Follow a list of (position, 6D-orientation) waypoints."""
        self.reset_vars()

        q_vect_trajectory = []
        trajectory, t_delay, start_q = traj_t_delay_start_q
        frames = len(trajectory)

        if start_q is not None:
            self.rob.q_vect = self._to_tensor(start_q)

        self.start_q = self.rob.q_vect.clone()

        Goal_posi_traj   = trajectory[:, 0:3]
        Goal_Ori_6D_traj = trajectory[:, 3:]

        for i in range(len(trajectory)):
            if Print_Bool:
                print(f'solving frame {i + 1}/{frames}')
            Goal_posi   = Goal_posi_traj[i]
            Goal_Ori_6D = Goal_Ori_6D_traj[i]

            self.q_curr = self.rob.q_vect.clone()
            result      = self.get_IK(Goal_Posi=Goal_posi, Goal_Ori_6D=Goal_Ori_6D)
            q_abs       = self.rob.q_vect.clone()
            q_vect_trajectory.append((result, q_abs))

            self.current_traj_grade.append(self.L.target_weight)

        self.current_trajectory = q_vect_trajectory
        self.current_time_delay = t_delay

    # ------------------------------------------------------------------
    # Traditional IK (finds poses, no initial pose required)
    # ------------------------------------------------------------------
    def _cost_and_grad_traditional(self, x_np: np.ndarray,
                                   Goal_Posi: torch.Tensor,
                                   Goal_Ori_6D: torch.Tensor):
        x = torch.tensor(x_np, dtype=torch.float64,
                         device=self.device, requires_grad=True)

        self.rob.q_vect = x
        my_pos   = self.rob.give_ds()[-1]
        my_R_SO3 = self.rob.give_Rs()[-1]

        my_G_SO3 = to_SO3(Goal_Ori_6D)

        pos_loss = self.L.err_pos(pos_curr=my_pos, pos_G_ws=Goal_Posi)
        ori_loss = self.L.err_ori(new_ori_SO3=my_R_SO3, G_SO3=my_G_SO3)

        loss = pos_loss + ori_loss
        loss.backward()

        grad = x.grad.detach().cpu().numpy().astype(np.float64)
        return float(loss.detach()), grad

    def get_IK_traditional(self, Goal_Posi, Goal_ori_SO3) -> torch.Tensor:
        Goal_Posi   = self._to_tensor(Goal_Posi)
        Goal_Ori_6D = to_6D_R(self._to_tensor(Goal_ori_SO3))

        x0 = np.zeros(len(self.q_curr))

        res = minimize(
            fun=lambda x: self._cost_and_grad_traditional(x, Goal_Posi, Goal_Ori_6D),
            x0=x0,
            method="L-BFGS-B",
            bounds=self.rob.bounds,
            jac=True,
        )

        return torch.tensor(res.x, dtype=torch.float64, device=self.device)


def test_trad_IK():
    import numpy as np

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
    robot.WT = robot.make_homogenous_transformation(yaw=0, pitch=0, roll=180, x=0, y=0, z=0)

    solver = Oracle(robot_class=robot)

    solver.L.pos_w = 50
    solver.L.rot_w = 1

    my_goal_posi = torch.tensor([20, 0, 5], dtype=torch.float64, device=device)
    my_goal_ori  = YPR_SO3(0, 0, 180, device=device)

    q_vect = solver.get_IK_traditional(Goal_Posi=my_goal_posi, Goal_ori_SO3=my_goal_ori)

    print(f'goal posi = {my_goal_posi}\ngoal ori =\n{my_goal_ori}')
    print(f'found pose = {q_vect}')
    print(f'found position = {robot.give_ds()[-1]}')
    print(f'found ori =\n{robot.give_Rs()[-1]}')


if __name__ == "__main__":
    test_trad_IK()