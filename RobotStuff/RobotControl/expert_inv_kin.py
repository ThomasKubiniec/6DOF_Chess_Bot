"""
L-BFGS-B Inverse Kinematics solver.
Acts as an "Oracle" — checks whether a trajectory is reachable without
self-collision, and produces ground-truth joint deltas for imitation learning.
Pure PyTorch implementation.
"""
import torch
import numpy as np
from scipy.optimize import minimize, Bounds

from forward_kinematics import Robot_math
from loss_math import Loss_Math
from rot_math import YPR_SO3, Rx_SO3, Ry_SO3, Rz_SO3, to_6D_R, to_SO3
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

        self.rob = robot_class

        self.L = Loss_Math(my_robot=self.rob,
                           pos_w=pos_w,
                           rot_w=rot_w,
                           crash_w=crash_w,
                           dist_w=dist_w,
                           vel_lambda=vel_lambda,
                           acc_lambda=acc_lambda)

        n = len(self.rob.a)
        self.delta_q_prev = torch.zeros(n, dtype=torch.float64)
        self.delta_q_next = torch.zeros(n, dtype=torch.float64)
        self.q_curr       = self.rob.q_vect.clone()

        self.current_XYZ_targ = torch.zeros(3, dtype=torch.float64)
        self.current_YPR_targ = torch.zeros(3, dtype=torch.float64)

        self.current_trajectory = []
        self.current_time_delay = 0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _to_tensor(self, x) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            return x.to(torch.float64)
        return torch.tensor(x, dtype=torch.float64)

    def get_delta_q(self, new_delta_q):
        """Shift velocity history forward one step."""
        self.delta_q_prev = self._to_tensor(self.delta_q_next)
        self.delta_q_next = self._to_tensor(new_delta_q)

    def reset_vars(self):
        n = len(self.rob.a)
        self.delta_q_prev = torch.zeros(n, dtype=torch.float64)
        self.delta_q_next = torch.zeros(n, dtype=torch.float64)
        self.L.reset_checks()


    # ------------------------------------------------------------------
    # scipy objective (must return a plain Python float)
    # ------------------------------------------------------------------
    def _cost(self, x: np.ndarray,
              Goal_Posi: torch.Tensor,
              Goal_Ori_6D: torch.Tensor) -> float:
        """
        Wrapper around Loss_Math.Loss for scipy.optimize.minimize.
        x  — delta_q as a NumPy array (shape (n,))
        """
        Goal_Ori_SO3 = to_SO3(Goal_Ori_6D)

        loss = self.L.Loss(
            delta_q_prev=self.delta_q_prev,
            q_curr=self.q_curr,
            delta_q_next=x,
            pos_G_workspace=Goal_Posi,
            ori_G_SO3=Goal_Ori_SO3,
        )
        return float(loss)


    # ------------------------------------------------------------------
    # Single-step IK
    # ------------------------------------------------------------------
    def get_IK(self,
               Goal_Posi: torch.Tensor,
               Goal_Ori_6D: torch.Tensor,
               start_q=None) -> np.ndarray:
        """
        Find delta_q that moves the robot toward Goal_Posi / Goal_Ori_6D
        while minimising velocity, acceleration, and self-collision penalty.

        Returns the raw scipy OptimizeResult (access .x for the delta_q array).
        """
        if start_q is not None:
            self.rob.q_vect = self._to_tensor(start_q)
        self.q_curr = self.rob.q_vect.clone()

        Goal_Posi   = self._to_tensor(Goal_Posi)
        Goal_Ori_6D = self._to_tensor(Goal_Ori_6D)

        my_fun = lambda x: self._cost(x, Goal_Posi=Goal_Posi, Goal_Ori_6D=Goal_Ori_6D)

        # Moving bounds: shift the absolute joint limits by q_curr so that
        # q_curr + delta_q is guaranteed to stay within joint limits.
        # The raw joint_bounds would wrongly constrain delta_q itself as if
        # it were an absolute angle rather than a change in angle.
        
        moving_bounds = self.rob.delta_q_bounds(q_curr=self.q_curr).detach().cpu().numpy()
        bounds_obj = Bounds(moving_bounds[:, 0], moving_bounds[:, 1])
        res = minimize(fun=my_fun,
                       x0=np.zeros(len(self.q_curr)),
                       method="L-BFGS-B",
                       bounds=bounds_obj) # moving_bounds)

        # moving_bounds = self.rob.delta_q_bounds(q_curr=self.q_curr)
        # res = minimize(fun=my_fun,
        #                x0=np.zeros(len(self.q_curr)),
        #                method="L-BFGS-B",
        #                bounds=moving_bounds)

        self.get_delta_q(new_delta_q=res.x)

        # apply the found delta to the robot for the next step
        self.rob.q_vect = self.q_curr + self._to_tensor(res.x)

        return res


    # ------------------------------------------------------------------
    # Single-step IK with internally given target
    # ------------------------------------------------------------------
    def get_IK_given_targ(self, start_q=None) -> np.ndarray:
        """
        Find delta_q that moves the robot toward Goal_Posi / Goal_Ori_6D
        while minimising velocity, acceleration, and self-collision penalty.

        Returns the raw scipy OptimizeResult (access .x for the delta_q array).
        """
        if start_q is not None:
            self.rob.q_vect = self._to_tensor(start_q)
        self.q_curr = self.rob.q_vect.clone()

        Goal_Posi   = self.current_XYZ_targ
        Goal_Ori_6D = to_6D_R(YPR_SO3(yaw_deg= self.current_YPR_targ[0],
                                      pitch_deg= self.current_YPR_targ[1],
                                      roll_deg= self.current_YPR_targ[2]))

        my_fun = lambda x: self._cost(x, Goal_Posi=Goal_Posi, Goal_Ori_6D=Goal_Ori_6D)

        # Moving bounds: shift the absolute joint limits by q_curr so that
        # q_curr + delta_q is guaranteed to stay within joint limits.
        # The raw joint_bounds would wrongly constrain delta_q itself as if
        # it were an absolute angle rather than a change in angle.
        
        moving_bounds = self.rob.delta_q_bounds(q_curr=self.q_curr).detach().cpu().numpy()
        bounds_obj = Bounds(moving_bounds[:, 0], moving_bounds[:, 1])
        res = minimize(fun=my_fun,
                       x0=np.zeros(len(self.q_curr)),
                       method="L-BFGS-B",
                       bounds=bounds_obj) # moving_bounds)

        # moving_bounds = self.rob.delta_q_bounds(q_curr=self.q_curr)
        # res = minimize(fun=my_fun,
        #                x0=np.zeros(len(self.q_curr)),
        #                method="L-BFGS-B",
        #                bounds=moving_bounds)

        self.get_delta_q(new_delta_q=res.x)

        # apply the found delta to the robot for the next step
        self.rob.q_vect = self.q_curr + self._to_tensor(res.x)

        return res





    # ------------------------------------------------------------------
    # Trajectory following
    # ------------------------------------------------------------------
    def follow_trajectory(self, traj_t_delay_start_q: tuple) -> tuple:
        """
        Follow a list of (position, 6D-orientation) waypoints.
        """
        self.reset_vars()
        

        # q_vect_trajectory stores (result, q_abs) pairs so the visualiser
        # always has the correct absolute pose without re-deriving it from deltas.
        q_vect_trajectory = []

        trajectory, t_delay, start_q = traj_t_delay_start_q

        if start_q is not None:
            self.rob.q_vect = self._to_tensor(start_q)

        self.start_q = self.rob.q_vect.clone()   # remember for playback

        # split the trajectory into positions and orientations once and iterate through them
        Goal_posi_traj = trajectory[:, 0:3]
        Goal_Ori_6D_traj = trajectory[:, 3:]
        for i in range(len(trajectory)):
            Goal_posi = Goal_posi_traj[i]
            Goal_Ori_6D = Goal_Ori_6D_traj[i]

            self.q_curr = self.rob.q_vect.clone()
            result = self.get_IK(Goal_Posi=Goal_posi, Goal_Ori_6D=Goal_Ori_6D)
            # Store absolute q alongside the result so the visualiser never
            # needs to accumulate deltas from an unknown starting point.
            q_abs = self.rob.q_vect.clone()   # get_IK already applied delta
            q_vect_trajectory.append((result, q_abs))

        print(f'joint trajectory = {[r.x for r, _ in q_vect_trajectory]}')
        print(f'crashed? = {self.L.crashed}')

        self.current_trajectory = q_vect_trajectory
        self.current_time_delay = t_delay

# # ----------------------------------------------------------------------
# # Smoke test
# # ----------------------------------------------------------------------
# def test_point_to_point():
#     # ---- Define a 3-DOF robot (2 revolute + 1 prismatic) ----
#     a     = [500.0, 500.0, 0.0]
#     alpha = [0.0,   0.0,   0.0]
#     d     = [400.0, 0.0,   0.0]
#     theta = [0.0,   0.0,   0.0]
#     joint_types  = ['r', 'r', 'p']
#     joint_bounds = [
#         (np.deg2rad(-90), np.deg2rad(90)),
#         (np.deg2rad(-90), np.deg2rad(90)),
#         (-200.0, 0.0),
#     ]
#     link_radii = [10.0, 10.0, 10.0]

#     R = Robot_math(a=a, alpha=alpha, d=d, theta=theta,
#                    joint_type=joint_types,
#                    bounds=joint_bounds,
#                    fail_dist=link_radii)

#     R.q_vect = torch.zeros(3, dtype=torch.float64)

#     oracle = Oracle(robot_class=R)
#     oracle.reset_vars()

#     Goal_Posi   = torch.tensor([707.0, 0.0, 0.0], dtype=torch.float64)
#     Goal_Ori_6D = to_6D_R(Rz_SO3(theta_z_deg=45.0))

#     result = oracle.get_IK(Goal_Posi=Goal_Posi, Goal_Ori_6D=Goal_Ori_6D)

#     print(f'\nOptimisation success : {result.success}')
#     print(f'delta_q found        : {result.x}')
#     print(f'end-effector pos     :\n{R.give_ds()[-1]}')
#     print(f'end-effector ori     :\n{R.give_Rs()[-1]}')
#     print(f'crashed?             : {oracle.L.crashed}')


# if __name__ == "__main__":
#     test_point_to_point()