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

        self.current_traj_grade = [] # 'good' (1), 'okay' (0.3), 'bad'(0) grades

        self.loss_curr = torch.tensor([0.0], requires_grad=True)

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
        self.current_traj_grade = []


    # ------------------------------------------------------------------
    # scipy objective (must return a plain Python float)
    # ------------------------------------------------------------------
    # def _cost(self, x: np.ndarray,
    #           Goal_Posi: torch.Tensor,
    #           Goal_Ori_6D: torch.Tensor) -> float:
    #     """
    #     Wrapper around Loss_Math.Loss for scipy.optimize.minimize.
    #     x  — delta_q as a NumPy array (shape (n,))
    #     """
    #     Goal_Ori_SO3 = to_SO3(Goal_Ori_6D)

    #     loss = self.L.Loss(
    #         delta_q_prev=self.delta_q_prev,
    #         q_curr=self.q_curr,
    #         delta_q_next=x,
    #         pos_G_workspace=Goal_Posi,
    #         ori_G_SO3=Goal_Ori_SO3,
    #     )

    #     self.loss_curr = loss
    #     return float(loss)


    # ------------------------------------------------------------------
    # scipy objective with gradient backpropegation inherent. 
    # ------------------------------------------------------------------
    def _cost_and_grad(self, x_np: np.ndarray,
                   Goal_Posi: torch.Tensor,
                   Goal_Ori_6D: torch.Tensor):

    # ---- Convert to torch with gradients ----
        x = torch.tensor(x_np, dtype=torch.float64, requires_grad=True)

        Goal_Ori_SO3 = to_SO3(Goal_Ori_6D)

        loss = self.L.Loss(
            delta_q_prev=self.delta_q_prev,
            q_curr=self.q_curr,
            delta_q_next=x,
            pos_G_workspace=Goal_Posi,
            ori_G_SO3=Goal_Ori_SO3,
        )

        # ---- Backprop ----
        loss.backward()

        grad = x.grad.detach().cpu().numpy().astype(np.float64)

        return float(loss.detach()), grad

    #--------------------------------
    # Wrapped Scipy objective
    #--------------------------------
    def _fun_and_jac(self, x, Goal_Posi, Goal_Ori_6D):
        return self._cost_and_grad(x, Goal_Posi, Goal_Ori_6D)


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

        # my_fun = lambda x: self._cost(x, Goal_Posi=Goal_Posi, Goal_Ori_6D=Goal_Ori_6D)

        # Moving bounds: shift the absolute joint limits by q_curr so that
        # q_curr + delta_q is guaranteed to stay within joint limits.
        # The raw joint_bounds would wrongly constrain delta_q itself as if
        # it were an absolute angle rather than a change in angle.
        
        moving_bounds = self.rob.delta_q_bounds(q_curr=self.q_curr).detach().cpu().numpy()
        bounds_obj = Bounds(moving_bounds[:, 0], moving_bounds[:, 1])
        # res = minimize(fun=my_fun,
        #                x0=np.zeros(len(self.q_curr)),
        #                method="L-BFGS-B",
        #                bounds=bounds_obj,
        #                jac= self.loss_curr.backward()) 

        res = minimize(
            fun=lambda x: self._fun_and_jac(x, Goal_Posi, Goal_Ori_6D),
            x0=np.zeros(len(self.q_curr)),
            method="L-BFGS-B",
            bounds=bounds_obj,
            jac=True
        )

        self.get_delta_q(new_delta_q=res.x)

        # apply the found delta to the robot for the next step
        self.rob.q_vect = self.q_curr + self._to_tensor(res.x)

        return torch.tensor(res.x).double()


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

        # my_fun = lambda x: self._cost(x, Goal_Posi=Goal_Posi, Goal_Ori_6D=Goal_Ori_6D)

        # Moving bounds: shift the absolute joint limits by q_curr so that
        # q_curr + delta_q is guaranteed to stay within joint limits.
        # The raw joint_bounds would wrongly constrain delta_q itself as if
        # it were an absolute angle rather than a change in angle.
        
        moving_bounds = self.rob.delta_q_bounds(q_curr=self.q_curr).detach().cpu().numpy()
        bounds_obj = Bounds(moving_bounds[:, 0], moving_bounds[:, 1])
        # res = minimize(fun=my_fun,
        #                x0=np.zeros(len(self.q_curr)),
        #                method="L-BFGS-B",
        #                bounds=bounds_obj,
        #                jac= self.loss_curr.backward())

        res = minimize(
            fun=lambda x: self._fun_and_jac(x, Goal_Posi, Goal_Ori_6D),
            x0=np.zeros(len(self.q_curr)),
            method="L-BFGS-B",
            bounds=bounds_obj,
            jac=True
        )

        self.get_delta_q(new_delta_q=res.x)

        # apply the found delta to the robot for the next step
        self.rob.q_vect = self.q_curr + self._to_tensor(res.x)

        return torch.tensor(res.x).double()





    # ------------------------------------------------------------------
    # Trajectory following
    # ------------------------------------------------------------------
    def follow_trajectory(self, traj_t_delay_start_q: tuple, Print_Bool=True) -> tuple:
        """
        Follow a list of (position, 6D-orientation) waypoints.
        """
        self.reset_vars()
        

        # q_vect_trajectory stores (result, q_abs) pairs so the visualiser
        # always has the correct absolute pose without re-deriving it from deltas.
        q_vect_trajectory = []

        trajectory, t_delay, start_q = traj_t_delay_start_q

        frames = len(trajectory)

        if start_q is not None:
            self.rob.q_vect = self._to_tensor(start_q)

        self.start_q = self.rob.q_vect.clone()   # remember for playback

        # split the trajectory into positions and orientations once and iterate through them
        Goal_posi_traj = trajectory[:, 0:3]
        Goal_Ori_6D_traj = trajectory[:, 3:]
        for i in range(len(trajectory)):
            if Print_Bool == True:
                print(f'solving frame {i + 1}/{frames}')
            Goal_posi = Goal_posi_traj[i]
            Goal_Ori_6D = Goal_Ori_6D_traj[i]

            self.q_curr = self.rob.q_vect.clone()
            result = self.get_IK(Goal_Posi=Goal_posi, Goal_Ori_6D=Goal_Ori_6D)
            # Store absolute q alongside the result so the visualiser never
            # needs to accumulate deltas from an unknown starting point.
            q_abs = self.rob.q_vect.clone()   # get_IK already applied delta
            q_vect_trajectory.append((result, q_abs))

            self.current_traj_grade.append(self.L.target_weight)

        # print(f'joint trajectory = {[r.x for r, _ in q_vect_trajectory]}')
        # print(f'joint trajectory = {[r for r, _ in q_vect_trajectory]}')
        # print(f'crashed? = {self.L.crashed}')

        self.current_trajectory = q_vect_trajectory
        self.current_time_delay = t_delay






    #---------------------------------------------------------------------------
    # Traditional IK solver for finding poses in workspace (no initial pose)
    # --------------------------------------------------------------------------
    # ------------------------------------------------------------------
    # scipy objective with gradient backpropegation inherent. 
    # ------------------------------------------------------------------
    def _cost_and_grad_traditional(self, x_np: np.ndarray,
                                   Goal_Posi: torch.Tensor,
                                   Goal_Ori_6D: torch.Tensor):

    # ---- Convert to torch with gradients ----
        x = torch.tensor(x_np, dtype=torch.float64, requires_grad=True)
        
        self.rob.q_vect = x
        my_pos = self.rob.give_ds()[-1]
        my_R_SO3 = self.rob.give_Rs()[-1]

        # self.rob.q_vect = old_q_vect
        my_G_SO3 = to_SO3(Goal_Ori_6D)

        pos_loss = self.L.err_pos(pos_curr= my_pos, pos_G_ws= Goal_Posi)
        ori_loss = self.L.err_ori(new_ori_SO3= my_R_SO3, G_SO3= my_G_SO3)

        loss = pos_loss + ori_loss

        # ---- Backprop ----
        loss.backward()

        grad = x.grad.detach().cpu().numpy().astype(np.float64)

        return float(loss.detach()), grad
    

    def get_IK_traditional(self, Goal_Posi, Goal_ori_SO3):
        
        Goal_Ori_6D = to_6D_R(Goal_ori_SO3)



        res = minimize(
            fun=lambda x: self._cost_and_grad_traditional(x, Goal_Posi, Goal_Ori_6D),
            x0=np.zeros(len(self.q_curr)),
            method="L-BFGS-B",
            bounds=self.rob.bounds,
            jac=True
        )
        
        # converted directly to torch tensor for convienience of making dataset
        return torch.from_numpy(res.x).double() # float64
    


def test_trad_IK():
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
    )
    robot.WT = robot.make_homogenous_transformation(
        yaw=0, pitch=0, roll=180, x=0, y=0, z=0)

    solver = Oracle(robot_class=robot)

    solver.L.pos_w = 50
    solver.L.rot_w = 1

    my_goal_posi = torch.tensor([20, 0, 5])
    my_goal_ori = YPR_SO3(0, 0, 180)


    q_vect = solver.get_IK_traditional(Goal_Posi= my_goal_posi, 
                                       Goal_ori_SO3= my_goal_ori)
    
    print(f'goal posi = {my_goal_posi}\ngoal ori =\n{my_goal_ori}')
    
    print(f'found pose = {q_vect}\n found position = {robot.give_ds()[-1]}\n found ori = \n{robot.give_Rs()[-1]}')



if __name__ == "__main__":
    test_trad_IK()