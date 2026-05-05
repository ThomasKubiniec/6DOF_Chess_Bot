"""
Loss functions for the IK Neural Network and the L-BFGS-B Oracle.
Pure PyTorch implementation.
"""
import torch
from scipy.optimize import minimize

from rot_math import to_SO3, to_6D_R, Rx_SO3
from forward_kinematics import Robot_math


'''
L = e_pos + e_ori + e_vel + e_acc + e_dist

e_pos  = pos_w  * ||normalised_epsilon_goal||
e_ori  = rot_w  * ||G_SO3 - R_curr||_F
e_vel  = ||vel_lambda @ normalised_delta_q||
e_acc  = ||acc_lambda @ normalised_delta_delta_q||
e_dist = dist_w * sigmoid(-k * normalised_min_dist) + crash_w (if crashed)
'''


class Loss_Math:
    def __init__(self, my_robot: Robot_math,
                 pos_w=None,
                 rot_w=None,
                 crash_w=None,
                 dist_w=None,
                 vel_lambda=None,   # list of per-joint weights
                 acc_lambda=None):  # list of per-joint weights

        self.rob = my_robot

        # ---- scalar weights (with defaults) ----
        self.pos_w   = pos_w   if pos_w   is not None else 1.0
        self.rot_w   = rot_w   if rot_w   is not None else 0.5
        self.crash_w = crash_w if crash_w is not None else 2.0
        self.dist_w  = dist_w  if dist_w  is not None else 0.1

        n = len(self.rob.a)

        # ---- diagonal lambda matrices ----
        if vel_lambda is None:
            vel_lambda = [0.1] * n
        if acc_lambda is None:
            acc_lambda = [0.1] * n

        self.vel_lambda = torch.diag(torch.tensor(vel_lambda, dtype=torch.float64))
        self.acc_lambda = torch.diag(torch.tensor(acc_lambda, dtype=torch.float64))

        # ---- joint bound tensors ----
        bounds_tensor   = torch.tensor(self.rob.joint_bounds, dtype=torch.float64)  # (n, 2)
        self.q_l = bounds_tensor[:, 0]   # lower bounds
        self.q_h = bounds_tensor[:, 1]   # upper bounds

        # ---- state ----
        self.crashed = False
        self.Pass = True


        # ---------- pass or fail thresholds ----------
        # ideal offset
        self.max_good_err_pos = 0.1
        self.max_good_err_deg = 0.2

        # acceptable
        self.max_ok_err_pos = 0.1
        self.max_ok_err_deg = 0.2

        # the target weight associated with being good, okay, or bad
        # good = 1, okay = 0.3, bad = 0 (bad targets are not in the dataset for the Neural Network)
        self.target_weight = 0 


    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _to_tensor(self, x) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            return x.to(torch.float64)
        return torch.tensor(x, dtype=torch.float64)

    def reset_checks(self):
        self.crashed = False
        self.Pass = True



    # ------------------------------------------------------------------
    # Normalisation utilities
    # ------------------------------------------------------------------
    def get_normal_dist_to_goal(self, pos_curr: torch.Tensor,
                                 pos_G_ws: torch.Tensor) -> torch.Tensor:
        """Normalised vector from current EE position to goal."""
        pos_curr = self._to_tensor(pos_curr)
        pos_G_ws = self._to_tensor(pos_G_ws)
        return (pos_G_ws - pos_curr) / self.rob.max_reach

    def get_normal_joint_vel(self, delta_q: torch.Tensor) -> torch.Tensor:
        """Map delta_q into [-1, 1] using the joint range."""
        delta_q = self._to_tensor(delta_q)
        return 2.0 * ((delta_q - self.q_l) / (self.q_h - self.q_l)) - 1.0

    def get_normal_joint_acc(self, delta_q_prev: torch.Tensor,
                              delta_q_new: torch.Tensor) -> torch.Tensor:
        """Normalised change in normalised velocity."""
        Nv_prev = self.get_normal_joint_vel(delta_q_prev)
        Nv_new  = self.get_normal_joint_vel(delta_q_new)
        return (Nv_new - Nv_prev) / 2.0

    def get_normal_link_dist(self, link_dist: torch.Tensor) -> torch.Tensor:
        return self._to_tensor(link_dist) / self.rob.max_reach


    # ------------------------------------------------------------------
    # Individual error terms
    # ------------------------------------------------------------------
    def err_pos(self, pos_curr: torch.Tensor,
                pos_G_ws: torch.Tensor) -> torch.Tensor:
        """Normalised positional error (vector norm)."""
        eps = self.get_normal_dist_to_goal(pos_curr=pos_curr, pos_G_ws=pos_G_ws)
        return self.pos_w * torch.linalg.vector_norm(eps)

    def err_ori(self, new_ori_SO3: torch.Tensor,
                G_SO3: torch.Tensor) -> torch.Tensor:
        """Frobenius-norm orientation error."""
        new_ori_SO3 = self._to_tensor(new_ori_SO3)
        G_SO3       = self._to_tensor(G_SO3)
        return self.rot_w * torch.linalg.matrix_norm(G_SO3 - new_ori_SO3)

    def err_vel(self, joint_velocity: torch.Tensor) -> torch.Tensor:
        """Penalise large joint velocities."""
        nv = self.get_normal_joint_vel(joint_velocity)
        return torch.linalg.vector_norm(self.vel_lambda @ nv)

    def err_acc(self, joint_vel_new: torch.Tensor,
                joint_vel_old: torch.Tensor) -> torch.Tensor:
        """Penalise large joint accelerations (jerk avoidance)."""
        na = self.get_normal_joint_acc(delta_q_new=joint_vel_new,
                                       delta_q_prev=joint_vel_old)
        return torch.linalg.vector_norm(self.acc_lambda @ na)

    def dist_sigmoid(self, dist: torch.Tensor, m: float, b: float) -> torch.Tensor:
        """
        Sigmoid-based proximity penalty.
        Large when dist≈0, approaches 0 as dist grows.
        """
        z = m * dist + b
        return torch.sigmoid(z)

    def err_dist(self, new_pose, m: float = -1.0, b: float = 1.0) -> torch.Tensor:
        """
        Self-collision penalty.
        Adds a hard crash weight if the robot is in collision.
        """
        self.rob.q_vect = self._to_tensor(new_pose)
        crash, min_dist = self.rob.do_fk_and_check_crash()

        crash_value = torch.tensor(self.crash_w if crash else 0.0, dtype=torch.float64)

        norm_dist  = self.get_normal_link_dist(min_dist)
        dist_value = self.dist_w * self.dist_sigmoid(dist=norm_dist, m=m, b=b)

        if crash:
            self.crashed = True

        return crash_value + dist_value


    # ------------------------------------------------------------------
    # Combined Loss
    # ------------------------------------------------------------------
    def Loss(self,
             delta_q_prev,
             q_curr,
             delta_q_next,
             pos_G_workspace,
             ori_G_SO3) -> torch.Tensor:
        """
        Full loss used by both the Oracle and the Neural Network.

        Args:
            delta_q_prev    : previous joint velocity  (n,)
            q_curr          : current joint angles      (n,)
            delta_q_next    : predicted joint velocity  (n,)  — the variable being optimised
            pos_G_workspace : goal EE position in world frame  (3,)
            ori_G_SO3       : goal rotation matrix              (3, 3)
        """
        delta_q_prev = self._to_tensor(delta_q_prev)
        q_curr       = self._to_tensor(q_curr)
        delta_q_next = self._to_tensor(delta_q_next)
        pos_G_workspace = self._to_tensor(pos_G_workspace)
        ori_G_SO3    = self._to_tensor(ori_G_SO3)

        new_pose = q_curr + delta_q_next

        self.rob.q_vect = new_pose
        new_position    = self.rob.give_ds()[-1]
        new_orientation = self.rob.give_Rs()[-1]

        e_pos  = self.err_pos(pos_curr=new_position, pos_G_ws=pos_G_workspace)
        e_ori  = self.err_ori(new_ori_SO3=new_orientation, G_SO3=ori_G_SO3)
        e_vel  = self.err_vel(joint_velocity=delta_q_next)
        e_acc  = self.err_acc(joint_vel_new=delta_q_next, joint_vel_old=delta_q_prev)
        e_dist = self.err_dist(new_pose=new_pose, m=-1.0, b=1.0)

        # print(f'new_position = {new_position}')
        # print(f'e_pos  = {e_pos.item():.6f}')
        # print(f'e_ori  = {e_ori.item():.6f}')
        # print(f'e_vel  = {e_vel.item():.6f}')
        # print(f'e_acc  = {e_acc.item():.6f}')
        # print(f'e_dist = {e_dist.item():.6f}')

        my_error = e_pos + e_ori + e_vel + e_acc + e_dist
        
        self.Pass = self.get_pass_or_fail(e_pos= e_pos, e_ori= e_ori)

        return my_error
    


    
    # --------------------------------------------------------------------------
    # Automatically Detect if IK reached an acceptable closeness to the target
    # --------------------------------------------------------------------------
    def set_pass_thresholds(self, max_good_dist, max_good_deg, max_ok_dist, max_ok_deg):
        max_good_dist = torch.tensor([max_good_dist])
        self.max_good_err_pos = self.pos_w * torch.linalg.norm((max_good_dist) / self.rob.max_reach)

        max_good_deg = Rx_SO3(theta_x_deg= max_good_deg)
        self.max_good_err_deg = self.rot_w * torch.linalg.matrix_norm(torch.eye(3) - max_good_deg)


        max_ok_dist = torch.tensor([max_ok_dist])
        self.max_ok_err_pos = self.pos_w * torch.linalg.norm((max_ok_dist) / self.rob.max_reach)

        max_ok_deg = Rx_SO3(theta_x_deg= max_ok_deg)
        self.max_ok_err_deg = self.rot_w * torch.linalg.matrix_norm(torch.eye(3) - max_ok_deg)
        
        

    def get_pass_or_fail(self, e_pos, e_ori):
        '''
        Solutions are within an ideal distance and angle from target
        '''
        if e_pos + e_ori > self.max_good_err_pos + self.max_good_err_deg: 
            # did not meet criteria for being 'good', check if it meets criteria for being 'okay'
            self.get_okay_pass_or_fail(e_pos= e_pos, e_ori= e_ori)
        self.target_weight = 1 # this is a good datapoint, loss should be multiplied by 1 for training    
        return True
    

    def get_okay_pass_or_fail(self, e_pos, e_ori):
        '''
        Solutions are within an acceptable distance and angle from target
        '''
        if e_pos + e_ori > self.max_ok_err_pos + self.max_ok_err_deg:
            self.target_weight = 0
            return False # this is a bad datapoint, it should not be included in the dataset
        self.target_weight = 0.3 # this is an okay datapoint, loss should be multiplied by 0.3 for training
        return True