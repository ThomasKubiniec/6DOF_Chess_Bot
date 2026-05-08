"""
Imitation Learning based Inverse Kinematics.
Distill the iterative optimizer-based solutions into one forward pass to save time.

GPU-aware: the network and all tensor ops run on robot.device.
Fixed bugs vs. original:
  - `class IK(nn):`         → `class IK(nn.Module):`
  - `def __int__`           → `def __init__`
  - `super.__init__()`      → `super().__init__()`
  - `nn.Tanh(hid_dim, ...)` → `nn.Linear(hid_dim, output_dim)` + `nn.Tanh()`
    (nn.Tanh takes no arguments; the last linear projection was missing)
  - `nn.ModuleList` layers now use `nn.Sequential`-style forward with activations
"""

import torch
import torch.nn as nn

from forward_kinematics import Robot_math
from path_planning_math import PathPlannerMath
from expert_inv_kin import Oracle
from loss_math import Loss_Math
from rot_math import to_SO3, to_6D_R


class IK(nn.Module):
    def __init__(self,
                 robot: Robot_math,
                 hid_dim: int,
                 hid_layers: int,
                 pos_w=None, rot_w=None, crash_w=None, dist_w=None,
                 vel_lambda=None, acc_lambda=None):
        super().__init__()

        self.robot  = robot
        self.device = robot.device

        self.L = Loss_Math(my_robot=self.robot,
                           pos_w=pos_w, rot_w=rot_w,
                           crash_w=crash_w, dist_w=dist_w,
                           vel_lambda=vel_lambda, acc_lambda=acc_lambda)

        self.n          = len(self.robot.a)
        self.input_dim  = (2 * self.n) + 3 + 6  # 2×joints + xyz + 6D orientation
        self.output_dim = self.n

        # ── Build layers ──────────────────────────────────────────────
        layer_list = []

        # Input → first hidden
        layer_list.append(nn.Linear(self.input_dim, hid_dim))
        layer_list.append(nn.ReLU())

        # Hidden → hidden (hid_layers additional blocks)
        for _ in range(hid_layers):
            layer_list.append(nn.Linear(hid_dim, hid_dim))
            layer_list.append(nn.ReLU())

        # Hidden → output (Tanh squashes output to [-1, 1])
        layer_list.append(nn.Linear(hid_dim, self.output_dim))
        layer_list.append(nn.Tanh())

        self.net = nn.Sequential(*layer_list)

        # Move the whole network to the target device
        self.to(device=self.device, dtype=torch.float64)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input tensor (batch, input_dim):
            - previous joint velocity normalised       (n,)
            - current joint pose normalised            (n,)
            - distance from goal to end-effector norm  (3,)
            - rotation to goal in 6D-R representation  (6,)

        Output tensor (batch, n):
            - normalised joint velocity toward goal
        """
        return self.net(x)

    # ------------------------------------------------------------------
    def _build_input(self,
                     delta_q_prev: torch.Tensor,
                     q_curr: torch.Tensor,
                     goal_xyz: torch.Tensor,
                     goal_ori_6D: torch.Tensor) -> torch.Tensor:
        """
        Normalise raw inputs and concatenate into a single input vector.
        Mirrors the normalisation done in data_generator.normalize_inputs().
        """
        delta_q_N = self.L.get_normal_joint_vel(delta_q_prev)
        q_N       = self.L.get_normal_joint_value(q_curr)

        self.robot.q_vect = q_curr
        current_position  = self.robot.give_ds()[-1]
        dist_N            = self.L.get_normal_dist_to_goal(pos_curr=current_position,
                                                           pos_G_ws=goal_xyz)

        current_ori_SO3 = self.robot.give_Rs()[-1]
        goal_ori_SO3    = to_SO3(goal_ori_6D)
        rot_to_targ_SO3 = goal_ori_SO3.T @ current_ori_SO3
        rot_6D          = to_6D_R(rot_to_targ_SO3)

        return torch.cat([delta_q_N, q_N, dist_N, rot_6D])  # (input_dim,)

    def solve_IK(self,
                 delta_q_prev: torch.Tensor,
                 q_curr: torch.Tensor,
                 goal_xyz: torch.Tensor,
                 goal_ori_6D: torch.Tensor) -> torch.Tensor:
        """
        Normalise raw inputs → forward pass → rescale output back to joint-space.

        Returns delta_q in joint space (radians / metres).
        """
        # Move inputs to device
        def _t(v):
            return v.to(dtype=torch.float64, device=self.device)

        delta_q_prev = _t(delta_q_prev)
        q_curr       = _t(q_curr)
        goal_xyz     = _t(goal_xyz)
        goal_ori_6D  = _t(goal_ori_6D)

        x = self._build_input(delta_q_prev, q_curr, goal_xyz, goal_ori_6D)

        with torch.no_grad():
            delta_q_N = self.forward(x.unsqueeze(0)).squeeze(0)  # (n,)

        # Network output is in [-1, 1] normalised velocity space.
        # Invert get_normal_joint_vel: nv = 2*(dq - q_l)/(q_h - q_l) - 1
        # → dq = (nv + 1) / 2 * (q_h - q_l) + q_l
        q_l = self.L.q_l
        q_h = self.L.q_h
        delta_q = (delta_q_N + 1.0) / 2.0 * (q_h - q_l) + q_l

        return delta_q