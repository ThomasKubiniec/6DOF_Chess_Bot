'''
Traditional IK neural network:
    Predicts a delta_q from (current_pose, goal_xyz, goal_ori_6D),
    adds it to the normalised current pose, denormalises, then runs FK
    to get the found (xyz, SO3) — loss is computed on those coordinates.

Input  : (q_N, dist_to_goal_N, rot_to_goal_6D)  →  n + 3 + 6  values
Output : delta_q_N                               →  n           values
'''

import torch
import torch.nn as nn

from forward_kinematics import Robot_math
from loss_math import Loss_Math
from rot_math import to_6D_R, to_6D_R_batch


class simpleIK(nn.Module):
    def __init__(self,
                 robot: Robot_math,
                 device: torch.device,
                 hid_dim: int = 256,
                 hid_layers: int = 4):
        super().__init__()

        self.robot  = robot
        self.device = device
        self.L      = Loss_Math(my_robot=self.robot)

        # Derive dims from robot
        self.n          = len(robot.bounds)          # number of joints
        self.input_dim  = self.n + 3 + 6            # q_N + dist_N(3) + rot_6D(6)
        self.output_dim = self.n                     # delta_q_N

        self.hid_dim = hid_dim
        self.hid_layers = hid_layers

        # ── Build layers ──────────────────────────────────────────────
        layers = [nn.Linear(self.input_dim, hid_dim), nn.ReLU()]

        for _ in range(hid_layers):
            layers += [nn.Linear(hid_dim, hid_dim), nn.ReLU()]

        # Tanh squashes output to [-1, 1], consistent with [-1, 1] normalisation
        layers += [nn.Linear(hid_dim, self.output_dim), nn.Tanh()]

        self.net = nn.Sequential(*layers)
        self.to(device=self.device, dtype=torch.float32)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        x : (B, input_dim)  — already-normalised input vector
        returns delta_q_N : (B, n)  in [-1, 1]
        '''
        x = x.to(torch.float32)
        return self.net(x)

    # ------------------------------------------------------------------
    # Input construction  (single-sample — used at inference time)
    # ------------------------------------------------------------------
    def _build_input(self,
                     q_curr:       torch.Tensor,   # (n,)
                     goal_xyz:     torch.Tensor,   # (3,)
                     goal_ori_SO3: torch.Tensor    # (3, 3)
                     ) -> torch.Tensor:            # (input_dim,)
        '''
        Normalise raw inputs and concatenate into a single input vector.
        Keeps raw coordinates out of the network; normalisation lives here.
        '''
        q_N = self.L.get_normal_joint_value(q_curr)                        # (n,)

        self.robot.q_vect = q_curr
        current_xyz     = self.robot.give_ds()[-1]                          # (3,)
        dist_N          = self.L.get_normal_dist_to_goal(
                              pos_curr=current_xyz, pos_G_ws=goal_xyz)      # (3,)

        current_ori_SO3 = self.robot.give_Rs()[-1]                          # (3,3)
        rot_to_targ     = goal_ori_SO3.T @ current_ori_SO3                  # (3,3)
        rot_6D          = to_6D_R(rot_to_targ)                              # (6,)

        return torch.cat([q_N, dist_N, rot_6D])                             # (input_dim,)

    # ------------------------------------------------------------------
    # Batched input construction  (used during training)
    # ------------------------------------------------------------------
    def build_input_batch(self,
                          q_batch:       torch.Tensor,   # (B, n)
                          goal_xyz_batch: torch.Tensor,  # (B, 3)
                          goal_SO3_batch: torch.Tensor   # (B, 3, 3)
                          ) -> torch.Tensor:             # (B, input_dim)
        '''
        Fully vectorised version of _build_input.
        Requires batched FK: give_ds_batch → (B, 3) and give_Rs_batch → (B, 3, 3).

        NOTE: your Robot_math will need give_ds_batch / give_Rs_batch methods
        that accept a (B, n) joint tensor and return batched positions / rotations.
        See the docstring on _batched_fk for the expected interface.
        '''
        B = q_batch.shape[0]

        # ── Normalise joint angles ──────────────────────────────────
        # get_normal_joint_value must accept (B, n) and return (B, n)
        q_N = self.L.get_normal_joint_value(q_batch)                        # (B, n)
        q_N = q_N.to(torch.float32)

        # ── Batched FK ──────────────────────────────────────────────
        curr_xyz_batch, curr_SO3_batch = self._batched_fk(q_batch)          # (B,3), (B,3,3)
        curr_xyz_batch = curr_xyz_batch.to(torch.float32)
        curr_SO3_batch = curr_SO3_batch.to(torch.float32)

        # ── Normalise distance to goal ──────────────────────────────
        # get_normal_dist_to_goal must accept (B,3),(B,3) and return (B,3)
        dist_N = self.L.get_normal_dist_to_goal(
                     pos_curr=curr_xyz_batch,
                     pos_G_ws=goal_xyz_batch).to(torch.float32)             # (B, 3)

        # ── 6D rotation representation ──────────────────────────────
        # rot_to_targ[i] = goal_SO3[i].T @ curr_SO3[i]
        # torch.bmm(A, B) = A @ B for batched matrices
        rot_to_targ = torch.bmm(
            goal_SO3_batch.transpose(1, 2),   # (B, 3, 3)  goal^T
            curr_SO3_batch                    # (B, 3, 3)  current
        )                                                                    # (B, 3, 3)

        # to_6D_R must accept (B, 3, 3) and return (B, 6)  — first two cols of R, row-major
        rot_6D = to_6D_R_batch(rot_to_targ)                                       # (B, 6)

        return torch.cat([q_N, dist_N, rot_6D], dim=1)                      # (B, input_dim)

    # ------------------------------------------------------------------
    # Batched FK helper
    # ------------------------------------------------------------------
    def _batched_fk(self,
                    q_batch: torch.Tensor          # (B, n)
                    ):
        '''
        Calls the batched FK methods on self.robot.

        Expected Robot_math interface:
            robot.give_ds_batch(q_batch)  → (B, 3)    end-effector XYZ
            robot.give_Rs_batch(q_batch)  → (B, 3, 3) end-effector SO3

        If your Robot_math does not yet have these methods, a simple
        fallback loops over the batch (slow but correct):

            xyz_list = []
            SO3_list = []
            for i in range(q_batch.shape[0]):
                self.robot.q_vect = q_batch[i]
                xyz_list.append(self.robot.give_ds()[-1])
                SO3_list.append(self.robot.give_Rs()[-1])
            return torch.stack(xyz_list), torch.stack(SO3_list)

        Implement give_ds_batch / give_Rs_batch in Robot_math for real
        speed gains — vectorising DH chain multiplication over a batch
        dimension with torch.bmm is straightforward.
        '''
        xyz = self.robot.give_ds_batch(q_batch)    # (B, 3)
        SO3 = self.robot.give_Rs_batch(q_batch)    # (B, 3, 3)
        return xyz, SO3

    # ------------------------------------------------------------------
    # IK solve — single sample, used at inference
    # ------------------------------------------------------------------
    def solve_IK(self,
                 q_curr:       torch.Tensor,   # (n,)
                 goal_xyz:     torch.Tensor,   # (3,)
                 goal_ori_SO3: torch.Tensor    # (3, 3)
                 ) -> torch.Tensor:            # (n,)  real joint angles
        '''
        Predicts a normalised delta, adds it to the normalised current pose,
        then denormalises back to real joint angles.
        Both q_N and delta_q_N are in [-1, 1], so addition is in the same space.
        '''
        x           = self._build_input(q_curr, goal_xyz, goal_ori_SO3)     # (input_dim,)
        delta_q_N   = self.forward(x.unsqueeze(0)).squeeze(0)               # (n,)

        q_curr_N    = self.L.get_normal_joint_value(q_curr)                 # (n,)
        q_pred_N    = (q_curr_N + delta_q_N).clamp(-1.0, 1.0)              # (n,) stay in range
        real_pose   = self.L.get_original_joint_value(q_norm=q_pred_N)     # (n,)
        return real_pose

    # ------------------------------------------------------------------
    # Batched IK + FK — used during training to get (xyz, SO3) predictions
    # ------------------------------------------------------------------
    def give_IK_found_pose_batch(self,
                                 q_batch:       torch.Tensor,   # (B, n)
                                 goal_xyz_batch: torch.Tensor,  # (B, 3)
                                 goal_SO3_batch: torch.Tensor   # (B, 3, 3)
                                 ) -> torch.Tensor:             # (B, 3 + 9)
        '''
        Full forward pass for training:
          1. Build normalised input batch
          2. Predict delta_q_N
          3. Add to normalised start pose, clamp, denormalise
          4. Run batched FK to get (xyz, SO3)
          5. Return concatenated (xyz, flattened SO3)  shape (B, 12)

        Loss is computed externally against the target (goal_xyz, goal_SO3).
        '''
        X          = self.build_input_batch(q_batch, goal_xyz_batch, goal_SO3_batch)  # (B, input_dim)
        delta_q_N  = self.forward(X)                                                   # (B, n)

        # Loss_Math / Robot_math operate in float64; cast each output back to
        # float32 immediately so the entire backward graph stays uniform.
        q_curr_N   = self.L.get_normal_joint_value(q_batch).to(torch.float32)            # (B, n)
        q_pred_N   = (q_curr_N + delta_q_N).clamp(-1.0, 1.0)                            # (B, n) stay in range
        q_pred     = self.L.get_original_joint_value(q_norm=q_pred_N).to(torch.float32)  # (B, n)

        pred_xyz, pred_SO3 = self._batched_fk(q_pred)                                    # (B,3), (B,3,3)

        pred_SO3_flat = pred_SO3.reshape(pred_SO3.shape[0], -1).to(torch.float32)        # (B, 9)
        return torch.cat([pred_xyz.to(torch.float32), pred_SO3_flat], dim=1)             # (B, 12)

    # ------------------------------------------------------------------
    # Single-sample inference convenience (unchanged interface)
    # ------------------------------------------------------------------
    def give_IK_found_pose(self,
                           q_curr:       torch.Tensor,
                           goal_xyz:     torch.Tensor,
                           goal_ori_SO3: torch.Tensor
                           ) -> torch.Tensor:                               # (12,)
        found_pose  = self.solve_IK(q_curr, goal_xyz, goal_ori_SO3)

        self.robot.q_vect = found_pose
        found_xyz   = self.robot.give_ds()[-1].to(torch.float32)
        found_SO3   = self.robot.give_Rs()[-1].to(torch.float32).flatten()

        return torch.cat([found_xyz, found_SO3])                            # (12,)