"""
Reward functions for the TD3 Robot Arm Agent.
Pure PyTorch implementation — GPU-aware via Robot_math.device.

─────────────────────────────────────────────────────────────────────────────
Reward decomposition
─────────────────────────────────────────────────────────────────────────────

  r = r_pass + r_pos + r_ori + r_vel + r_acc + r_crash + r_timestep

  r_pass      = +1   if EE is within both pos AND ori threshold, else 0
  r_pos       = -clamp( ln(e_pos) + 2,  min=0, max=pos_w * 2 )
                  e_pos = pos_w * ‖xyz_goal − xyz_curr‖ / max_reach
  r_ori       = -clamp( ln(e_ori) + 2,  min=0, max=ori_w * 3 )
                  e_ori = ori_w * ‖R_goal − R_curr‖_F
  r_vel       = -‖Λ_vel @ δq_norm‖          (normalised joint velocity)
  r_acc       = -‖Λ_acc @ Δ(δq_norm)/2‖     (normalised joint acceleration)
  r_crash     = -crash_w  (hard penalty, only when capsule collision fires)
              +  dist_w * sigmoid( m * d_norm + b )  (soft proximity shaping)
                  NOTE: the sign convention makes r_crash always ≤ 0.
  r_timestep  = -1   every step (encourages fast solutions)

─────────────────────────────────────────────────────────────────────────────
Pass / fail grades (for HER target relabelling and logging)
─────────────────────────────────────────────────────────────────────────────

  GOOD  — e_pos_raw + e_ori_raw  ≤  good_pos_thresh + good_ori_thresh
  OK    — e_pos_raw + e_ori_raw  ≤  ok_pos_thresh   + ok_ori_thresh
  FAIL  — otherwise
"""

import torch
from rot_math import Rx_SO3
from forward_kinematics import Robot_math


# ─────────────────────────────────────────────────────────────────────────────
# Reward_Math
# ─────────────────────────────────────────────────────────────────────────────

class Reward_Math:
    """
    Compute per-step rewards for the TD3 robot arm agent.

    Parameters
    ----------
    my_robot   : Robot_math instance (provides FK, device, bounds, max_reach)
    pos_w      : weight applied to the positional error            (default 1.0)
    rot_w      : weight applied to the orientation error           (default 0.5)
    crash_w    : hard penalty applied on collision detection       (default 2.0)
    dist_w     : soft proximity weight (sigmoid proximity shaping) (default 0.1)
    vel_lambda : per-joint velocity penalty weights  list[float]   (default 0.1 each)
    acc_lambda : per-joint acceleration penalty weights list[float](default 0.1 each)
    """

    def __init__(self,
                 my_robot:   Robot_math,
                 pos_w:      float | None = None,
                 rot_w:      float | None = None,
                 crash_w:    float | None = None,
                 dist_w:     float | None = None,
                 vel_lambda: list  | None = None,
                 acc_lambda: list  | None = None):

        self.rob    = my_robot
        self.device = my_robot.device

        # ── Scalar weights ──────────────────────────────────────────────────
        self.pos_w   = float(pos_w   if pos_w   is not None else 1.0)
        self.rot_w   = float(rot_w   if rot_w   is not None else 0.5)
        self.crash_w = float(crash_w if crash_w is not None else 2.0)
        self.dist_w  = float(dist_w  if dist_w  is not None else 0.1)

        # ── Per-joint diagonal penalty matrices ─────────────────────────────
        n = len(self.rob.a)
        vel_lambda = vel_lambda if vel_lambda is not None else [0.1] * n
        acc_lambda = acc_lambda if acc_lambda is not None else [0.1] * n
        self._build_lambda_matrices(vel_lambda, acc_lambda)

        # ── Joint bound tensors ─────────────────────────────────────────────
        bounds = torch.tensor(self.rob.joint_bounds,
                              dtype=torch.float64, device=self.device)
        self.q_l = bounds[:, 0]   # (n,)  lower limits
        self.q_h = bounds[:, 1]   # (n,)  upper limits

        # ── Pass / fail thresholds (set sensible defaults; override via
        #    set_pass_thresholds if needed) ───────────────────────────────────
        self._good_pos_thresh = self._norm_pos_thresh(0.05)   # 5 cm
        self._ok_pos_thresh   = self._norm_pos_thresh(0.10)   # 10 cm
        self._good_ori_thresh = self._norm_ori_thresh(2.0)    # 2 deg
        self._ok_ori_thresh   = self._norm_ori_thresh(5.0)    # 5 deg

        # ── Episode state (reset every episode) ─────────────────────────────
        self.crashed = False
        self.passed  = False
        self.grade   = "FAIL"    # "GOOD" | "OK" | "FAIL"

    # ─────────────────────────────────────────────────────────────────────────
    # Construction helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _build_lambda_matrices(self, vel_lambda: list, acc_lambda: list):
        self.vel_lambda = torch.diag(
            torch.tensor(vel_lambda, dtype=torch.float64, device=self.device))
        self.acc_lambda = torch.diag(
            torch.tensor(acc_lambda, dtype=torch.float64, device=self.device))

    def _norm_pos_thresh(self, dist_m: float) -> torch.Tensor:
        """Convert a Cartesian distance (metres) to normalised positional error."""
        return self.pos_w * torch.tensor(dist_m, dtype=torch.float64,
                                         device=self.device) / self.rob.max_reach

    def _norm_ori_thresh(self, deg: float) -> torch.Tensor:
        """Convert a rotation angle (degrees) to weighted Frobenius norm threshold."""
        R_err = Rx_SO3(theta_x_deg=deg, device=self.device)
        I     = torch.eye(3, dtype=torch.float64, device=self.device)
        return self.rot_w * torch.linalg.matrix_norm(I - R_err)

    # ─────────────────────────────────────────────────────────────────────────
    # Public configuration
    # ─────────────────────────────────────────────────────────────────────────

    def set_pass_thresholds(self,
                            good_pos_m:  float = 0.05,
                            good_deg:    float = 2.0,
                            ok_pos_m:    float = 0.10,
                            ok_deg:      float = 5.0):
        """
        Override default pass/fail thresholds.

        Parameters
        ----------
        good_pos_m  : max Cartesian error for GOOD grade  (metres)
        good_deg    : max angular error for GOOD grade     (degrees)
        ok_pos_m    : max Cartesian error for OK grade     (metres)
        ok_deg      : max angular error for OK grade       (degrees)
        """
        self._good_pos_thresh = self._norm_pos_thresh(good_pos_m)
        self._ok_pos_thresh   = self._norm_pos_thresh(ok_pos_m)
        self._good_ori_thresh = self._norm_ori_thresh(good_deg)
        self._ok_ori_thresh   = self._norm_ori_thresh(ok_deg)

    def reset_episode(self):
        """Call at the start of every episode to clear crash / pass state."""
        self.crashed = False
        self.passed  = False
        self.grade   = "FAIL"

    # ─────────────────────────────────────────────────────────────────────────
    # Tensor cast helper
    # ─────────────────────────────────────────────────────────────────────────

    def _t(self, x) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            return x.to(dtype=torch.float64, device=self.device)
        return torch.tensor(x, dtype=torch.float64, device=self.device)

    # ─────────────────────────────────────────────────────────────────────────
    # Normalisation utilities  (reused by state builder and reward terms)
    # ─────────────────────────────────────────────────────────────────────────

    def normalize_to_range(self, x: torch.Tensor,
                           low: torch.Tensor,
                           high: torch.Tensor) -> torch.Tensor:
        """Map x ∈ [low, high] → [-1, 1]."""
        denom = torch.clamp(high - low, min=1e-8)
        return 2.0 * ((x - low) / denom) - 1.0

    def denormalize_from_range(self, x_norm: torch.Tensor,
                               low: torch.Tensor,
                               high: torch.Tensor) -> torch.Tensor:
        """Map x_norm ∈ [-1, 1] → [low, high]."""
        return ((x_norm + 1.0) * (high - low) / 2.0) + low

    def get_normal_joint_value(self, q: torch.Tensor) -> torch.Tensor:
        """Joint angles → [-1, 1]."""
        return self.normalize_to_range(q, self.q_l, self.q_h)

    def get_original_joint_value(self, q_norm: torch.Tensor) -> torch.Tensor:
        """[-1, 1] → original joint radians."""
        return self.denormalize_from_range(q_norm, self.q_l, self.q_h)

    def get_normal_joint_vel(self, delta_q: torch.Tensor) -> torch.Tensor:
        """
        Normalise joint velocity (delta_q) to [-1, 1].
        Extreme negative vel = low→high in one step; extreme positive = high→low.
        """
        low  = self.q_l - self.q_h
        high = self.q_h - self.q_l
        return self.normalize_to_range(delta_q, low, high)

    def get_normal_joint_acc(self,
                             delta_q_prev: torch.Tensor,
                             delta_q_new:  torch.Tensor) -> torch.Tensor:
        """Normalised change in normalised velocity → [-1, 1]."""
        nv_prev = self.get_normal_joint_vel(delta_q_prev)
        nv_new  = self.get_normal_joint_vel(delta_q_new)
        return (nv_new - nv_prev) / 2.0

    def get_normal_dist_to_goal(self,
                                pos_curr: torch.Tensor,
                                pos_goal: torch.Tensor) -> torch.Tensor:
        """
        Normalised direction vector from current EE to goal EE.
        Used directly as part of the state observation.

        Returns: (3,) tensor in normalised workspace units.
        """
        pos_curr = self._t(pos_curr)
        pos_goal = self._t(pos_goal)
        return (pos_goal - pos_curr) / self.rob.max_reach

    def get_normal_link_dist(self, link_dist: torch.Tensor) -> torch.Tensor:
        return self._t(link_dist) / self.rob.max_reach

    # ─────────────────────────────────────────────────────────────────────────
    # State vector builder
    # ─────────────────────────────────────────────────────────────────────────

    def build_state(self,
                    pos_curr:      torch.Tensor,   # (3,)   current EE position
                    pos_goal:      torch.Tensor,   # (3,)   goal EE position
                    R_goal_SO3:    torch.Tensor,   # (3,3)  goal rotation matrix
                    delta_q_prev:  torch.Tensor,   # (n,)   previous joint delta
                    ) -> torch.Tensor:
        """
        Assemble the TD3 state vector:

            s = [ normalized_dist_to_goal  (3,)
                | rotation_to_goal_6D      (6,)
                | prev_delta_q_normalized  (n,) ]

        Total dimension: 3 + 6 + n

        The caller is responsible for running FK before calling this so that
        pos_curr is the actual EE position for the current joint config.

        Parameters
        ----------
        pos_curr     : current end-effector position in workspace (3,)
        pos_goal     : goal   end-effector position in workspace  (3,)
        R_goal_SO3   : goal   rotation matrix SO3                 (3,3)
        delta_q_prev : previous joint velocity used               (n,)

        Returns
        -------
        state : (3 + 6 + n,) float64 tensor on self.device
        """
        pos_curr     = self._t(pos_curr)
        pos_goal     = self._t(pos_goal)
        R_goal_SO3   = self._t(R_goal_SO3)
        delta_q_prev = self._t(delta_q_prev)

        # Normalised displacement to goal  (3,)
        norm_dist = self.get_normal_dist_to_goal(pos_curr, pos_goal)

        # 6D rotation representation of goal: first two columns of R_goal  (6,)
        rot_6d = torch.cat([R_goal_SO3[:, 0], R_goal_SO3[:, 1]])

        # Previous action in normalised velocity space  (n,)
        prev_vel_norm = self.get_normal_joint_vel(delta_q_prev)

        return torch.cat([norm_dist, rot_6d, prev_vel_norm])  # (3+6+n,)

    def build_state_batch(self,
                          pos_curr_batch:     torch.Tensor,
                          pos_goal:           torch.Tensor,
                          R_goal_SO3:         torch.Tensor,
                          delta_q_prev_batch: torch.Tensor,
                          ) -> torch.Tensor:
        """
        Vectorised state builder for an entire episode (T steps) in one tensor op.

            s = [ norm_dist_to_goal (3,) | R_goal_6D (6,) | prev_vel_norm (n,) ]

        Parameters
        ----------
        pos_curr_batch     : (T, 3)           EE position at each step
        pos_goal           : (3,) or (T, 3)   goal position (broadcast if scalar)
        R_goal_SO3         : (3,3) or (T,3,3) goal rotation (broadcast if scalar)
        delta_q_prev_batch : (T, n)           previous joint deltas at each step

        Returns
        -------
        states : (T, 3 + 6 + n) float64 on self.device
        """
        T = pos_curr_batch.shape[0]
        pos_curr_batch     = pos_curr_batch.to(dtype=torch.float64, device=self.device)
        delta_q_prev_batch = delta_q_prev_batch.to(dtype=torch.float64, device=self.device)

        if pos_goal.dim() == 1:
            pos_goal = pos_goal.unsqueeze(0).expand(T, -1).clone()
        if R_goal_SO3.dim() == 2:
            R_goal_SO3 = R_goal_SO3.unsqueeze(0).expand(T, -1, -1).clone()
        pos_goal   = pos_goal.to(dtype=torch.float64, device=self.device)
        R_goal_SO3 = R_goal_SO3.to(dtype=torch.float64, device=self.device)

        norm_dist = (pos_goal - pos_curr_batch) / self.rob.max_reach        # (T, 3)
        rot_6d    = torch.cat([R_goal_SO3[:, :, 0],
                               R_goal_SO3[:, :, 1]], dim=1)                 # (T, 6)
        low  = self.q_l - self.q_h
        high = self.q_h - self.q_l
        prev_vel_norm = (2.0 * ((delta_q_prev_batch - low)
                                / (high - low).clamp(min=1e-8)) - 1.0)     # (T, n)

        return torch.cat([norm_dist, rot_6d, prev_vel_norm], dim=1)         # (T, S)


    # ─────────────────────────────────────────────────────────────────────────
    # Individual reward terms
    # ─────────────────────────────────────────────────────────────────────────

    def r_pos(self, pos_curr: torch.Tensor,
              pos_goal: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Positional reward — quadratic shaping.

            e_pos  = pos_w * ‖pos_goal − pos_curr‖ / max_reach
            r_pos  = -clamp( e_pos²,  min=0, max=pos_w * 2 )

        Properties:
          • Passes through exactly 0 at e_pos = 0 (penalty fully off at perfect IK).
          • Gradient = -2 * e_pos, so the signal *sharpens* as the agent converges —
            giving maximum precision guidance in the regime that matters most.
          • Clamp prevents the quadratic from producing very large penalties at large
            errors; the agent still gets a consistent signal everywhere.

        Returns
        -------
        r_pos_val  : scalar reward
        e_pos_raw  : unweighted normalised positional error (for grading)
        """
        eps       = self.get_normal_dist_to_goal(pos_curr, pos_goal)
        e_pos_raw = torch.linalg.vector_norm(eps)           # unweighted (0–1 scale)
        e_pos     = self.pos_w * e_pos_raw

        r = -torch.clamp(e_pos ** 2, min=0.0, max=self.pos_w * 2.0)
        return r, e_pos_raw

    def r_ori(self, R_curr_SO3: torch.Tensor,
              R_goal_SO3: torch.Tensor,
              e_pos_raw:  torch.Tensor | None = None
              ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Orientation reward — quadratic shaping with optional focal coupling.

        When e_pos_raw is supplied the orientation weight is amplified as
        position error falls to zero, so the agent focuses on orientation
        once it is nearly in position (focal shaping).

            e_ori     = eff_rot_w * ‖R_goal − R_curr‖_F
            r_ori     = -clamp( e_ori²,  min=0, max=rot_w * 3 )

        Properties:
          • Passes through exactly 0 at perfect alignment.
          • The Frobenius norm saturates naturally around 2√2 at 180° error,
            so the quadratic stays bounded even without the weight; the clamp
            is still kept as a hard ceiling.
          • Gradient sharpens near zero — same benefit as r_pos.

        Returns
        -------
        r_ori_val  : scalar reward
        e_ori_raw  : unweighted Frobenius norm (for grading)
        """
        R_curr    = self._t(R_curr_SO3)
        R_goal    = self._t(R_goal_SO3)
        e_ori_raw = torch.linalg.matrix_norm(R_goal - R_curr)   # Frobenius

        if e_pos_raw is not None:
            # Focal coupling: boost orientation weight as position error shrinks
            delta     = 1e-2
            k         = 1.0
            k_max     = 10.0
            w_mult    = torch.clamp(k / (e_pos_raw + delta), min=1.0, max=k_max)
            eff_rot_w = self.rot_w * w_mult
        else:
            eff_rot_w = self.rot_w

        e_ori = eff_rot_w * e_ori_raw
        r     = -torch.clamp(e_ori ** 2, min=0.0, max=self.rot_w * 3.0)
        return r, e_ori_raw

    def r_vel(self, delta_q: torch.Tensor) -> torch.Tensor:
        """
        Velocity smoothness reward.

            r_vel = -‖Λ_vel @ δq_norm‖

        Penalises large joint motions.
        """
        nv = self.get_normal_joint_vel(self._t(delta_q))
        return -torch.linalg.vector_norm(self.vel_lambda @ nv)

    def r_acc(self, delta_q_new: torch.Tensor,
              delta_q_prev: torch.Tensor) -> torch.Tensor:
        """
        Acceleration smoothness reward.

            r_acc = -‖Λ_acc @ Δ(δq_norm)/2‖

        Penalises abrupt changes in joint velocity (jerk avoidance).
        """
        na = self.get_normal_joint_acc(delta_q_prev=self._t(delta_q_prev),
                                       delta_q_new=self._t(delta_q_new))
        return -torch.linalg.vector_norm(self.acc_lambda @ na)

    def r_crash(self, q_new: torch.Tensor,
                sigmoid_m: float = -1.0,
                sigmoid_b: float =  1.0) -> torch.Tensor:
        """
        Collision reward.

        Sets self.crashed = True when a capsule collision is detected.

            soft_proximity  = dist_w * sigmoid( m * d_norm + b )
                             (positive when far, shrinks toward 0 when close,
                              flips negative if d_norm is tiny — hence always ≤ 0
                              relative to the far-field value)
            r_crash = -crash_w * crashed  -  soft_proximity

        The soft term provides a continuous gradient pushing the arm away from
        near-miss configurations even before an actual collision fires.
        """
        old_q          = self.rob.q_vect.clone()
        self.rob.q_vect = self._t(q_new)
        crash, min_dist = self.rob.do_fk_and_check_crash()
        self.rob.q_vect = old_q

        if crash:
            self.crashed = True

        hard = torch.tensor(self.crash_w if crash else 0.0,
                            dtype=torch.float64, device=self.device)
        d_norm = self.get_normal_link_dist(min_dist)
        soft   = self.dist_w * torch.sigmoid(
                    torch.tensor(sigmoid_m, dtype=torch.float64, device=self.device) * d_norm
                    + torch.tensor(sigmoid_b, dtype=torch.float64, device=self.device))

        return -(hard + soft)

    def r_pass(self, e_pos_raw: torch.Tensor,
               e_ori_raw: torch.Tensor) -> torch.Tensor:
        """
        Sparse success reward: +1 on GOOD, 0 otherwise.
        Also updates self.grade and self.passed.
        """
        e_pos_w = self.pos_w * e_pos_raw
        e_ori_w = self.rot_w * e_ori_raw

        if e_pos_w <= self._good_pos_thresh and e_ori_w <= self._good_ori_thresh:
            self.grade  = "GOOD"
            self.passed = True
            return torch.tensor(1.0, dtype=torch.float64, device=self.device)

        if e_pos_w <= self._ok_pos_thresh and e_ori_w <= self._ok_ori_thresh:
            self.grade  = "OK"
            self.passed = False
            return torch.tensor(0.0, dtype=torch.float64, device=self.device)

        self.grade  = "FAIL"
        self.passed = False
        return torch.tensor(0.0, dtype=torch.float64, device=self.device)

    def r_sparse_batch(self,
                       pos_next_batch: torch.Tensor,   # (B, 3)
                       R_next_batch:   torch.Tensor,   # (B, 3, 3)
                       pos_goal_batch: torch.Tensor,   # (B, 3)  or (3,)
                       R_goal_batch:   torch.Tensor,   # (B,3,3) or (3,3)
                       ) -> torch.Tensor:
        """
        Batched sparse reward for TD3 + HER training.

            r = 0.0   if GOOD threshold reached  (episode success)
            r = -1.0  otherwise                  (timestep penalty)

        This is the pure sparse signal recommended by the HER literature.
        Dense reward terms (pos, ori, vel, acc) are omitted entirely —
        they distort HER relabelling and have been shown to reduce
        performance relative to sparse + HER in goal-conditioned tasks.

        Parameters
        ----------
        pos_next_batch : (B, 3)      EE position after action
        R_next_batch   : (B, 3, 3)   EE rotation after action
        pos_goal_batch : (B, 3) or (3,)   goal EE position
        R_goal_batch   : (B,3,3) or (3,3) goal EE rotation

        Returns
        -------
        r : (B,) float64 tensor of 0.0 or -1.0
        """
        B = pos_next_batch.shape[0]
        dev = self.device

        pos_next_batch = pos_next_batch.to(dtype=torch.float64, device=dev)
        R_next_batch   = R_next_batch.to(dtype=torch.float64, device=dev)

        if pos_goal_batch.dim() == 1:
            pos_goal_batch = pos_goal_batch.unsqueeze(0).expand(B, -1)
        if R_goal_batch.dim() == 2:
            R_goal_batch = R_goal_batch.unsqueeze(0).expand(B, -1, -1)
        pos_goal_batch = pos_goal_batch.to(dtype=torch.float64, device=dev)
        R_goal_batch   = R_goal_batch.to(dtype=torch.float64, device=dev)

        # Positional error (normalised)
        eps       = (pos_goal_batch - pos_next_batch) / self.rob.max_reach  # (B,3)
        e_pos_raw = torch.linalg.vector_norm(eps, dim=1)                    # (B,)

        # Orientation error (Frobenius norm)
        e_ori_raw = torch.linalg.matrix_norm(R_goal_batch - R_next_batch)  # (B,)

        # Success mask — both pos and ori within GOOD thresholds
        success = ((self.pos_w * e_pos_raw <= self._good_pos_thresh) &
                   (self.rot_w * e_ori_raw <= self._good_ori_thresh))       # (B,) bool

        # r = 0.0 on success, -1.0 otherwise
        r = torch.where(success,
                        torch.zeros(B, dtype=torch.float64, device=dev),
                        -torch.ones(B, dtype=torch.float64, device=dev))
        return r

    # ─────────────────────────────────────────────────────────────────────────
    # Combined reward  (single-step, called inside the environment step)
    # ─────────────────────────────────────────────────────────────────────────

    def reward(self,
               q_new:        torch.Tensor,   # (n,)   new joint angles after applying delta_q
               delta_q_new:  torch.Tensor,   # (n,)   joint delta just applied
               delta_q_prev: torch.Tensor,   # (n,)   joint delta from the previous step
               pos_goal:     torch.Tensor,   # (3,)   goal EE position (world frame)
               R_goal_SO3:   torch.Tensor,   # (3,3)  goal rotation matrix
               use_focal:    bool = True,
               ) -> tuple[torch.Tensor, dict]:
        """
        Compute the total reward for one environment step.

        The FK is run on q_new internally (without mutating self.rob.q_vect
        permanently) so the caller does not need to manage robot state.

        Parameters
        ----------
        q_new        : joint angles after applying this step's delta_q   (n,)
        delta_q_new  : joint velocity (action) applied this step         (n,)
        delta_q_prev : joint velocity applied the previous step          (n,)
        pos_goal     : goal end-effector position in world frame          (3,)
        R_goal_SO3   : goal rotation matrix                               (3,3)
        use_focal    : couple orientation weight to positional closeness  (bool)

        Returns
        -------
        r_total : scalar float64 tensor
        info    : dict with individual components for logging / debugging
        """
        q_new        = self._t(q_new)
        delta_q_new  = self._t(delta_q_new)
        delta_q_prev = self._t(delta_q_prev)
        pos_goal     = self._t(pos_goal)
        R_goal_SO3   = self._t(R_goal_SO3)

        # ── Forward kinematics (non-mutating) ──────────────────────────────
        old_q           = self.rob.q_vect.clone()
        self.rob.q_vect = q_new
        pos_curr        = self.rob.give_ds()[-1]    # (3,)
        R_curr_SO3      = self.rob.give_Rs()[-1]    # (3,3)
        self.rob.q_vect = old_q

        # ── Reward terms ────────────────────────────────────────────────────
        rv_pos,  e_pos_raw = self.r_pos(pos_curr, pos_goal)
        rv_ori,  e_ori_raw = self.r_ori(R_curr_SO3, R_goal_SO3,
                                         e_pos_raw if use_focal else None)
        rv_pass            = self.r_pass(e_pos_raw, e_ori_raw)
        rv_vel             = self.r_vel(delta_q_new)
        rv_acc             = self.r_acc(delta_q_new, delta_q_prev)
        rv_crash           = self.r_crash(q_new)       # also sets self.crashed
        rv_time            = torch.tensor(-1.0, dtype=torch.float64, device=self.device)

        r_total = rv_pass + rv_pos + rv_ori + rv_vel + rv_acc + rv_crash + rv_time

        info = {
            "r_total":   r_total.item(),
            "r_pass":    rv_pass.item(),
            "r_pos":     rv_pos.item(),
            "r_ori":     rv_ori.item(),
            "r_vel":     rv_vel.item(),
            "r_acc":     rv_acc.item(),
            "r_crash":   rv_crash.item(),
            "r_time":    rv_time.item(),
            "e_pos_raw": e_pos_raw.item(),
            "e_ori_raw": e_ori_raw.item(),
            "crashed":   self.crashed,
            "grade":     self.grade,
        }
        return r_total, info

    # ─────────────────────────────────────────────────────────────────────────
    # Batched reward (for offline replay / loss diagnostics, no FK or crash)
    # ─────────────────────────────────────────────────────────────────────────

    def reward_batch(self,
                     pos_curr_batch:    torch.Tensor,   # (B, 3)
                     R_curr_SO3_batch:  torch.Tensor,   # (B, 3, 3)
                     delta_q_new_batch: torch.Tensor,   # (B, n)
                     delta_q_prev_batch:torch.Tensor,   # (B, n)
                     pos_goal_batch:    torch.Tensor,   # (B, 3)
                     R_goal_SO3_batch:  torch.Tensor,   # (B, 3, 3)
                     use_focal:         bool = True,
                     ) -> tuple[torch.Tensor, dict]:
        """
        Vectorised reward computation over a batch — used for logging /
        sanity-checking sampled transitions.  Crash detection is OMITTED
        because capsule collision checking is inherently sequential.

        Returns
        -------
        r_total_batch : (B,) scalar rewards
        info          : dict of (B,) tensors for each component
        """
        B = pos_curr_batch.shape[0]

        pos_curr_batch     = pos_curr_batch.to(torch.float64)
        R_curr_SO3_batch   = R_curr_SO3_batch.to(torch.float64)
        delta_q_new_batch  = delta_q_new_batch.to(torch.float64)
        delta_q_prev_batch = delta_q_prev_batch.to(torch.float64)
        pos_goal_batch     = pos_goal_batch.to(torch.float64)
        R_goal_SO3_batch   = R_goal_SO3_batch.to(torch.float64)

        # ── Positional reward ───────────────────────────────────────────────
        eps_batch    = (pos_goal_batch - pos_curr_batch) / self.rob.max_reach   # (B,3)
        e_pos_raw_b  = torch.linalg.vector_norm(eps_batch, dim=1)               # (B,)
        e_pos_b      = self.pos_w * e_pos_raw_b
        rv_pos_b     = -torch.clamp(e_pos_b ** 2,
                                    min=0.0, max=self.pos_w * 2.0)              # (B,)

        # ── Orientation reward ──────────────────────────────────────────────
        ori_diff_b   = R_goal_SO3_batch - R_curr_SO3_batch                      # (B,3,3)
        e_ori_raw_b  = torch.linalg.matrix_norm(ori_diff_b)                     # (B,)

        if use_focal:
            delta   = 1e-2
            k       = 1.0
            k_max   = 10.0
            w_mult  = torch.clamp(k / (e_pos_raw_b + delta), min=1.0, max=k_max)  # (B,)
            eff_rot_w = self.rot_w * w_mult                                         # (B,)
        else:
            eff_rot_w = self.rot_w

        e_ori_b      = eff_rot_w * e_ori_raw_b
        rv_ori_b     = -torch.clamp(e_ori_b ** 2,
                                    min=0.0, max=self.rot_w * 3.0)              # (B,)

        # ── Pass reward ─────────────────────────────────────────────────────
        e_pos_w_b   = self.pos_w * e_pos_raw_b
        e_ori_w_b   = self.rot_w * e_ori_raw_b
        rv_pass_b   = ((e_pos_w_b <= self._good_pos_thresh) &
                       (e_ori_w_b <= self._good_ori_thresh)).to(torch.float64)  # (B,)

        # ── Velocity reward ─────────────────────────────────────────────────
        low  = self.q_l - self.q_h                                              # (n,)
        high = self.q_h - self.q_l                                              # (n,)
        denom     = torch.clamp(high - low, min=1e-8)
        nv_batch  = 2.0 * ((delta_q_new_batch - low) / denom) - 1.0            # (B, n)
        Lv_batch  = nv_batch @ self.vel_lambda.T                                # (B, n)
        rv_vel_b  = -torch.linalg.vector_norm(Lv_batch, dim=1)                 # (B,)

        # ── Acceleration reward ─────────────────────────────────────────────
        nv_prev_b = 2.0 * ((delta_q_prev_batch - low) / denom) - 1.0           # (B, n)
        na_batch  = (nv_batch - nv_prev_b) / 2.0                               # (B, n)
        La_batch  = na_batch @ self.acc_lambda.T                                # (B, n)
        rv_acc_b  = -torch.linalg.vector_norm(La_batch, dim=1)                 # (B,)

        # ── Timestep penalty ────────────────────────────────────────────────
        rv_time_b = -torch.ones(B, dtype=torch.float64, device=self.device)    # (B,)

        # ── Total ────────────────────────────────────────────────────────────
        r_total_b = rv_pass_b + rv_pos_b + rv_ori_b + rv_vel_b + rv_acc_b + rv_time_b

        info = {
            "r_total":   r_total_b,
            "r_pass":    rv_pass_b,
            "r_pos":     rv_pos_b,
            "r_ori":     rv_ori_b,
            "r_vel":     rv_vel_b,
            "r_acc":     rv_acc_b,
            "r_time":    rv_time_b,
            "e_pos_raw": e_pos_raw_b,
            "e_ori_raw": e_ori_raw_b,
        }
        return r_total_b, info