"""
DH-based Forward Kinematics and Numerical Jacobians — pure PyTorch implementation.

GPU-aware: pass `device=torch.device("cuda")` (or let it auto-detect) at
construction time.  All internal tensors live on that device.
scipy.optimize.minimize requires CPU numpy arrays, so tensors are detached
and moved to CPU only at the scipy boundary (_max_reach_cost, delta_q_bounds).
"""
import torch

from scipy.optimize import minimize
from rot_math import YPR_SO3, DEVICE


class Robot_math:
    def __init__(self,
                 a,
                 alpha,
                 d,
                 theta,
                 WT=None,
                 joint_type=None,
                 bounds=None,
                 fail_dist=None,
                 pad_dist=None,
                 device=None):

        self.device = device or DEVICE

        # Store DH parameters as float64 tensors on the target device
        self.a     = torch.tensor(a,     dtype=torch.float64, device=self.device)
        self.alpha = torch.tensor(alpha, dtype=torch.float64, device=self.device)
        self.d     = torch.tensor(d,     dtype=torch.float64, device=self.device)
        self.theta = torch.tensor(theta, dtype=torch.float64, device=self.device)

        if WT is None:
            self.WT = torch.eye(4, dtype=torch.float64, device=self.device)
        else:
            self.WT = torch.tensor(WT, dtype=torch.float64, device=self.device)

        self.joint_type   = joint_type   # list of 'r' / 'p'
        self.joint_bounds = bounds       # list of (low, high) tuples  [used by scipy]

        self.low_bounds  = []
        self.high_bounds = []
        self.get_tensor_bounds()

        self.fail_dist_vect = (torch.tensor(fail_dist, dtype=torch.float64, device=self.device)
                               if fail_dist is not None else None)
        self.pad_dist_vect  = (torch.tensor(pad_dist,  dtype=torch.float64, device=self.device)
                               if pad_dist  is not None else None)

        self.q_vect = torch.zeros(len(a), dtype=torch.float64, device=self.device)

        self.curr_joint_H  = []
        self.curr_jacobian = []

        self.get_max_reach()
        self.get_active_joints()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _to_tensor(self, x) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            return x.to(dtype=torch.float64, device=self.device)
        return torch.tensor(x, dtype=torch.float64, device=self.device)

    @property
    def bounds(self):
        """Alias so Loss_Math can do torch.tensor(self.rob.bounds)."""
        return self.joint_bounds

    def get_tensor_bounds(self):
        for bound in self.joint_bounds:
            self.low_bounds.append(bound[0])
            self.high_bounds.append(bound[1])
        self.low_bounds  = torch.tensor(self.low_bounds,  dtype=torch.float64, device=self.device)
        self.high_bounds = torch.tensor(self.high_bounds, dtype=torch.float64, device=self.device)

    def delta_q_bounds(self, q_curr=None) -> torch.Tensor:
        """
        Per-joint bounds for delta_q given current joint angles, so that
        q_curr + delta_q always stays within absolute joint limits.

        Returns a CPU tensor (n, 2) because scipy needs plain numpy.
        """
        if q_curr is None:
            q_curr = self.q_vect
        q = self._to_tensor(q_curr)
        my_vel_bounds = torch.column_stack([self.low_bounds - q, self.high_bounds - q])
        return my_vel_bounds.cpu()   # scipy boundary — must be on CPU

    # ------------------------------------------------------------------
    # DH homogeneous transform for one link
    # ------------------------------------------------------------------
    def build_DH_A(self, a_i, alpha_i, d_i, theta_i) -> torch.Tensor:
        sa = torch.sin(alpha_i)
        ca = torch.cos(alpha_i)
        st = torch.sin(theta_i)
        ct = torch.cos(theta_i)

        zero = torch.zeros_like(ct)
        one  = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float64, device=self.device)

        A = torch.stack([
            torch.stack([ct,    -st * ca,  st * sa,  a_i * ct]),
            torch.stack([st,     ct * ca, -ct * sa,  a_i * st]),
            torch.stack([zero,   sa,       ca,        d_i]),
            one,
        ])
        return A

    # ------------------------------------------------------------------
    # Forward Kinematics
    # ------------------------------------------------------------------
    def FK(self) -> list:
        """
        Returns a list of 4×4 homogeneous transforms, one per joint,
        expressed in the world frame.
        """
        q = self._to_tensor(self.q_vect)

        H = self.WT.clone()
        joint_FKs = []

        for i in range(len(self.a)):
            if self.joint_type[i] == 'r':
                t = self.theta[i] + q[i]
                d = self.d[i]
            elif self.joint_type[i] == 'p':
                t = self.theta[i]
                d = self.d[i] + q[i]
            else:
                raise ValueError(f"Unknown joint type: {self.joint_type[i]!r}")

            A = self.build_DH_A(a_i=self.a[i], alpha_i=self.alpha[i],
                                 d_i=d, theta_i=t)
            H = H @ A
            joint_FKs.append(H)

        self.curr_joint_H = joint_FKs
        return self.curr_joint_H

    # ------------------------------------------------------------------
    # Convenience extractors
    # ------------------------------------------------------------------
    def give_ds(self) -> list:
        """Translation vectors (3-D) of every frame."""
        self.FK()
        ds = [self.WT[:3, 3]]
        for H in self.curr_joint_H:
            ds.append(H[:3, 3])
        return ds

    def give_Rs(self) -> list:
        """3×3 rotation matrices of every frame."""
        self.FK()
        Rs = [self.WT[:3, :3]]
        for H in self.curr_joint_H:
            Rs.append(H[:3, :3])
        return Rs

    # ------------------------------------------------------------------
    # Geometric Jacobian
    # ------------------------------------------------------------------
    def J(self) -> list:
        """
        Geometric Jacobian for each intermediate frame.
        Returns a list of (6 × i) matrices, one per joint i = 1 … n.
        """
        my_Hs = [self.WT] + list(self.curr_joint_H)

        os = [H[:3, 3] for H in my_Hs]
        zs = [H[:3, 2] for H in my_Hs]

        Jacobians = []
        ee = os[-1]

        for i in range(1, len(my_Hs)):
            cols = []
            for j in range(i):
                if self.joint_type[j] == 'r':
                    Jv = torch.linalg.cross(zs[j], (ee - os[j]))
                    Jw = zs[j]
                else:
                    Jv = zs[j]
                    Jw = torch.zeros(3, dtype=torch.float64, device=self.device)
                cols.append(torch.cat([Jv, Jw]))
            Jacobians.append(torch.stack(cols, dim=1))

        self.curr_jacobian = Jacobians
        return Jacobians

    # ------------------------------------------------------------------
    # Collision detection
    # ------------------------------------------------------------------
    def closest_point_segment_segment(self,
                                       p1: torch.Tensor, p2: torch.Tensor,
                                       p3: torch.Tensor, p4: torch.Tensor) -> torch.Tensor:
        """Minimum distance between line segments p1-p2 and p3-p4."""
        d1 = p2 - p1
        d2 = p4 - p3
        r  = p1 - p3

        a = torch.dot(d1, d1)
        e = torch.dot(d2, d2)
        f = torch.dot(d2, r)

        if a < 1e-10 and e < 1e-10:
            return torch.linalg.vector_norm(r)

        if a < 1e-10:
            s = torch.tensor(0.0, dtype=torch.float64, device=self.device)
            t = torch.clamp(f / e, 0.0, 1.0)
        else:
            c = torch.dot(d1, r)
            if e < 1e-10:
                t = torch.tensor(0.0, dtype=torch.float64, device=self.device)
                s = torch.clamp(-c / a, 0.0, 1.0)
            else:
                b = torch.dot(d1, d2)
                denom = a * e - b * b
                if denom.abs() > 1e-10:
                    s = torch.clamp((b * f - c * e) / denom, 0.0, 1.0)
                else:
                    s = torch.tensor(0.0, dtype=torch.float64, device=self.device)

                t = (b * s + f) / e
                if t < 0:
                    t = torch.tensor(0.0, dtype=torch.float64, device=self.device)
                    s = torch.clamp(-c / a, 0.0, 1.0)
                elif t > 1:
                    t = torch.tensor(1.0, dtype=torch.float64, device=self.device)
                    s = torch.clamp((b - c) / a, 0.0, 1.0)

        closest_on_1 = p1 + s * d1
        closest_on_2 = p3 + t * d2
        return torch.linalg.vector_norm(closest_on_1 - closest_on_2)

    def get_active_joints(self):
        """
        Find frames whose position is NOT a duplicate of any earlier frame
        (coincident DH frames).  These are used by the collision detector.
        """
        self.q_vect = torch.zeros(len(self.a), dtype=torch.float64, device=self.device)
        my_ds = self.give_ds()

        first_instance_idxs = []
        seen_positions = []

        for i, d in enumerate(my_ds):
            is_duplicate = any(torch.allclose(d, prev, atol=1e-6) for prev in seen_positions)
            if not is_duplicate:
                seen_positions.append(d)
                first_instance_idxs.append(i)

        self.active_joints = first_instance_idxs

    def check_self_collision(self, joint_positions: list, link_radii: torch.Tensor,
                             skip_adjacent: bool = True):
        """
        Find collisions through capsule approximation.

        joint_positions : list of 3-D tensors [j0, j1, ..., jn]  (output of give_ds())
        link_radii      : radius of each capsule, indexed by ORIGINAL frame index
        Returns (collision: bool, min_dist: Tensor)
        """
        active = self.active_joints

        my_links = []
        for j in range(len(active) - 1):
            orig_start = active[j]
            orig_end   = active[j + 1]
            p_start    = joint_positions[orig_start]
            p_end      = joint_positions[orig_end]
            radius     = link_radii[orig_start]
            my_links.append((p_start, p_end, radius))

        min_dist = torch.tensor(1e5, dtype=torch.float64, device=self.device)
        L1 = my_links[:-2]
        L2 = my_links[2:]

        for j, l1 in enumerate(L1):
            for l2 in L2[j:]:
                dist = self.closest_point_segment_segment(
                    p1=l1[0], p2=l1[1],
                    p3=l2[0], p4=l2[1],
                )
                if dist < min_dist:
                    min_dist = dist
                if dist <= (l1[2] + l2[2]):
                    return True, min_dist

        return False, min_dist

    def do_fk_and_check_crash(self):
        """Run FK and check for self-collision. Returns (crash: bool, min_dist: Tensor)."""
        joint_pos = self.give_ds()
        crash, dist = self.check_self_collision(
            joint_positions=joint_pos,
            link_radii=self.fail_dist_vect,
        )
        return crash, dist

    # ------------------------------------------------------------------
    # Workspace normalisation  (scipy boundary — runs on CPU)
    # ------------------------------------------------------------------
    def _max_reach_cost(self, x):
        """scipy objective: maximise end-effector distance from base."""
        self.q_vect = torch.tensor(x, dtype=torch.float64, device=self.device)
        ee = self.give_ds()[-1]
        return -torch.linalg.vector_norm(ee).item()  # .item() gives plain Python float

    def get_max_reach(self):
        """Use L-BFGS-B to find the robot's maximum reach."""
        x0 = torch.zeros(len(self.a)).numpy()  # scipy needs CPU numpy
        res = minimize(fun=self._max_reach_cost,
                       x0=x0,
                       method="L-BFGS-B",
                       bounds=self.joint_bounds)
        self.q_vect  = torch.tensor(res.x, dtype=torch.float64, device=self.device)
        self.max_reach = torch.linalg.vector_norm(self.give_ds()[-1])
        self.q_vect  = torch.zeros(len(self.a), dtype=torch.float64, device=self.device)

    # ------------------------------------------------------------------
    # Homogeneous Transformation for setting workspace origin
    # ------------------------------------------------------------------
    def make_homogenous_transformation(self, yaw, pitch, roll, x, y, z):
        R = YPR_SO3(yaw_deg=yaw, pitch_deg=pitch, roll_deg=roll, device=self.device)
        d = torch.tensor([x, y, z], dtype=torch.float64, device=self.device)
        homogeneous_row = torch.tensor([0.0, 0.0, 0.0, 1.0],
                                       dtype=torch.float64, device=self.device)
        H = torch.column_stack([R, d])
        H = torch.row_stack([H, homogeneous_row])
        return H