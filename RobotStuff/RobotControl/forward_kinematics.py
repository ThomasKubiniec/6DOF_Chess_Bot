"""
DH-based Forward Kinematics and Numerical Jacobians — pure PyTorch implementation.
"""
import torch

from scipy.optimize import minimize # to find maximum reach within joint ranges
from rot_math import YPR_SO3

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
                 pad_dist=None):

        # Store DH parameters as float64 tensors for numerical precision
        self.a     = torch.tensor(a,     dtype=torch.float64)
        self.alpha = torch.tensor(alpha, dtype=torch.float64)
        self.d     = torch.tensor(d,     dtype=torch.float64)
        self.theta = torch.tensor(theta, dtype=torch.float64)

        if WT is None:
            self.WT = torch.eye(4, dtype=torch.float64)
        else:
            self.WT = torch.tensor(WT, dtype=torch.float64)

        self.joint_type   = joint_type          # list of 'r' / 'p'
        self.joint_bounds = bounds              # list of (low, high) tuples  [used by scipy]
        self.low_bounds = [] # low bound tensor from joint bounds
        self.high_bounds = [] # high bound tensor from joint bounds
        self.get_tensor_bounds() # low bound and high bound tensors


        self.fail_dist_vect = torch.tensor(fail_dist, dtype=torch.float64) if fail_dist is not None else None
        self.pad_dist_vect  = torch.tensor(pad_dist,  dtype=torch.float64) if pad_dist  is not None else None

        self.q_vect = torch.zeros(len(a), dtype=torch.float64)

        self.curr_joint_H  = []   # list of 4×4 tensors, one per joint
        self.curr_jacobian = []   # list of Jacobian matrices

        # print("in initialize")
        self.get_max_reach() # the farthest the robot can reach in the workspace

        self.get_active_joints() # list of joints that aren't kinematic coincidents


    # ------------------------------------------------------------------
    # helpers to accept plain Python / NumPy arrays from scipy.optimize
    # ------------------------------------------------------------------
    def _to_tensor(self, x) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            return x.to(torch.float64)
        return torch.tensor(x, dtype=torch.float64)

    @property
    def bounds(self):
        """Alias so Loss_Math can do torch.tensor(self.rob.bounds)."""
        return self.joint_bounds
    
    def get_tensor_bounds(self):
        for bound in self.joint_bounds:
            self.low_bounds.append(bound[0])
            self.high_bounds.append(bound[1])

        self.low_bounds = torch.tensor(self.low_bounds, dtype=torch.float64)
        self.high_bounds = torch.tensor(self.high_bounds, dtype=torch.float64)

    # def delta_q_bounds(self, q_curr=None) -> list:
    def delta_q_bounds(self, q_curr=None) -> torch.Tensor:
        """
        Compute per-joint bounds for delta_q given the current joint angles,
        so that q_curr + delta_q always stays within the absolute joint limits.

        For joint i:
            delta_q_low[i]  = joint_bound_low[i]  - q_curr[i]
            delta_q_high[i] = joint_bound_high[i] - q_curr[i]

        This is the key fix over using the raw joint_bounds as optimizer bounds:
        the optimizer searches over delta_q, but the absolute limits must be
        enforced on q_curr + delta_q, not on delta_q alone.

        Args:
            q_curr: current joint angles (tensor or array). Defaults to self.q_vect.

        Returns:
            list of (low, high) tuples compatible with scipy.optimize.minimize bounds.
        """
        if q_curr is None:
            q_curr = self.q_vect
        q = self._to_tensor(q_curr)
        

        # return [
        #     (float(low - q[i]),  float(high - q[i]))
        #     for i, (low, high) in enumerate(self.joint_bounds)
        # ]
        
        my_vel_bounds = torch.column_stack([self.low_bounds - q, self.high_bounds - q])
        return my_vel_bounds


    # ------------------------------------------------------------------
    # DH homogeneous transform for one link
    # ------------------------------------------------------------------
    def build_DH_A(self, a_i, alpha_i, d_i, theta_i) -> torch.Tensor:
        sa = torch.sin(alpha_i)
        ca = torch.cos(alpha_i)
        st = torch.sin(theta_i)
        ct = torch.cos(theta_i)

        A = torch.stack([
            torch.stack([ct,       -st * ca,  st * sa,  a_i * ct]),
            torch.stack([st,        ct * ca, -ct * sa,  a_i * st]),
            torch.stack([torch.zeros_like(ct), sa, ca, d_i]),
            torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float64),
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
        """
        Return the translation vectors (3-D) of every frame:
        world-frame origin, then joint 1 … joint n.
        """
        self.FK()
        ds = [self.WT[:3, 3]]
        for H in self.curr_joint_H:
            ds.append(H[:3, 3])
        return ds

    def give_Rs(self) -> list:
        """
        Return the 3×3 rotation matrices of every frame:
        world frame, then joint 1 … joint n.
        """
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
        # Collect H matrices: [WT, H1, H2, ..., Hn]
        my_Hs = [self.WT] + list(self.curr_joint_H)

        os = [H[:3, 3] for H in my_Hs]   # origins
        zs = [H[:3, 2] for H in my_Hs]   # z-axes

        Jacobians = []
        for i in range(1, len(my_Hs)):
            cols = []
            for j in range(i):
                if self.joint_type[j] == 'r':
                    Jv = torch.linalg.cross(zs[j], os[i] - os[j])
                    Jw = zs[j]
                elif self.joint_type[j] == 'p':
                    Jv = zs[j]
                    Jw = torch.zeros(3, dtype=torch.float64)
                else:
                    raise ValueError(f"Unknown joint type: {self.joint_type[j]!r}")

                cols.append(torch.cat([Jv, Jw]))   # shape (6,)

            Jacobians.append(torch.stack(cols, dim=1))   # shape (6, i)

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
            s = torch.tensor(0.0, dtype=torch.float64)
            t = torch.clamp(f / e, 0.0, 1.0)
        else:
            c = torch.dot(d1, r)
            if e < 1e-10:
                t = torch.tensor(0.0, dtype=torch.float64)
                s = torch.clamp(-c / a, 0.0, 1.0)
            else:
                b = torch.dot(d1, d2)
                denom = a * e - b * b
                if denom.abs() > 1e-10:
                    s = torch.clamp((b * f - c * e) / denom, 0.0, 1.0)
                else:
                    s = torch.tensor(0.0, dtype=torch.float64)

                t = (b * s + f) / e
                if t < 0:
                    t = torch.tensor(0.0, dtype=torch.float64)
                    s = torch.clamp(-c / a, 0.0, 1.0)
                elif t > 1:
                    t = torch.tensor(1.0, dtype=torch.float64)
                    s = torch.clamp((b - c) / a, 0.0, 1.0)

        closest_on_1 = p1 + s * d1
        closest_on_2 = p3 + t * d2
        my_min_dist = torch.linalg.vector_norm(closest_on_1 - closest_on_2)
        # print(f'my_min_dist = {my_min_dist}')
        return my_min_dist


    def get_active_joints(self):
        '''
        Sometimes joints are kinematically modeled as being in the same place even if they are not,
        simply because the resulting transformation is the same as if the joints were in the same place.

        Assuming the robot is in the home pose, and has been defined using Standard DH Convention at
        that home pose, this function finds frames whose position is NOT a duplicate of any earlier
        frame.  Those are the "active" (non-coincident) frame indices used by the collision detector.

        Bug fixed: the original code used `d not in list` which triggers an ambiguous-boolean
        RuntimeError on multi-element tensors.  We now compare with torch.allclose.
        '''
        self.q_vect = torch.zeros(len(self.a), dtype=torch.float64)  # ensure home pose
        my_ds = self.give_ds()

        first_instance_idxs = []
        seen_positions = []

        for i, d in enumerate(my_ds):
            is_duplicate = any(torch.allclose(d, prev, atol=1e-6) for prev in seen_positions)
            if not is_duplicate:
                seen_positions.append(d)
                first_instance_idxs.append(i)

        self.active_joints = first_instance_idxs  # frame indices into give_ds() output




    def check_self_collision(self, joint_positions: list, link_radii: torch.Tensor,
                             skip_adjacent: bool = True):
        """
        Find collisions through capsule approximation.

        joint_positions : list of 3-D tensors [j0, j1, ..., jn]  (output of give_ds())
        link_radii      : radius of each capsule, indexed by ORIGINAL frame index
                          (same length as joint_positions - 1)
        Returns (collision: bool, min_dist: Tensor)

        Three bugs fixed vs. the original:
          1. get_active_joints used `d not in list` → ambiguous boolean on tensors.
             Fixed in get_active_joints() above.
          2. link_radii were indexed by filtered-link counter j instead of the
             original frame index active_joints[j].  When coincident frames are
             skipped the counter and the original index diverge, giving the wrong
             capsule radius to each link.
          3. The crash-detection return was commented out, so collisions were never
             reported.
        """
        # ── Build the filtered link list, preserving original frame indices ──
        # active_joints holds the frame indices (into joint_positions) that are
        # NOT duplicates of an earlier frame.  We use those indices to:
        #   (a) skip zero-length capsules caused by coincident DH frames, and
        #   (b) look up the correct capsule radius for each surviving link.
        #
        # Link k in the original chain spans frames k → k+1 and has radius
        # link_radii[k].  After filtering, a surviving link spans original frames
        # active_joints[j] → active_joints[j+1], so its radius is
        # link_radii[active_joints[j]]  (the start-frame index of that link).

        active = self.active_joints  # e.g. [0,1,2,4,6] for this robot

        my_links = []
        for j in range(len(active) - 1):
            orig_start = active[j]       # original frame index of link start
            orig_end   = active[j + 1]   # original frame index of link end
            p_start    = joint_positions[orig_start]
            p_end      = joint_positions[orig_end]
            radius     = link_radii[orig_start]   # ← Bug 2 fix: use orig index
            my_links.append((p_start, p_end, radius))

        # ── Compare every non-adjacent pair of filtered links ─────────────────
        # Skip self (i==j) and immediate neighbours (|i-j|==1); the slicing
        # L1 = my_links[:-2], L2 = my_links[2:] with the inner loop L2[j:]
        # gives exactly the upper-triangle with distance ≥ 2.
        min_dist = torch.tensor(1e5, dtype=torch.float64)
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
                if dist <= (l1[2] + l2[2]):   # ← Bug 3 fix: crash detection re-enabled
                    return True, min_dist

        return False, min_dist


    def do_fk_and_check_crash(self):
        """
        Run FK, update joint positions, and check for self-collision.
        Returns (crash: bool, min_dist: Tensor).
        """
        joint_pos = self.give_ds()
        crash, dist = self.check_self_collision(
            joint_positions=joint_pos,
            link_radii=self.fail_dist_vect,
        )
        return crash, dist
    

    # ------------------------------------------------------------------
    # Workspace normalisation
    # ------------------------------------------------------------------
    def _max_reach_cost(self, x):
        """scipy objective: maximise end-effector distance from base."""
        self.q_vect = torch.tensor(x, dtype=torch.float64)
        ee = self.give_ds()[-1]
        return -torch.linalg.vector_norm(ee).item()

    def get_max_reach(self):
        """Use L-BFGS-B to find the robot's maximum reach."""
        # print("in get_max_reach")
        x0 = torch.zeros(len(self.a)).numpy()
        res = minimize(fun=self._max_reach_cost,
                       x0=x0,
                       method="L-BFGS-B",
                       bounds=self.joint_bounds)
        self.q_vect = torch.tensor(res.x, dtype=torch.float64)
        self.max_reach  = torch.linalg.vector_norm(self.give_ds()[-1])
        # print(f'self.max_reach = {self.max_reach}')
        # restore home pose
        self.q_vect = torch.zeros(len(self.a), dtype=torch.float64)



    # ------------------------------------------------------------------
    # Homogeneous Transformation for setting workspace origin
    # ------------------------------------------------------------------
    def make_homogenous_transformation(self, yaw, pitch, roll, x, y, z):
        R =  YPR_SO3(yaw_deg= yaw, pitch_deg= pitch, roll_deg= roll)
        d = torch.tensor([x, y, z])
        homogeneous_row = torch.tensor([0, 0, 0, 1])

        H = torch.column_stack([R, d])
        H = torch.row_stack([H, homogeneous_row])

        return H