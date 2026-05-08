"""
Path Planning provides trajectories for the end-effector to get from one place to another.
MoveL interpolates a straight line between points.
MoveJ interpolates the joints directly to their new values.

GPU-aware: all tensor operations run on robot.device.
scipy calls (none currently, but watch out if you add them) would need .cpu() before .numpy().
"""

import torch
from forward_kinematics import Robot_math
from rot_math import to_6D_R


class PathPlannerMath:
    def __init__(self,
                 my_robot: Robot_math,
                 dtype=torch.float64,
                 device=None):

        self.robot  = my_robot
        self.dtype  = dtype
        # Inherit device from robot if not overridden
        self.device = device or (my_robot.device if my_robot is not None else
                                 torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        self.q_start = self.robot.q_vect if my_robot is not None else None
        self.q_end   = self.robot.q_vect if my_robot is not None else None

        self.tot_time = 1
        self.frames   = 20

    def _kw(self):
        """Shorthand keyword dict for tensor creation."""
        return dict(dtype=self.dtype, device=self.device)

    def cubic_interp_vect(self,
                          vect_0,
                          vect_1,
                          f: int = 20,
                          t: float = 1.0,
                          vect_vel_0=None,
                          vect_vel_1=None) -> torch.Tensor:
        """
        Cubically interpolates every component of vect_0 to vect_1 over f frames.

        Returns:
            Tensor of shape (f, n).
        """
        kw = self._kw()

        vect_0 = torch.as_tensor(vect_0, **kw)
        vect_1 = torch.as_tensor(vect_1, **kw)
        n = vect_0.shape[0]

        vect_vel_0 = (torch.zeros(n, **kw) if vect_vel_0 is None
                      else torch.as_tensor(vect_vel_0, **kw))
        vect_vel_1 = (torch.zeros(n, **kw) if vect_vel_1 is None
                      else torch.as_tensor(vect_vel_1, **kw))

        t0 = 0.0
        t1 = float(t)

        M = torch.tensor([
            [1, t0,   t0**2,    t0**3],
            [0,  1,  2*t0,   3*t0**2],
            [1, t1,   t1**2,    t1**3],
            [0,  1,  2*t1,   3*t1**2],
        ], **kw)

        b = torch.stack([vect_0, vect_vel_0, vect_1, vect_vel_1], dim=0)  # (4, n)
        C = torch.linalg.solve(M, b)                                        # (4, n)

        tau = torch.linspace(t0, t1, steps=f, **kw)
        T   = torch.stack([torch.ones_like(tau), tau, tau**2, tau**3], dim=1)  # (f, 4)

        return T @ C   # (f, n)

    def MoveL(self,
              tot_time,
              frames,
              q_init=None,
              q_end=None,
              start_pos_ori=None,
              end_pos_ori=None):
        """
        Move in a straight line from pose/orientation 1 to pose/orientation 2.
        Uses 6D rotation representation to prevent gimbal lock.
        """
        kw = self._kw()

        saved_q = self.robot.q_vect.clone()

        if q_init is not None:
            self.robot.q_vect = torch.as_tensor(q_init, **kw)
            my_start_pos = self.robot.give_ds()[-1]
            my_start_ori = to_6D_R(self.robot.give_Rs()[-1])
            start_pos_ori = torch.cat([my_start_pos, my_start_ori])

        if q_end is not None:
            self.robot.q_vect = torch.as_tensor(q_end, **kw)
            my_end_pos = self.robot.give_ds()[-1]
            my_end_ori = to_6D_R(self.robot.give_Rs()[-1])
            end_pos_ori = torch.cat([my_end_pos, my_end_ori])

        # Restore robot to its original pose — MoveL is read-only w.r.t. state
        self.robot.q_vect = saved_q

        if (q_init is None) and (start_pos_ori is None):
            print("Please enter a valid starting robot pose or coordinate set.")

        my_workspace_traj = self.cubic_interp_vect(
            vect_0=start_pos_ori, vect_vel_0=None,
            vect_1=end_pos_ori,   vect_vel_1=None,
            f=frames, t=tot_time,
        )

        my_time_delay = tot_time / frames
        return my_workspace_traj, my_time_delay, q_init

    def make_random_poses(self):
        low  = self.robot.low_bounds
        high = self.robot.high_bounds
        random_q = low + (high - low) * torch.rand(low.size(),
                                                    dtype=self.dtype,
                                                    device=self.device)
        return random_q

    def make_random_q_start(self):
        self.q_start = self.make_random_poses()

    def make_random_q_end(self):
        self.q_end = self.make_random_poses()

    def MoveL_mutable_start_stop(self):
        return self.MoveL(tot_time=self.tot_time, frames=self.frames,
                          q_init=self.q_start, q_end=self.q_end,
                          start_pos_ori=None, end_pos_ori=None)

    def random_MoveL(self, tot_time, frames):
        rand_q_init = self.make_random_poses()
        rand_q_end  = self.make_random_poses()
        return self.MoveL(tot_time=tot_time, frames=frames,
                          q_init=rand_q_init, q_end=rand_q_end,
                          start_pos_ori=None, end_pos_ori=None)


# ── Quick smoke test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {device}")

    planner = PathPlannerMath(my_robot=None, device=device)

    v0 = torch.tensor([0.0, 0.0, 0.0], device=device)
    v1 = torch.tensor([1.0, 2.0, 3.0], device=device)

    traj = planner.cubic_interp_vect(v0, v1, f=5, t=1.0)

    print("Trajectory shape:", traj.shape)
    print("Frame 0 (start):", traj[0])
    print("Frame 4 (end):  ", traj[-1])
    print("\nFull trajectory (f, n):")
    print(traj)