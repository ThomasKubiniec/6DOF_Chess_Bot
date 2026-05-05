"""
Path Planning provides trajectories for the end-effector to get to one place to another.
MoveL interpolates a straight line between points.
MoveJ interpolates the joints directly to their new values.

PyTorch rewrite: all for-loops replaced with batched tensor ops.
"""

'''
Consider making this just a function instead of a whole class.
Unless you plan on making the class do MoveL and MoveJ
using a robot_math and a inverse kinematics solver
'''



import torch
from forward_kinematics import Robot_math
from rot_math import to_6D_R

class PathPlannerMath:
    def __init__(self, 
                 my_robot : Robot_math, 
                 dtype=torch.float64, 
                 device=None):
        
        self.robot = my_robot

        self.dtype = dtype
        self.device = device or torch.device("cpu")

        self.q_start = self.robot.q_vect
        self.q_end = self.robot.q_vect

        self.tot_time = 1
        self.frames = 20

    def cubic_interp_vect(self,
                          vect_0,
                          vect_1,
                          f: int = 20,
                          t: float = 1.0,
                          vect_vel_0=None,
                          vect_vel_1=None) -> torch.Tensor:
        """
        Cubically interpolates every component of vect_0 to vect_1 over f frames.

        Each component follows an independent cubic polynomial fitted to:
            position at t0, velocity at t0, position at t1, velocity at t1.

        Components are solved in one batched linalg call — no Python loops.

        Args:
            vect_0:     Start vector, shape (n,)
            vect_1:     End vector,   shape (n,)
            f:          Number of frames (timesteps)
            t:          Total duration
            vect_vel_0: Start velocities, shape (n,). Defaults to zeros.
            vect_vel_1: End velocities,   shape (n,). Defaults to zeros.

        Returns:
            Tensor of shape (f, n) — i.e. [[v_t0], [v_t1], ..., [v_tf]].
            Access the vector at frame i as result[i].
            Access component j across all frames as result[:, j].
        """
        kw = dict(dtype=self.dtype, device=self.device)

        vect_0 = torch.as_tensor(vect_0, **kw)          # (n,)
        vect_1 = torch.as_tensor(vect_1, **kw)          # (n,)
        n = vect_0.shape[0]

        if vect_vel_0 is None:
            vect_vel_0 = torch.zeros(n, **kw)
        else:
            vect_vel_0 = torch.as_tensor(vect_vel_0, **kw)

        if vect_vel_1 is None:
            vect_vel_1 = torch.zeros(n, **kw)
        else:
            vect_vel_1 = torch.as_tensor(vect_vel_1, **kw)

        t0 = 0.0
        t1 = float(t)

        # ── Constraint matrix M (4×4), same for every component ──────────────
        # Rows encode: q(t0)=q0, q'(t0)=v0, q(t1)=q1, q'(t1)=v1
        M = torch.tensor([
            [1, t0,   t0**2,    t0**3],
            [0,  1,  2*t0,   3*t0**2],
            [1, t1,   t1**2,    t1**3],
            [0,  1,  2*t1,   3*t1**2],
        ], **kw)                                         # (4, 4)

        # ── Boundary conditions b, one column per component ───────────────────
        # b shape: (4, n)  →  rows = [q0, v0, q1, v1] for all n components
        b = torch.stack([vect_0, vect_vel_0, vect_1, vect_vel_1], dim=0)  # (4, n)

        # ── Solve M @ C = b for coefficient matrix C ──────────────────────────
        # torch.linalg.solve handles the (4,4) x (4,n) broadcast natively.
        # C shape: (4, n)  →  C[:, i] are the 4 coefficients for component i.
        C = torch.linalg.solve(M, b)                    # (4, n)

        # ── Build the time Vandermonde matrix T ───────────────────────────────
        # Each row is [1, t, t², t³] for one frame.
        tau = torch.linspace(t0, t1, steps=f, **kw)     # (f,)
        T = torch.stack([torch.ones_like(tau), tau, tau**2, tau**3], dim=1)  # (f, 4)

        # ── Evaluate all components at all timesteps in one matmul ────────────
        # (f, 4) @ (4, n) → (f, n)
        result = T @ C                                   # (f, n)

        return result # shape (f, n); result[i] = vector at frame i
    


    def MoveL(self,
              tot_time,
              frames,
              q_init= None,
              q_end= None, 
              start_pos_ori= None,
              end_pos_ori= None):
        '''
        move in a straight line from point and orinetation 1 to point and orientation 2

        use 6D Rotation representation to prevent gymbal lock.
        '''

        if q_init is not None:
            my_start_pos = self.robot.give_ds()[-1]
            my_start_ori = to_6D_R(self.robot.give_Rs()[-1])

            start_pos_ori = torch.cat([my_start_pos, my_start_ori])

        if q_end is not None:
            my_end_pos = self.robot.give_ds()[-1]
            my_end_ori = to_6D_R(self.robot.give_Rs()[-1])

            end_pos_ori = torch.cat([my_end_pos, my_end_ori])

        if (q_init is None) and (start_pos_ori is None):
            print(f'Please enter a valid starting robot pose or a valid starting coordinate set.\n pose was {q_init}\ncoordinates were {start_pos_ori}')
        

        my_workspace_traj = self.cubic_interp_vect(vect_0= start_pos_ori,
                                                   vect_vel_0= None,
                                                   vect_1= end_pos_ori,
                                                   vect_vel_1= None,
                                                   f= frames,
                                                   t= tot_time)
        
        my_time_delay = tot_time / frames
         # trajectory, time delay between frames of trajectory, starting pose of the trajectory if provided
        return my_workspace_traj, my_time_delay, q_init
    

    def make_random_poses(self):
        low = self.robot.low_bounds
        high = self.robot.high_bounds

        random_q = low + (high - low) * torch.rand(low.size())
        return random_q
    

    def make_random_q_start(self):
        self.q_start = self.make_random_poses()

    def make_random_q_end(self):
        self.q_end = self.make_random_poses()

    def MoveL_mutable_start_stop(self):
        workspace_traj, time_delay, initial_pose = self.MoveL(tot_time= self.tot_time,
                                                              frames= self.frames,
                                                              q_init= self.q_start,
                                                              q_end= self.q_end,
                                                              start_pos_ori= None,
                                                              end_pos_ori= None)
        return workspace_traj, time_delay, initial_pose

    def random_MoveL(self, tot_time, frames):
        rand_q_init = self.make_random_poses()
        rand_q_end = self.make_random_poses()

        workspace_traj, time_delay, initial_pose = self.MoveL(tot_time= tot_time,
                                                              frames= frames,
                                                              q_init= rand_q_init,
                                                              q_end= rand_q_end,
                                                              start_pos_ori= None,
                                                              end_pos_ori= None)
        
        return workspace_traj, time_delay, initial_pose






# ── Quick smoke test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    planner = PathPlannerMath(my_robot=None)

    v0 = torch.tensor([0.0, 0.0, 0.0])
    v1 = torch.tensor([1.0, 2.0, 3.0])

    traj = planner.cubic_interp_vect(v0, v1, f=5, t=1.0)

    print("Trajectory shape:", traj.shape)   # (5, 3)
    print("Frame 0 (start):", traj[0])       # ≈ [0, 0, 0]
    print("Frame 4 (end):  ", traj[-1])      # ≈ [1, 2, 3]
    print()
    print("Full trajectory (f, n):")
    print(traj)