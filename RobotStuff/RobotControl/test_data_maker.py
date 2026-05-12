'''
Generate, solve, grade, and save chessboard trajectories for all three IK solvers.

Trajectory set per chess move (4 legs):
    Home → A  (descend to grasp height at source square)
    A    → B  (lift to clearance height at source)
    B    → C  (move across board at clearance height)
    C    → D  (descend to drop height at destination)

Saved structure (pickle, list of move_dicts):

    move_dict = {
        'waypoints' : [xyz_A, xyz_B, xyz_C, xyz_D],
        'legs'      : [
            {
                'oracle'    : [frame_dict, ...],
                'simple_ik' : [frame_dict, ...],
                'imit_ik'   : [frame_dict, ...],
            },
            ...
        ]
    }
'''

import torch
import numpy as np
import pickle

from chess_trajectories import chess_coords
from path_planning_math import PathPlannerMath
from rot_math import to_SO3, to_6D_R, Rx_SO3
from loss_math import Loss_Math
from forward_kinematics import Robot_math


class generate_test_data:

    def __init__(self,
                 robot            : Robot_math,
                 my_board         : chess_coords,
                 path_planner     : PathPlannerMath,
                 loss             : Loss_Math,
                 iterative_solver,
                 simpleIKNN,
                 imitationIKNN,
                 max_good_dist    = 0.1,
                 max_good_deg     = 2.0,
                 max_okay_dist    = 0.25,
                 max_okay_deg     = 6.0,
                 frames_per_leg   = 100,
                 dataset_filename = 'chessboard_test_dataset'):

        self.robot    = robot
        self.board    = my_board
        self.planner  = path_planner
        self.L        = loss

        self.oracle   = iterative_solver
        self.simp_IK  = simpleIKNN
        self.imit_IK  = imitationIKNN

        self.frames_per_leg   = frames_per_leg
        self.dataset_filename = dataset_filename

        # ============================================================
        # Use ONE dtype consistently for NN inference continuity
        # ============================================================
        self.dtype = torch.float32

        self.L.set_pass_thresholds(
            max_good_dist=max_good_dist,
            max_good_deg=max_good_deg,
            max_ok_dist=max_okay_dist,
            max_ok_deg=max_okay_deg
        )

        self.targ_SO3 = Rx_SO3(
            theta_x_deg=180,
            device=robot.device
        ).to(self.dtype)

        self.targ_6DR = to_6D_R(
            self.targ_SO3
        ).to(self.dtype)

        self.home_q = torch.tensor([
            np.deg2rad(0),
            np.deg2rad(-150),
            np.deg2rad(-74),
            np.deg2rad(0),
            np.deg2rad(-74),
            np.deg2rad(0),
        ], dtype=self.dtype, device=robot.device)

        self.saved_moves = []

    # ================================================================
    # Helpers
    # ================================================================
    def _xyz_to_tensor(self, xyz):

        if isinstance(xyz, torch.Tensor):
            return xyz.to(
                dtype=self.dtype,
                device=self.robot.device
            )

        return torch.tensor(
            xyz,
            dtype=self.dtype,
            device=self.robot.device
        )

    def _fk(self, q):

        saved = self.robot.q_vect.clone()

        # Avoid aliasing continuity bugs
        self.robot.q_vect = q.clone()

        xyz = self.robot.give_ds()[-1].clone()
        SO3 = self.robot.give_Rs()[-1].clone()

        self.robot.q_vect = saved

        return xyz, SO3

    def _grade(self, e_pos: float, e_ori: float):

        combined = e_pos + e_ori

        if combined <= float(
            self.L.max_good_err_pos +
            self.L.max_good_err_deg
        ):
            return 'good'

        elif combined <= float(
            self.L.max_ok_err_pos +
            self.L.max_ok_err_deg
        ):
            return 'okay'

        return 'bad'

    def _frame_dict(self,
                    q,
                    goal_xyz,
                    goal_SO3):

        found_xyz, found_SO3 = self._fk(q)

        e_pos = self.L.err_pos(
            pos_curr=found_xyz,
            pos_G_ws=goal_xyz
        ).item()

        e_ori = self.L.err_ori(
            new_ori_SO3=found_SO3,
            G_SO3=goal_SO3
        ).item()

        return {
            'q'       : q.detach().clone(),
            'xyz'     : found_xyz,
            'SO3'     : found_SO3,
            'quality' : self._grade(e_pos, e_ori),
            'e_pos'   : e_pos,
            'e_ori'   : e_ori,
        }

    # ================================================================
    # Trajectory generation
    # ================================================================
    def _make_leg(self,
                  start_xyz,
                  end_xyz):

        start_t = self._xyz_to_tensor(start_xyz)
        end_t   = self._xyz_to_tensor(end_xyz)

        start_6D = torch.cat([
            start_t,
            self.targ_6DR
        ])

        end_6D = torch.cat([
            end_t,
            self.targ_6DR
        ])

        traj, _, _ = self.planner.MoveL(
            tot_time      = 1.0,
            frames        = self.frames_per_leg,
            start_pos_ori = start_6D,
            end_pos_ori   = end_6D,
        )

        return traj

    def make_trajectory_set(self):

        waypoints = self.board.make_random_waypoints()

        home_xyz, _ = self._fk(self.home_q)

        home_xyz = tuple(home_xyz.tolist())

        all_points = [home_xyz] + list(waypoints)

        legs = []

        for i in range(4):

            legs.append(
                self._make_leg(
                    all_points[i],
                    all_points[i + 1]
                )
            )

        return waypoints, legs

    # ================================================================
    # Waypoint utilities
    # ================================================================
    def _goal_from_wp(self, wp):

        wp = wp.to(self.dtype)

        if wp.dim() > 1:
            wp = wp.squeeze(0)

        goal_xyz = wp[:3]

        if wp.shape[0] >= 9:

            goal_6D = wp[3:9]

            goal_SO3 = to_SO3(goal_6D)

        else:

            goal_SO3 = self.targ_SO3
            goal_6D  = self.targ_6DR

        return goal_xyz, goal_6D, goal_SO3

    @staticmethod
    def _iter_wps(leg_waypoints):

        import torch as _torch

        if isinstance(leg_waypoints, _torch.Tensor):

            for i in range(leg_waypoints.shape[0]):
                yield leg_waypoints[i]

        else:

            for wp in leg_waypoints:
                yield wp

    # ================================================================
    # Oracle Solver
    # ================================================================
    def _solve_leg_oracle(self,
                          leg_waypoints,
                          q_start):

        frames = []

        self.oracle.rob.q_vect = q_start.clone()
        self.oracle.reset_vars()

        for wp in self._iter_wps(leg_waypoints):

            goal_xyz, goal_6D, goal_SO3 = \
                self._goal_from_wp(wp)

            self.oracle.get_IK(
                Goal_Posi   = goal_xyz,
                Goal_Ori_6D = goal_6D,
            )

            q_next = self.oracle.rob.q_vect.clone()

            frames.append(
                self._frame_dict(
                    q_next,
                    goal_xyz,
                    goal_SO3
                )
            )

        return frames

    # ================================================================
    # Simple IK Solver
    # ================================================================
    def _solve_leg_simple_ik(self,
                             leg_waypoints,
                             q_start):

        frames = []

        q_curr = q_start.clone()

        for wp in self._iter_wps(leg_waypoints):

            goal_xyz, goal_6D, goal_SO3 = \
                self._goal_from_wp(wp)

            q_next = self.simp_IK.solve_IK(
                q_curr       = q_curr,
                goal_xyz     = goal_xyz,
                goal_ori_SO3 = goal_SO3,
            )

            frames.append(
                self._frame_dict(
                    q_next,
                    goal_xyz,
                    goal_SO3
                )
            )

            q_curr = q_next.detach().clone()

        return frames

    # ================================================================
    # Imitation IK Solver
    # ================================================================
    def _solve_leg_imit_ik(self,
                           leg_waypoints,
                           q_start,
                           delta_q_start):

        frames = []

        q_curr  = q_start.clone()
        delta_q = delta_q_start.clone()

        for wp in self._iter_wps(leg_waypoints):

            goal_xyz, goal_6D, goal_SO3 = \
                self._goal_from_wp(wp)

            q_next = self.imit_IK.solve_IK(
                delta_q_prev = delta_q,
                q_curr       = q_curr,
                goal_xyz     = goal_xyz,
                goal_ori_6D  = goal_6D,
            )

            delta_q = (
                q_next - q_curr
            ).detach()

            frames.append(
                self._frame_dict(
                    q_next,
                    goal_xyz,
                    goal_SO3
                )
            )

            q_curr = q_next.detach().clone()

        return frames, delta_q

    # ================================================================
    # Single Move
    # ================================================================
    def solve_move(self) -> dict:

        waypoints, legs = self.make_trajectory_set()

        move_legs = []

        # ============================================================
        # Independent continuity chains
        # ============================================================
        oracle_q = self.home_q.clone()
        simple_q = self.home_q.clone()
        imit_q   = self.home_q.clone()

        imit_delta_q = torch.zeros_like(self.home_q)

        for leg_idx, leg_wps in enumerate(legs):

            # --------------------------------------------------------
            # Oracle
            # --------------------------------------------------------
            oracle_frames = self._solve_leg_oracle(
                leg_wps,
                oracle_q
            )

            if oracle_frames:
                oracle_q = \
                    oracle_frames[-1]['q'].clone()

            # --------------------------------------------------------
            # Simple IK
            # --------------------------------------------------------
            simple_frames = self._solve_leg_simple_ik(
                leg_wps,
                simple_q
            )

            if simple_frames:
                simple_q = \
                    simple_frames[-1]['q'].clone()

            # --------------------------------------------------------
            # Imitation IK
            # --------------------------------------------------------
            imit_frames, imit_delta_q = \
                self._solve_leg_imit_ik(
                    leg_wps,
                    imit_q,
                    imit_delta_q
                )

            if imit_frames:
                imit_q = \
                    imit_frames[-1]['q'].clone()

            move_legs.append({
                'oracle'    : oracle_frames,
                'simple_ik' : simple_frames,
                'imit_ik'   : imit_frames,
            })

        return {
            'waypoints': waypoints,
            'legs': move_legs,
        }

    # ================================================================
    # Batch generation
    # ================================================================
    def generate_n_moves(self,
                         n: int,
                         verbose: bool = True):

        for i in range(n):

            move = self.solve_move()

            self.saved_moves.append(move)

            if verbose:
                self._print_move_summary(i + 1, move)

        print(
            f'\nDone. Total moves in memory: '
            f'{len(self.saved_moves)}'
        )

    def _print_move_summary(self,
                            idx: int,
                            move: dict):

        print(f'\n── Move {idx} {"─" * 40}')

        for leg_idx, leg in enumerate(move['legs']):

            for solver in (
                'oracle',
                'simple_ik',
                'imit_ik'
            ):

                frames = leg[solver]

                counts = {
                    'good': 0,
                    'okay': 0,
                    'bad': 0
                }

                for f in frames:
                    counts[f['quality']] += 1

                n = len(frames)

                print(
                    f'  Leg {leg_idx}  '
                    f'{solver:<12} '
                    f'good={counts["good"]}/{n}  '
                    f'okay={counts["okay"]}/{n}  '
                    f'bad={counts["bad"]}/{n}'
                )

    # ================================================================
    # Persistence
    # ================================================================
    def save(self,
             filename: str = None):

        path = f'{filename or self.dataset_filename}.pkl'

        with open(path, 'wb') as f:
            pickle.dump(self.saved_moves, f)

        print(
            f'Saved {len(self.saved_moves)} '
            f'moves → {path}'
        )

    @staticmethod
    def load(filename: str) -> list:

        path = f'{filename}.pkl'

        with open(path, 'rb') as f:
            data = pickle.load(f)

        print(
            f'Loaded {len(data)} '
            f'moves ← {path}'
        )

        return data
    

# ─────────────────────────────────────────────────────────────────────────────
# Entry point — edit the section below to swap models in/out
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import numpy as np
    import torch
    from forward_kinematics import Robot_math
    from path_planning_math import PathPlannerMath
    from chess_trajectories import chess_coords
    from loss_math import Loss_Math
    from train_simple_ik import build_robot
 
    # ── which models to include ───────────────────────────────────────────────
    # Set a flag to None to skip that solver (it will be absent from the saved
    # data; the visualizer will show "no data" for that solver).
    USE_ORACLE    = True
    USE_SIMPLE_IK = True
    USE_IMIT_IK   = True
 
    # ── model file paths ──────────────────────────────────────────────────────
    SIMPLE_IK_PATH = 'simp_ik_model_hdim256_hdep4.pt'
    IMIT_IK_PATH   = 'ik_model_hdim256_hdep4.pt'           # adjust to your filename
 
    # ── generation settings ───────────────────────────────────────────────────
    N_MOVES        = 20          # how many random chess moves to generate
    FRAMES_PER_LEG = 100         # MoveL waypoints per trajectory leg
    OUTPUT_FILE    = 'chessboard_test_dataset'
 
    # ── robot & path planner ─────────────────────────────────────────────────
    device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Running on: {device}')
 
    robot       = build_robot(device)
    path_planner = PathPlannerMath(my_robot=robot,
                                   dtype=torch.float32, # float64
                                   device=device)
 
    # ── board ────────────────────────────────────────────────────────────────
    # Adjust board dimensions and a1 origin to match your physical setup.
    board = chess_coords(
        board_width     = 12,
        board_length    = 12,
        board_thickness = 2,
        a1_xy_tup       = (4.0, -7.75),
    )
 
    # ── loss ─────────────────────────────────────────────────────────────────
    loss = Loss_Math(my_robot=robot)
 
    # ── oracle ───────────────────────────────────────────────────────────────
    oracle = None
    if USE_ORACLE:
        from expert_inv_kin import Oracle
        oracle = Oracle(robot_class=robot, 
                        pos_w= 100, rot_w=5,
                        crash_w= 2, dist_w= 0.5,
                        vel_lambda= [0.1] * 6,
                        acc_lambda= [0.1] * 6)
 
    # ── simple IK ────────────────────────────────────────────────────────────
    simple_ik = None
    if USE_SIMPLE_IK:
        from simple_ik_model_maker import simpleIK
 
        # Infer hid_dim and hid_layers from the checkpoint so the architecture
        # always matches regardless of what the model was trained with.
        # net.0 is Input->Hidden; hid_dim = out-features of net.0.weight.
        # Count Linear layers: 1 (in->hid) + hid_layers (hid->hid) + 1 (hid->out).
        _sd          = torch.load(SIMPLE_IK_PATH, map_location=device)
        _hid_dim     = _sd['net.0.weight'].shape[0]
        _n_linears   = sum(1 for k in _sd if k.endswith('.weight'))
        _hid_layers  = _n_linears - 2   # subtract input and output Linear layers
 
        simple_ik = simpleIK(robot=robot, device= device, 
                             hid_dim=_hid_dim, hid_layers=_hid_layers)
        simple_ik.load_state_dict(_sd)
        simple_ik.eval()
        print(f'Loaded simple IK  ← {SIMPLE_IK_PATH}  '
              f'(hid_dim={_hid_dim}, hid_layers={_hid_layers})')
 
    # ── imitation IK ─────────────────────────────────────────────────────────
    imit_ik = None
    if USE_IMIT_IK:
        from ik_nn import IK
        _imit_sd         = torch.load(IMIT_IK_PATH, map_location=device)
        _imit_hid_dim    = _imit_sd['net.0.weight'].shape[0]
        _imit_n_linears  = sum(1 for k in _imit_sd if k.endswith('.weight'))
        _imit_hid_layers = _imit_n_linears - 2
        imit_ik = IK(robot=robot,
                     hid_dim=_imit_hid_dim, hid_layers=_imit_hid_layers)
        imit_ik.load_state_dict(_imit_sd)
        imit_ik.eval()
        print(f'Loaded imitation IK ← {IMIT_IK_PATH}  '
              f'(hid_dim={_imit_hid_dim}, hid_layers={_imit_hid_layers})')
 
    # ── generate ─────────────────────────────────────────────────────────────
    gen = generate_test_data(
        robot            = robot,
        my_board         = board,
        path_planner     = path_planner,
        loss             = loss,
        iterative_solver = oracle,
        simpleIKNN       = simple_ik,
        imitationIKNN    = imit_ik,
        max_good_dist    = 0.1,
        max_good_deg     = 2.0,
        max_okay_dist    = 0.25,
        max_okay_deg     = 6.0,
        frames_per_leg   = FRAMES_PER_LEG,
        dataset_filename = OUTPUT_FILE,
    )
 
    gen.generate_n_moves(N_MOVES)
    gen.save()