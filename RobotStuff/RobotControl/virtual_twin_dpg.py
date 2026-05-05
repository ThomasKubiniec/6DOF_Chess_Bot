"""
Virtual Twin — DearPyGUI control panel + matplotlib robot visualizer.

Architecture (inverted from previous version):
  - main() owns the main thread and drives the matplotlib render loop.
  - DPG runs in a background thread (dpg_thread).
  - A worker thread handles solver / path-planner button actions.
  - SharedState is the thread-safe bridge between all three threads.

Thread ownership:
  main thread  → viz.render() on a tight loop (~20 Hz)
  dpg_thread   → dpg.create_context() … dpg.start_dearpygui() (blocking)
  worker_thread → polls SharedState, runs IK / path planning, calls viz.set_*()

DPG slider write-back after IK:
  Worker queues the solved q into _pending_slider_write (lock-guarded).
  dpg_thread reads it each frame via a dpg.set_frame_callback and calls
  dpg.set_value() — the only thread that is allowed to touch DPG widgets.
"""

import threading
import time

import dearpygui.dearpygui as dpg
import torch
import numpy as np

from forward_kinematics import Robot_math
from path_planning_math import PathPlannerMath
from expert_inv_kin import Oracle
from robot_visualizer import RobotVisualizer, _to_tensor
from rot_math import YPR_SO3, to_SO3


# ─────────────────────────────────────────────────────────────────────────────
# Shared State
# ─────────────────────────────────────────────────────────────────────────────

class SharedState:
    def __init__(self, initial: dict = None):
        self._data = dict(initial or {})
        self._lock = threading.Lock()

    def set(self, key, value):
        with self._lock:
            self._data[key] = value

    def get(self, key, default=None):
        with self._lock:
            return self._data.get(key, default)

    def snapshot(self):
        with self._lock:
            return dict(self._data)


# ─────────────────────────────────────────────────────────────────────────────
# Robot / solver initialisation (module level so all threads share them)
# ─────────────────────────────────────────────────────────────────────────────

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

solver       = Oracle(robot_class=robot)
path_planner = PathPlannerMath(my_robot=robot)

# viz is created in main() after matplotlib is initialised on the main thread


# ─────────────────────────────────────────────────────────────────────────────
# Pending slider write-back
# (worker sets it; dpg_thread flushes it each frame via frame callback)
# ─────────────────────────────────────────────────────────────────────────────

_pending_slider_write: torch.Tensor | None = None
_pending_lock = threading.Lock()


def _queue_slider_write(q: torch.Tensor):
    global _pending_slider_write
    with _pending_lock:
        _pending_slider_write = q.clone()


# ─────────────────────────────────────────────────────────────────────────────
# LiveUI
# ─────────────────────────────────────────────────────────────────────────────

class LiveUI:
    def __init__(self, state: SharedState, config: list):
        self.state  = state
        self.config = config
        self.joint_tags: dict[str, str] = {}   # "j1" → dpg tag

    def build(self):
        with dpg.window(label="Control Panel", width=420, height=700, no_close=True):
            for group in self.config:
                self._create_group(group)

    def _create_group(self, group):
        with dpg.collapsing_header(label=group["group"], default_open=True):
            for item in group["items"]:
                self._create_widget(item)

    def _create_widget(self, item):
        name    = item["name"]
        default = item.get("default", 0)
        self.state.set(name, default)

        tag = f"_tag_{name}"

        def cb(sender, app_data, user_data=name):
            self.state.set(user_data, app_data)

        wtype = item["type"]

        if wtype == "float":
            dpg.add_slider_float(
                label=name, tag=tag,
                default_value=default,
                min_value=item.get("min", 0.0),
                max_value=item.get("max", 1.0),
                callback=cb, user_data=name,
            )
        elif wtype == "int":
            dpg.add_slider_int(
                label=name, tag=tag,
                default_value=default,
                min_value=item.get("min", 0),
                max_value=item.get("max", 100),
                callback=cb, user_data=name,
            )
        elif wtype == "bool":
            dpg.add_checkbox(
                label=name, tag=tag,
                default_value=default,
                callback=cb, user_data=name,
            )
        elif wtype == "button":
            def btn_cb(sender, app_data, user_data=name):
                self.state.set(user_data, True)
            dpg.add_button(label=name, tag=tag, callback=btn_cb)

        # remember joint slider tags for write-back
        if name.startswith("j") and name[1:].isdigit() and wtype == "float":
            self.joint_tags[name] = tag

    def flush_slider_writeback(self):
        """
        Called each DPG frame (from the dpg_thread) to write IK-solved joint
        angles back into the joint sliders.  dpg.set_value must only be called
        from the DPG thread.
        """
        global _pending_slider_write
        with _pending_lock:
            q = _pending_slider_write
            if q is None:
                return
            _pending_slider_write = None

        for i, jname in enumerate(sorted(self.joint_tags)):   # j1, j2, …
            if i < len(q):
                dpg.set_value(self.joint_tags[jname], float(q[i]))
                self.state.set(jname, float(q[i]))


# ─────────────────────────────────────────────────────────────────────────────
# DPG config
# ─────────────────────────────────────────────────────────────────────────────

def _make_config():
    return [
        {
            "group": "Loss Weights",
            "items": [
                {"name": "position_w",      "type": "float", "default": 1.0,  "min": 0, "max": 100},
                {"name": "orientation_w",   "type": "float", "default": 0.5,  "min": 0, "max": 10},
                {"name": "closeset_dist_w", "type": "float", "default": 0.5,  "min": 0, "max": 10},
                {"name": "collision_w",     "type": "float", "default": 2.0,  "min": 0, "max": 20},
                {"name": "joint_vel_w1",    "type": "float", "default": 0.1,  "min": 0, "max": 10},
                {"name": "joint_vel_w2",    "type": "float", "default": 0.1,  "min": 0, "max": 10},
                {"name": "joint_vel_w3",    "type": "float", "default": 0.1,  "min": 0, "max": 10},
                {"name": "joint_vel_w4",    "type": "float", "default": 0.1,  "min": 0, "max": 10},
                {"name": "joint_vel_w5",    "type": "float", "default": 0.1,  "min": 0, "max": 10},
                {"name": "joint_vel_w6",    "type": "float", "default": 0.1,  "min": 0, "max": 10},
                {"name": "joint_acc_w1",    "type": "float", "default": 0.1,  "min": 0, "max": 10},
                {"name": "joint_acc_w2",    "type": "float", "default": 0.1,  "min": 0, "max": 10},
                {"name": "joint_acc_w3",    "type": "float", "default": 0.1,  "min": 0, "max": 10},
                {"name": "joint_acc_w4",    "type": "float", "default": 0.1,  "min": 0, "max": 10},
                {"name": "joint_acc_w5",    "type": "float", "default": 0.1,  "min": 0, "max": 10},
                {"name": "joint_acc_w6",    "type": "float", "default": 0.1,  "min": 0, "max": 10},
            ],
        },
        {
            "group": "Pass Criteria",
            "items": [
                {"name": "max_good_dist",    "type": "float", "default": 0.25, "min": 0, "max": 1},
                {"name": "max_good_ori_deg", "type": "float", "default": 5.0,  "min": 0, "max": 15},
                {"name": "max_okay_dist",    "type": "float", "default": 0.5,  "min": 0, "max": 1.5},
                {"name": "max_okay_ori_deg", "type": "float", "default": 15.0, "min": 0, "max": 30},
            ],
        },
        {
            "group": "IK Controls",
            "items": [
                {"name": "IK_target_X",     "type": "float", "default": 0,
                 "min": -float(robot.max_reach), "max": float(robot.max_reach)},
                {"name": "IK_target_Y",     "type": "float", "default": 0,
                 "min": -float(robot.max_reach), "max": float(robot.max_reach)},
                {"name": "IK_target_Z",     "type": "float", "default": 0,
                 "min": -float(robot.max_reach), "max": float(robot.max_reach)},
                {"name": "IK_target_ROLL",  "type": "float", "default": 0, "min": -90, "max": 90},
                {"name": "IK_target_PITCH", "type": "float", "default": 0, "min": -90, "max": 90},
                {"name": "IK_target_YAW",   "type": "float", "default": 0, "min": -90, "max": 90},
                {"name": "solve_IK",        "type": "button"},
            ],
        },
        {
            "group": "Path Planning",
            "items": [
                {"name": "randomize_Start_path_planning", "type": "button"},
                {"name": "randomize_End_path_planning",   "type": "button"},
                {"name": "solve_path_planning",           "type": "button"},
                {"name": "Play_Trajectory",               "type": "button"},
            ],
        },
        {
            "group": "Robot Joints (FK)",
            "items": [
                {"name": "j1", "type": "float", "default": 0.0,
                 "min": float(robot.low_bounds[0]), "max": float(robot.high_bounds[0])},
                {"name": "j2", "type": "float", "default": 0.0,
                 "min": float(robot.low_bounds[1]), "max": float(robot.high_bounds[1])},
                {"name": "j3", "type": "float", "default": 0.0,
                 "min": float(robot.low_bounds[2]), "max": float(robot.high_bounds[2])},
                {"name": "j4", "type": "float", "default": 0.0,
                 "min": float(robot.low_bounds[3]), "max": float(robot.high_bounds[3])},
                {"name": "j5", "type": "float", "default": 0.0,
                 "min": float(robot.low_bounds[4]), "max": float(robot.high_bounds[4])},
                {"name": "j6", "type": "float", "default": 0.0,
                 "min": float(robot.low_bounds[5]), "max": float(robot.high_bounds[5])},
            ],
        },
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Slider → solver/robot application (called by worker each tick)
# ─────────────────────────────────────────────────────────────────────────────

_LOSS_SETTERS = {
    "position_w":      lambda v: setattr(solver.L, "pos_w",   v),
    "orientation_w":   lambda v: setattr(solver.L, "rot_w",   v),
    "closeset_dist_w": lambda v: setattr(solver.L, "dist_w",  v),
    "collision_w":     lambda v: setattr(solver.L, "crash_w", v),
    "joint_vel_w1":    lambda v: solver.L.vel_lambda.__setitem__((0, 0), v),
    "joint_vel_w2":    lambda v: solver.L.vel_lambda.__setitem__((1, 1), v),
    "joint_vel_w3":    lambda v: solver.L.vel_lambda.__setitem__((2, 2), v),
    "joint_vel_w4":    lambda v: solver.L.vel_lambda.__setitem__((3, 3), v),
    "joint_vel_w5":    lambda v: solver.L.vel_lambda.__setitem__((4, 4), v),
    "joint_vel_w6":    lambda v: solver.L.vel_lambda.__setitem__((5, 5), v),
    "joint_acc_w1":    lambda v: solver.L.acc_lambda.__setitem__((0, 0), v),
    "joint_acc_w2":    lambda v: solver.L.acc_lambda.__setitem__((1, 1), v),
    "joint_acc_w3":    lambda v: solver.L.acc_lambda.__setitem__((2, 2), v),
    "joint_acc_w4":    lambda v: solver.L.acc_lambda.__setitem__((3, 3), v),
    "joint_acc_w5":    lambda v: solver.L.acc_lambda.__setitem__((4, 4), v),
    "joint_acc_w6":    lambda v: solver.L.acc_lambda.__setitem__((5, 5), v),
}

_PASS_ATTRS = {
    "max_good_dist":    "max_good_err_pos",
    "max_good_ori_deg": "max_good_err_deg",
    "max_okay_dist":    "max_ok_err_pos",
    "max_okay_ori_deg": "max_ok_err_deg",
}


def _apply_slider_state(snap: dict, viz: RobotVisualizer):
    for key, setter in _LOSS_SETTERS.items():
        if key in snap:
            setter(snap[key])

    for key, attr in _PASS_ATTRS.items():
        if key in snap:
            setattr(solver.L, attr, snap[key])

    solver.current_XYZ_targ[0] = snap.get("IK_target_X",     0.0)
    solver.current_XYZ_targ[1] = snap.get("IK_target_Y",     0.0)
    solver.current_XYZ_targ[2] = snap.get("IK_target_Z",     0.0)
    solver.current_YPR_targ[0] = snap.get("IK_target_YAW",   0.0)
    solver.current_YPR_targ[1] = snap.get("IK_target_PITCH", 0.0)
    solver.current_YPR_targ[2] = snap.get("IK_target_ROLL",  0.0)

    # FK joint sliders drive robot pose when no button action is running
    q_new = torch.tensor([
        snap.get("j1", 0.0), snap.get("j2", 0.0), snap.get("j3", 0.0),
        snap.get("j4", 0.0), snap.get("j5", 0.0), snap.get("j6", 0.0),
    ], dtype=torch.float64)
    robot.q_vect = q_new
    viz.set_current_q(q_new)


# ─────────────────────────────────────────────────────────────────────────────
# Button actions (run in worker thread)
# ─────────────────────────────────────────────────────────────────────────────

def _ik_quality() -> str:
    w = solver.L.target_weight
    if w >= 1.0:   return "good"
    if w >= 0.3:   return "okay"
    return "bad"


def _action_solve_ik(state: SharedState, viz: RobotVisualizer):
    pre_q = robot.q_vect.clone()
    solver.get_IK_given_targ()
    post_q = robot.q_vect.clone()

    ypr    = solver.current_YPR_targ
    R_goal = YPR_SO3(yaw_deg=float(ypr[0]),
                     pitch_deg=float(ypr[1]),
                     roll_deg=float(ypr[2])).numpy()
    pos_goal = solver.current_XYZ_targ.detach().numpy()

    viz.clear_scene()
    viz.set_target_markers([{"pos": pos_goal, "R": R_goal, "quality": _ik_quality()}])
    viz.set_current_q(pre_q)
    viz.set_target_q(post_q)
    _queue_slider_write(post_q)
    state.set("solve_IK", False)


def _action_randomize_start(state: SharedState):
    path_planner.make_random_q_start()
    state.set("randomize_Start_path_planning", False)


def _action_randomize_end(state: SharedState):
    path_planner.make_random_q_end()
    state.set("randomize_End_path_planning", False)


def _action_solve_path_planning(state: SharedState, viz: RobotVisualizer):
    traj_tuple = path_planner.MoveL_mutable_start_stop()
    solver.follow_trajectory(traj_tuple)

    traj_results = solver.current_trajectory   # list of OptimizeResult
    workspace    = traj_tuple[0]               # (frames, 9) tensor

    ghost_qs = []
    markers  = []
    q_running = _to_tensor(traj_tuple[2]) if traj_tuple[2] is not None else robot.q_vect.clone()

    for i, res in enumerate(traj_results):
        delta_q = _to_tensor(res.x)
        q_abs   = q_running + delta_q
        ghost_qs.append(q_abs.clone())
        q_running = q_abs   # carry forward for next step

        pos_6d  = workspace[i]
        pos     = pos_6d[:3].detach().numpy()
        R_mat   = to_SO3(pos_6d[3:]).detach().numpy()
        markers.append({"pos": pos, "R": R_mat, "quality": _ik_quality()})

    viz.clear_scene()
    viz.set_ghost_trail(ghost_qs)
    viz.set_target_markers(markers)
    if ghost_qs:
        viz.set_current_q(ghost_qs[-1])
        _queue_slider_write(ghost_qs[-1])

    state.set("solve_path_planning", False)


def _action_play_trajectory(state: SharedState, viz: RobotVisualizer):
    traj = solver.current_trajectory
    if not traj:
        state.set("Play_Trajectory", False)
        return

    viz.clear_scene()
    ghost_qs  = []
    q_running = robot.q_vect.clone()
    delay     = max(solver.current_time_delay, 0.04)

    for res in traj:
        delta_q  = _to_tensor(res.x)
        q_abs    = q_running + delta_q
        q_running = q_abs
        ghost_qs.append(q_abs.clone())

        viz.set_ghost_trail(list(ghost_qs))
        viz.set_current_q(q_abs)
        time.sleep(delay)

    state.set("Play_Trajectory", False)


# ─────────────────────────────────────────────────────────────────────────────
# Worker loop (background thread)
# ─────────────────────────────────────────────────────────────────────────────

_WORKER_HZ = 20

def worker_loop(state: SharedState, viz: RobotVisualizer):
    while True:
        snap = state.snapshot()
        _apply_slider_state(snap, viz)

        if snap.get("solve_IK"):
            _action_solve_ik(state, viz)
        if snap.get("randomize_Start_path_planning"):
            _action_randomize_start(state)
        if snap.get("randomize_End_path_planning"):
            _action_randomize_end(state)
        if snap.get("solve_path_planning"):
            _action_solve_path_planning(state, viz)
        if snap.get("Play_Trajectory"):
            _action_play_trajectory(state, viz)

        time.sleep(1.0 / _WORKER_HZ)


# ─────────────────────────────────────────────────────────────────────────────
# DPG thread
# ─────────────────────────────────────────────────────────────────────────────

def dpg_thread_fn(state: SharedState, ui_ready: threading.Event):
    dpg.create_context()

    config = _make_config()
    ui = LiveUI(state, config)
    ui.build()

    dpg.create_viewport(title="IK Tuning UI", width=440, height=720)
    dpg.setup_dearpygui()
    dpg.show_viewport()

    ui_ready.set()   # signal main thread that DPG is up

    # Manual render loop so we can flush slider write-backs each frame
    while dpg.is_dearpygui_running():
        ui.flush_slider_writeback()
        dpg.render_dearpygui_frame()

    dpg.destroy_context()


# ─────────────────────────────────────────────────────────────────────────────
# Main — owns the main thread, drives matplotlib
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # matplotlib must be initialised on the main thread
    viz   = RobotVisualizer(robot=robot)
    state = SharedState()

    # Start DPG in a background thread; wait until its window is ready
    ui_ready = threading.Event()
    dpg_t = threading.Thread(
        target=dpg_thread_fn, args=(state, ui_ready), daemon=True)
    dpg_t.start()
    ui_ready.wait()

    # Start worker
    worker_t = threading.Thread(
        target=worker_loop, args=(state, viz), daemon=True)
    worker_t.start()

    # Main thread: pump matplotlib at ~20 Hz until DPG window closes
    while dpg_t.is_alive():
        viz.render(pause_s=0.05)

    # DPG has exited — one last render then done
    viz.render(pause_s=0.1)


if __name__ == "__main__":
    main()