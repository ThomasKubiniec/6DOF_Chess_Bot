"""
Robot Visualizer — matplotlib-based, runs in a dedicated background thread.

Thread safety strategy:
  - All scene data is written through RobotVisualizer.set_*() methods which
    acquire a threading.Lock before touching any shared list/tensor.
  - The background thread only reads that data (also under the lock) and
    calls matplotlib.  Matplotlib itself is NOT thread-safe, so the render
    thread is the ONLY thread that ever touches the figure/axes.
  - DPG callbacks call set_*() to push new data; they never touch matplotlib.
"""

import threading
import time
import math

import matplotlib
matplotlib.use("TkAgg")          # use a backend that supports a separate window
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401  (needed for 3D projection)

import torch

from forward_kinematics import Robot_math
from expert_inv_kin import Oracle


# ── colour constants ─────────────────────────────────────────────────────────
_C_CURRENT   = "orange"       # robot at current / pre-IK pose
_C_TARGET    = "cyan"         # robot at IK solution pose
_C_GHOST_B   = (0.4, 0.8, 1.0)  # ghost trail base colour (light blue)
_C_JOINT     = "white"
_C_GOOD      = "limegreen"
_C_OKAY      = "orange"
_C_BAD       = "red"
_C_AXES      = ["red", "green", "blue"]   # X Y Z quiver colours

_QUIVER_LEN  = 0.4            # fraction of max_reach for frame quivers
_GHOST_ALPHA_START = 0.55     # alpha of the first (oldest) ghost
_GHOST_DISCOUNT    = 0.82     # multiply alpha by this each step (newest = brightest)
_REFRESH_HZ  = 20             # redraws per second


class RobotVisualizer:
    """
    Owns a matplotlib figure that lives in a background thread.
    Public set_*() methods let DPG callbacks push scene data safely.
    """

    def __init__(self, robot: Robot_math, oracle: Oracle):
        self.robot  = robot
        self.oracle = oracle

        self._lock = threading.Lock()

        # ── scene data (written by DPG thread, read by render thread) ─────────
        self._current_q  = robot.q_vect.clone()   # orange robot
        self._target_q   = None                    # cyan robot (IK result)
        self._ghost_qs   = []                      # list of q tensors, oldest first
        self._targets    = []                      # list of dicts: {pos, R, quality}
        #   quality: "good" | "okay" | "bad"

        self._dirty = True    # flag: scene changed, needs redraw

        # ── start render thread ───────────────────────────────────────────────
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._render_loop, daemon=True)
        self._thread.start()

    # ── public setters (called from DPG / solver callbacks) ──────────────────

    def set_current_q(self, q):
        """Update the orange 'current pose' robot."""
        with self._lock:
            self._current_q = _to_tensor(q)
            self._dirty = True

    def set_target_q(self, q):
        """Set the cyan 'IK target pose' robot.  Pass None to hide it."""
        with self._lock:
            self._target_q = _to_tensor(q) if q is not None else None
            self._dirty = True

    def set_ghost_trail(self, q_list):
        """
        Replace the ghost trail with a new list of joint-angle tensors.
        Oldest pose first — alpha decays oldest→newest so newest is brightest.
        Pass an empty list to clear.
        """
        with self._lock:
            self._ghost_qs = [_to_tensor(q) for q in q_list]
            self._dirty = True

    def set_target_markers(self, targets):
        """
        Replace all target-point markers.
        targets: list of dicts  { 'pos': tensor(3,), 'R': tensor(3,3), 'quality': str }
        quality is 'good', 'okay', or 'bad'.
        Pass [] to clear.
        """
        with self._lock:
            self._targets = list(targets)
            self._dirty = True

    def clear_scene(self):
        """Wipe everything except the current-pose robot."""
        with self._lock:
            self._target_q  = None
            self._ghost_qs  = []
            self._targets   = []
            self._dirty = True

    def stop(self):
        self._stop_event.set()

    # ── internal render helpers ───────────────────────────────────────────────

    def _snapshot(self):
        """Take a consistent copy of scene data under the lock."""
        with self._lock:
            return dict(
                current_q  = self._current_q.clone(),
                target_q   = self._target_q.clone() if self._target_q is not None else None,
                ghost_qs   = [q.clone() for q in self._ghost_qs],
                targets    = list(self._targets),   # shallow copy — dicts are read-only
                dirty      = self._dirty,
            )

    def _mark_clean(self):
        with self._lock:
            self._dirty = False

    def _get_robot_geometry(self, q):
        """Run FK for joint vector q, return (ds, Rs) without mutating robot permanently."""
        saved_q = self.robot.q_vect.clone()
        self.robot.q_vect = q
        ds = [d.detach().cpu().numpy() for d in self.robot.give_ds()]
        Rs = [R.detach().cpu().numpy() for R in self.robot.give_Rs()]
        self.robot.q_vect = saved_q
        return ds, Rs

    def _draw_robot_on_ax(self, ax, q, line_color, joint_color, alpha):
        """Draw one robot pose onto ax."""
        ds, Rs = self._get_robot_geometry(q)
        xs = [d[0] for d in ds]
        ys = [d[1] for d in ds]
        zs = [d[2] for d in ds]

        ax.plot(xs, ys, zs, "-", color=line_color, alpha=alpha, linewidth=1.8)
        ax.scatter(xs, ys, zs, color=joint_color, s=14, alpha=alpha, zorder=4)

        ql = float(self.robot.max_reach) * _QUIVER_LEN
        for pos, rot in zip(ds, Rs):
            for ci, c in enumerate(_C_AXES):
                ax.quiver(pos[0], pos[1], pos[2],
                          rot[0, ci], rot[1, ci], rot[2, ci],
                          length=ql, color=c, alpha=alpha * 0.55,
                          normalize=True)

    def _draw_target_marker(self, ax, tgt):
        """Draw a dot + orientation quivers for one target point."""
        pos = tgt["pos"]
        R   = tgt["R"]
        q_str = tgt.get("quality", "bad")
        color = {"good": _C_GOOD, "okay": _C_OKAY, "bad": _C_BAD}.get(q_str, _C_BAD)

        ax.scatter([pos[0]], [pos[1]], [pos[2]],
                   color=color, s=60, zorder=5, depthshade=False)

        ql = float(self.robot.max_reach) * _QUIVER_LEN * 0.8
        for ci, c in enumerate(_C_AXES):
            ax.quiver(pos[0], pos[1], pos[2],
                      R[0, ci], R[1, ci], R[2, ci],
                      length=ql, color=c, alpha=0.9, normalize=True)

    # ── render loop (background thread) ──────────────────────────────────────

    def _render_loop(self):
        plt.style.use("dark_background")
        fig = plt.figure(figsize=(7, 6))
        ax  = fig.add_subplot(111, projection="3d")
        fig.canvas.manager.set_window_title("Robot Virtual Twin")

        mr = float(self.robot.max_reach)

        while not self._stop_event.is_set():
            snap = self._snapshot()

            if snap["dirty"]:
                ax.cla()

                # axis limits and labels
                ax.set_xlim(-mr, mr)
                ax.set_ylim(-mr, mr)
                ax.set_zlim(-mr, mr)
                ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
                ax.set_title("Robot Virtual Twin", color="white")

                # ── ghost trail (oldest = most transparent) ───────────────────
                ghosts = snap["ghost_qs"]
                n = len(ghosts)
                if n > 0:
                    # build alphas: start bright at newest (last), fade toward oldest
                    # newest ghost is index n-1, oldest is index 0
                    alphas = []
                    a = _GHOST_ALPHA_START
                    for _ in range(n):
                        alphas.append(a)
                        a *= _GHOST_DISCOUNT
                    # alphas[0] = oldest (most transparent after decay from newest)
                    # reverse so index matches ghost list (oldest first)
                    alphas = list(reversed(alphas))

                    for q_ghost, alpha in zip(ghosts, alphas):
                        self._draw_robot_on_ax(ax, q_ghost,
                                               line_color=_C_GHOST_B,
                                               joint_color=_C_GHOST_B,
                                               alpha=alpha)

                # ── target markers ────────────────────────────────────────────
                for tgt in snap["targets"]:
                    self._draw_target_marker(ax, tgt)

                # ── cyan: IK target pose ──────────────────────────────────────
                if snap["target_q"] is not None:
                    self._draw_robot_on_ax(ax, snap["target_q"],
                                           line_color=_C_TARGET,
                                           joint_color=_C_TARGET,
                                           alpha=0.85)

                # ── orange: current / FK pose (always on top) ─────────────────
                self._draw_robot_on_ax(ax, snap["current_q"],
                                       line_color=_C_CURRENT,
                                       joint_color=_C_CURRENT,
                                       alpha=0.95)

                self._mark_clean()
                fig.canvas.draw()

            fig.canvas.flush_events()
            time.sleep(1.0 / _REFRESH_HZ)

        plt.close(fig)


# ── module-level helper ───────────────────────────────────────────────────────

def _to_tensor(x) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.detach().clone().to(torch.float64)
    return torch.tensor(x, dtype=torch.float64)