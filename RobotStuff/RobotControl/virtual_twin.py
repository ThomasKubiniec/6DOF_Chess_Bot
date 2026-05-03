"""
Virtual Twin Live — Interactive Loss Function Tuner + Trajectory Planner

Modes (radio button):
  FK   : Joint sliders drive the robot live.
  IK   : XYZ/RPY sliders set the goal; "Solve IK" runs the Oracle.
  Path : Define start/end poses (or randomise), set frames & speed,
         "Plan Trajectory" solves IK for every waypoint in a background
         thread, then renders the full ghost trail at once. "Play" steps
         through frames with the timer so the UI stays responsive.

Three robots always visible:
  Magenta (q_prev) — previous committed pose  (acc-loss context)
  Cyan    (q_curr) — current pose
  Orange  (q_new)  — IK solution / trajectory frame being previewed

Goal point colour:
  Lime  — IK solution passes the position + orientation thresholds
  Red   — IK solution fails

Trajectory ghost trail: all frames drawn simultaneously at decaying
opacity (oldest = most transparent).  Each frame's goal dot is
green/red according to its own pass/fail result.

Slider responsiveness fix: slider callbacks only set a dirty flag;
the 80 ms main-thread timer does all actual rendering.  The IK/path
worker threads never touch the canvas.
"""

import threading
import time
import numpy as np
import torch
import matplotlib

def _set_interactive_backend():
    for backend in ("TkAgg", "Qt5Agg", "Qt6Agg", "WxAgg"):
        try:
            matplotlib.use(backend)
            import matplotlib.pyplot as _p; _p.figure(); _p.close()
            return backend
        except Exception:
            continue
    return matplotlib.get_backend()

_active_backend = _set_interactive_backend()

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider, Button, TextBox, RadioButtons
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from copy import deepcopy

from forward_kinematics import Robot_math
from loss_math import Loss_Math
from expert_inv_kin import Oracle
from path_planning_math import PathPlannerMath
from rot_math import YPR_SO3, to_6D_R, to_SO3


# ─────────────────────────────────────────────────────────────────────────────
# Tiny helpers
# ─────────────────────────────────────────────────────────────────────────────

def _t(x) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(torch.float64)
    return torch.tensor(x, dtype=torch.float64)

def _np(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def _rpy_to_6d(rpy_deg: torch.Tensor) -> torch.Tensor:
    R = YPR_SO3(yaw_deg=float(rpy_deg[2]),
                pitch_deg=float(rpy_deg[1]),
                roll_deg=float(rpy_deg[0]))
    return to_6D_R(R)

def _rpy_to_SO3(rpy_deg: torch.Tensor) -> torch.Tensor:
    return YPR_SO3(yaw_deg=float(rpy_deg[2]),
                   pitch_deg=float(rpy_deg[1]),
                   roll_deg=float(rpy_deg[0]))


# ─────────────────────────────────────────────────────────────────────────────
# Main class
# ─────────────────────────────────────────────────────────────────────────────

class VirtualTwinLive:
    """
    Parameters
    ----------
    robot      : Robot_math instance  (deep-copied for ghost renders)
    pos_w / rot_w / crash_w / dist_w  : initial scalar loss weights
    vel_w / acc_w : uniform per-joint lambda initialisers
    quiver_len : orientation-arrow length in the 3-D viewport
    unit_label : axis tick label string  (e.g. 'inches')
    """

    def __init__(self,
                 robot: Robot_math,
                 pos_w:   float = 1.0,
                 rot_w:   float = 0.5,
                 vel_w:   float = 0.1,
                 acc_w:   float = 0.1,
                 crash_w: float = 2.0,
                 dist_w:  float = 0.1,
                 quiver_len: float = 1.0,
                 unit_label: str  = "units"):

        self.robot      = robot
        self.n          = len(robot.a)
        self.quiver_len = quiver_len
        self.unit_label = unit_label

        # Three independent robot copies for rendering (no shared q_vect state)
        self._rob_prev = deepcopy(robot)
        self._rob_curr = deepcopy(robot)
        self._rob_new  = deepcopy(robot)

        # ── pose state ────────────────────────────────────────────────────────
        self.q_prev = torch.zeros(self.n, dtype=torch.float64)  # magenta
        self.q_curr = torch.zeros(self.n, dtype=torch.float64)  # cyan
        self.q_new  = None                                       # orange

        self.delta_q_prev = torch.zeros(self.n, dtype=torch.float64)

        # ── mode & goal ───────────────────────────────────────────────────────
        self.mode     = "FK"
        self.goal_pos = torch.zeros(3, dtype=torch.float64)
        self.goal_rpy = torch.zeros(3, dtype=torch.float64)   # roll, pitch, yaw deg

        # Pass/fail for the current single IK solve
        self._ik_passed: bool | None = None

        # ── loss weights ──────────────────────────────────────────────────────
        self.pos_w       = pos_w
        self.rot_w       = rot_w
        self.crash_w     = crash_w
        self.dist_w      = dist_w
        self.vel_lambdas = [vel_w] * self.n
        self.acc_lambdas = [acc_w] * self.n

        # pass/fail distance and angle thresholds (physical units / degrees)
        self.thresh_dist_val = 0.5   # physical units
        self.thresh_deg_val  = 5.0   # degrees

        # ── Oracle & planner ──────────────────────────────────────────────────
        self.oracle  = self._make_oracle()
        self.planner = PathPlannerMath(my_robot=robot)

        # ── trajectory state ──────────────────────────────────────────────────
        # Stored after "Plan Trajectory":
        #   _traj_qs    : list of q tensors, one per waypoint
        #   _traj_goals : list of (pos, rpy_deg) tuples
        #   _traj_pass  : list of bool
        self._traj_qs:    list = []
        self._traj_goals: list = []
        self._traj_pass:  list = []
        self._traj_frame: int  = 0       # frame index during playback
        self._traj_playing: bool = False

        # Path-mode endpoint state
        self._path_start_pos = torch.zeros(3, dtype=torch.float64)
        self._path_start_rpy = torch.zeros(3, dtype=torch.float64)
        self._path_end_pos   = torch.zeros(3, dtype=torch.float64)
        self._path_end_rpy   = torch.zeros(3, dtype=torch.float64)

        # ── threading ─────────────────────────────────────────────────────────
        self._worker_running = False
        self._redraw_flag    = False   # worker sets; timer reads on main thread

        # ── build UI ──────────────────────────────────────────────────────────
        self._build_figure(float(robot.max_reach))
        self._draw()

        # 80 ms heartbeat: flushes redraws and advances trajectory playback
        self._timer = self.fig.canvas.new_timer(interval=80)
        self._timer.add_callback(self._tick)
        self._timer.start()

        plt.show()

    # ─────────────────────────────────────────────────────────────────────────
    # Oracle / planner factories
    # ─────────────────────────────────────────────────────────────────────────

    def _make_oracle(self) -> Oracle:
        return Oracle(robot_class=self.robot,
                      pos_w=self.pos_w, rot_w=self.rot_w,
                      crash_w=self.crash_w, dist_w=self.dist_w,
                      vel_lambda=self.vel_lambdas,
                      acc_lambda=self.acc_lambdas)

    # ─────────────────────────────────────────────────────────────────────────
    # Main-thread heartbeat
    # ─────────────────────────────────────────────────────────────────────────

    def _tick(self):
        """Called every 80 ms on the main thread."""
        # Flush any redraw requested by a worker thread
        if self._redraw_flag:
            self._redraw_flag = False
            self._draw()
            self.fig.canvas.draw()

        # Advance trajectory playback one frame per tick
        if self._traj_playing and self._traj_qs:
            self._traj_frame = (self._traj_frame + 1) % len(self._traj_qs)
            self.q_new = self._traj_qs[self._traj_frame]
            self._draw()
            self.fig.canvas.draw()

    # ─────────────────────────────────────────────────────────────────────────
    # Figure construction
    # ─────────────────────────────────────────────────────────────────────────

    def _build_figure(self, max_reach: float):
        self.fig = plt.figure(figsize=(20, 11))
        self.fig.patch.set_facecolor("#1a1a2e")

        # Left: 3-D viewport   Right: controls
        gs = gridspec.GridSpec(1, 2, width_ratios=[1.3, 1],
                               left=0.01, right=0.99,
                               bottom=0.02, top=0.98, wspace=0.04)

        self.ax3d = self.fig.add_subplot(gs[0, 0], projection="3d")
        self._style_3d(max_reach)

        # Right panel: 9 rows
        gsr = gridspec.GridSpecFromSubplotSpec(
            9, 1, subplot_spec=gs[0, 1], hspace=0.5,
            height_ratios=[0.7,   # [0] loss weights
                           0.35,  # [1] pass thresholds
                           0.4,   # [2] mode radio
                           2.2,   # [3] joint/goal/path sliders
                           1.5,   # [4] lambda sliders
                           0.38,  # [5] path extra controls (frames, time)
                           0.38,  # [6] action buttons row 1
                           0.38,  # [7] action buttons row 2
                           1.0],  # [8] loss readout + status
        )

        def _panel(row, color="#16213e", title=None):
            ax = self.fig.add_subplot(gsr[row])
            ax.set_facecolor(color); ax.set_xticks([]); ax.set_yticks([])
            if title:
                ax.set_title(title, color="white", fontsize=8, pad=2)
            return ax

        self._ax_loss     = _panel(0, title="Loss Weights")
        self._ax_thresh   = _panel(1, title="Pass Thresholds")
        self._ax_mode     = _panel(2, title="Mode")
        self._ax_sliders  = _panel(3, "#0d1b2a",
                                   title="Joints (FK) / Goal XYZ·RPY (IK) / Path Endpoints (Path)")
        self._ax_lambda   = _panel(4, "#0d1b2a", title="vel_λ / acc_λ  per joint")
        self._ax_path_ex  = _panel(5, title="Path Controls")
        self._ax_btns1    = _panel(6)
        self._ax_btns2    = _panel(7)
        self._ax_readout  = _panel(8, "#0a0a1a", title="Loss / Status")

        self._readout_text = self._ax_readout.text(
            0.02, 0.92, "", ha="left", va="top",
            color="#e0e0e0", fontsize=7, family="monospace",
            transform=self._ax_readout.transAxes)

        self._joint_sliders: list = []
        self._goal_sliders:  list = []
        self._path_start_sliders: list = []
        self._path_end_sliders:   list = []

        self._widgets_created = False
        self.fig.canvas.mpl_connect("draw_event", self._on_first_draw)

    def _style_3d(self, max_reach: float):
        ax = self.ax3d
        lim = max_reach * 1.1
        ax.set_facecolor("#0f0f23")
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
        ax.set_xlabel(f"X ({self.unit_label})", color="white")
        ax.set_ylabel(f"Y ({self.unit_label})", color="white")
        ax.set_zlabel(f"Z ({self.unit_label})", color="white")
        ax.tick_params(colors="white")
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False

    # ─────────────────────────────────────────────────────────────────────────
    # Widget creation (deferred until bboxes are valid)
    # ─────────────────────────────────────────────────────────────────────────

    def _on_first_draw(self, event):
        if self._widgets_created:
            return
        self._widgets_created = True
        self._add_loss_textboxes()
        self._add_thresh_controls()
        self._add_mode_radio()
        self._add_joint_sliders()
        self._add_lambda_sliders()
        self._add_path_extra_controls()
        self._add_buttons()
        self._update_slider_visibility()
        self.fig.canvas.draw_idle()

    # ── Loss weight text-boxes ────────────────────────────────────────────────

    def _add_loss_textboxes(self):
        bb = self._ax_loss.get_position()
        labels   = ["pos_w", "rot_w", "crash_w", "dist_w"]
        initvals = [self.pos_w, self.rot_w, self.crash_w, self.dist_w]
        w, h, gap = 0.065, 0.026, 0.008
        total = len(labels) * (w + gap) - gap
        x0 = bb.x0 + (bb.width - total) / 2
        y  = bb.y0 + 0.008

        self._textboxes: dict = {}
        for i, (lbl, val) in enumerate(zip(labels, initvals)):
            ax = self.fig.add_axes([x0 + i*(w+gap), y, w, h])
            ax.set_facecolor("#2a2a4a")
            tb = TextBox(ax, lbl+"\n", initial=str(val),
                         color="#2a2a4a", hovercolor="#3a3a5a", label_pad=0.01)
            tb.label.set_color("white");        tb.label.set_fontsize(6.5)
            tb.text_disp.set_color("#00ffcc");  tb.text_disp.set_fontsize(7.5)
            self._textboxes[lbl] = tb

    # ── Pass threshold controls ───────────────────────────────────────────────

    def _add_thresh_controls(self):
        bb = self._ax_thresh.get_position()
        w, h, gap = 0.065, 0.026, 0.008
        labels   = [f"dist ({self.unit_label})", "orient (deg)"]
        initvals = [self.thresh_dist_val, self.thresh_deg_val]
        keys     = ["thresh_dist", "thresh_deg"]
        total = 2 * (w + gap) - gap
        x0 = bb.x0 + (bb.width - total) / 2
        y  = bb.y0 + 0.005

        for i, (lbl, val, key) in enumerate(zip(labels, initvals, keys)):
            ax = self.fig.add_axes([x0 + i*(w+gap), y, w, h])
            ax.set_facecolor("#2a2a3a")
            tb = TextBox(ax, lbl+"\n", initial=str(val),
                         color="#2a2a3a", hovercolor="#3a3a5a", label_pad=0.01)
            tb.label.set_color("#ffdd88");      tb.label.set_fontsize(6.5)
            tb.text_disp.set_color("#ffd166");  tb.text_disp.set_fontsize(7.5)
            self._textboxes[key] = tb

    # ── Mode radio ────────────────────────────────────────────────────────────

    def _add_mode_radio(self):
        bb = self._ax_mode.get_position()
        ax = self.fig.add_axes([bb.x0+0.01, bb.y0-0.003,
                                bb.width*0.55, bb.height+0.01])
        ax.set_facecolor("#16213e")
        self._radio = RadioButtons(ax, ("FK", "IK", "Path"),
                                   active=0, activecolor="#00ff88")
        for lbl in self._radio.labels:
            lbl.set_color("white"); lbl.set_fontsize(9)
        self._radio.on_clicked(self._on_mode_change)

    # ── Joint / Goal / Path sliders ───────────────────────────────────────────

    def _add_joint_sliders(self):
        bb = self._ax_sliders.get_position()
        rows = max(self.n, 6)   # at least 6 rows to fit IK/Path goal sliders
        h, pad = 0.016, 0.003
        total_h = rows * (h + pad)
        y0 = bb.y0 + (bb.height - total_h)/2 + total_h
        sx = bb.x0 + bb.width * 0.28
        sw = bb.width * 0.62

        mr = float(self.robot.max_reach)

        # Joint sliders (FK)
        for i in range(self.n):
            y  = y0 - (i+1)*(h+pad)
            lo, hi = self.robot.joint_bounds[i]
            ax_s = self.fig.add_axes([sx, y, sw, h])
            ax_s.set_facecolor("#0d1b2a")
            if self.robot.joint_type[i] == 'r':
                sl = Slider(ax_s, f"J{i+1}",
                            np.rad2deg(float(lo)), np.rad2deg(float(hi)),
                            valinit=0.0, valfmt="%.0f°",
                            color="#00b4d8", initcolor="none")
                sl._is_revolute = True
            else:
                sl = Slider(ax_s, f"J{i+1}", float(lo), float(hi),
                            valinit=0.0, valfmt="%.2f",
                            color="#90e0ef", initcolor="none")
                sl._is_revolute = False
            sl.label.set_color("white");     sl.label.set_fontsize(7.5)
            sl.valtext.set_color("#00ffcc"); sl.valtext.set_fontsize(6.5)
            sl.on_changed(self._on_joint_slider)
            self._joint_sliders.append(sl)

        # Goal sliders (IK) — same 6 rows
        goal_cfg = [
            ("X",     -mr,  mr, "#f4a261"),
            ("Y",     -mr,  mr, "#f4a261"),
            ("Z",     -mr,  mr, "#f4a261"),
            ("Roll",  -180, 180, "#e76f51"),
            ("Pitch", -180, 180, "#e76f51"),
            ("Yaw",   -180, 180, "#e76f51"),
        ]
        for i, (lbl, lo, hi, col) in enumerate(goal_cfg):
            y = y0 - (i+1)*(h+pad)
            ax_s = self.fig.add_axes([sx, y, sw, h])
            ax_s.set_facecolor("#0d1b2a")
            sl = Slider(ax_s, lbl, lo, hi, valinit=0.0,
                        color=col, initcolor="none")
            sl.label.set_color("white");     sl.label.set_fontsize(7.5)
            sl.valtext.set_color("#ffd166"); sl.valtext.set_fontsize(6.5)
            sl.on_changed(self._on_goal_slider)
            ax_s.set_visible(False)
            self._goal_sliders.append(sl)

        # Path start sliders — 6 rows (labelled S:X, S:Y … S:Yaw)
        path_s_cfg = [
            ("S:X",     -mr,  mr, "#80cbc4"),
            ("S:Y",     -mr,  mr, "#80cbc4"),
            ("S:Z",     -mr,  mr, "#80cbc4"),
            ("S:Roll",  -180, 180, "#4db6ac"),
            ("S:Pitch", -180, 180, "#4db6ac"),
            ("S:Yaw",   -180, 180, "#4db6ac"),
        ]
        for i, (lbl, lo, hi, col) in enumerate(path_s_cfg):
            y = y0 - (i+1)*(h+pad)
            ax_s = self.fig.add_axes([sx, y, sw, h])
            ax_s.set_facecolor("#0d1b2a")
            sl = Slider(ax_s, lbl, lo, hi, valinit=0.0,
                        color=col, initcolor="none")
            sl.label.set_color("white");     sl.label.set_fontsize(7.5)
            sl.valtext.set_color("#80cbc4"); sl.valtext.set_fontsize(6.5)
            sl.on_changed(self._on_path_start_slider)
            ax_s.set_visible(False)
            self._path_start_sliders.append(sl)

        # Path end sliders — 6 rows below start sliders
        path_e_cfg = [
            ("E:X",     -mr,  mr, "#ffb347"),
            ("E:Y",     -mr,  mr, "#ffb347"),
            ("E:Z",     -mr,  mr, "#ffb347"),
            ("E:Roll",  -180, 180, "#ff8c42"),
            ("E:Pitch", -180, 180, "#ff8c42"),
            ("E:Yaw",   -180, 180, "#ff8c42"),
        ]
        extra_rows = max(self.n, 6)
        for i, (lbl, lo, hi, col) in enumerate(path_e_cfg):
            y = y0 - (extra_rows + i + 1)*(h+pad)
            ax_s = self.fig.add_axes([sx, y, sw, h])
            ax_s.set_facecolor("#0d1b2a")
            sl = Slider(ax_s, lbl, lo, hi, valinit=0.0,
                        color=col, initcolor="none")
            sl.label.set_color("white");     sl.label.set_fontsize(7.5)
            sl.valtext.set_color("#ffb347"); sl.valtext.set_fontsize(6.5)
            sl.on_changed(self._on_path_end_slider)
            ax_s.set_visible(False)
            self._path_end_sliders.append(sl)

    # ── Lambda sliders ────────────────────────────────────────────────────────

    def _add_lambda_sliders(self):
        bb = self._ax_lambda.get_position()
        h, pad = 0.013, 0.003
        total_h = self.n * (h+pad)
        y0 = bb.y0 + (bb.height - total_h)/2 + total_h
        hw  = bb.width * 0.36
        gap = bb.width * 0.05
        vx  = bb.x0 + bb.width * 0.10
        ax_ = vx + hw + gap

        self.fig.text(vx+hw*0.5, bb.y0+bb.height+0.001,
                      "vel_λ", ha="center", color="#aaaaff", fontsize=6.5)
        self.fig.text(ax_+hw*0.5, bb.y0+bb.height+0.001,
                      "acc_λ", ha="center", color="#ffaaaa", fontsize=6.5)

        self._vel_sliders: list = []
        self._acc_sliders: list = []
        for i in range(self.n):
            y = y0 - (i+1)*(h+pad)
            av = self.fig.add_axes([vx, y, hw, h])
            sv = Slider(av, f"J{i+1}", 0.0, 2.0,
                        valinit=self.vel_lambdas[i], color="#aaaaff", initcolor="none")
            sv.label.set_color("white");     sv.label.set_fontsize(6.5)
            sv.valtext.set_color("#aaaaff"); sv.valtext.set_fontsize(6)
            sv.on_changed(lambda v, idx=i: self._on_vel_lambda(v, idx))
            self._vel_sliders.append(sv)

            aa = self.fig.add_axes([ax_, y, hw, h])
            sa = Slider(aa, f"J{i+1}", 0.0, 2.0,
                        valinit=self.acc_lambdas[i], color="#ffaaaa", initcolor="none")
            sa.label.set_color("white");     sa.label.set_fontsize(6.5)
            sa.valtext.set_color("#ffaaaa"); sa.valtext.set_fontsize(6)
            sa.on_changed(lambda v, idx=i: self._on_acc_lambda(v, idx))
            self._acc_sliders.append(sa)

    # ── Path extra controls (frames & time sliders) ───────────────────────────

    def _add_path_extra_controls(self):
        bb = self._ax_path_ex.get_position()
        w = bb.width * 0.38
        gap = bb.width * 0.06
        h = 0.022
        y = bb.y0 + (bb.height - h)/2
        x0 = bb.x0 + bb.width*0.05

        # Frames slider (int 2–100)
        af = self.fig.add_axes([x0, y, w, h])
        self._sl_frames = Slider(af, "Frames", 2, 100,
                                 valinit=20, valstep=1,
                                 color="#b5e48c", initcolor="none")
        self._sl_frames.label.set_color("white");     self._sl_frames.label.set_fontsize(7.5)
        self._sl_frames.valtext.set_color("#b5e48c"); self._sl_frames.valtext.set_fontsize(7)

        # Time-per-frame slider (0.01 – 2.0 s)
        at = self.fig.add_axes([x0 + w + gap, y, w, h])
        self._sl_spf = Slider(at, "s/frame", 0.01, 2.0,
                              valinit=0.08,
                              color="#f6c90e", initcolor="none")
        self._sl_spf.label.set_color("white");     self._sl_spf.label.set_fontsize(7.5)
        self._sl_spf.valtext.set_color("#f6c90e"); self._sl_spf.valtext.set_fontsize(7)
        self._sl_spf.on_changed(self._on_spf_change)

        # Hide until Path mode
        af.set_visible(False)
        at.set_visible(False)
        self._ax_path_ex_axes = [af, at]

    def _on_spf_change(self, val):
        ms = max(40, int(val * 1000))
        self._timer.interval = ms

    # ── Action buttons ────────────────────────────────────────────────────────

    def _add_buttons(self):
        bw, bh, gap = 0.095, 0.030, 0.010

        # Row 1  (IK / Weights / Commit)
        bb1 = self._ax_btns1.get_position()
        total = 3*bw + 2*gap
        bx = bb1.x0 + (bb1.width - total)/2
        by = bb1.y0 + (bb1.height - bh)/2

        def _btn(x, label, col, hcol, cb):
            ax = self.fig.add_axes([x, by, bw, bh])
            b  = Button(ax, label, color=col, hovercolor=hcol)
            b.label.set_color("white"); b.label.set_fontsize(8)
            b.on_clicked(cb)
            return b

        self._btn_ik     = _btn(bx,              "Solve IK",     "#264653", "#2a9d8f", self._on_solve_ik)
        self._btn_aw     = _btn(bx+bw+gap,       "Apply Weights","#3d405b", "#e07a5f", self._on_apply_weights)
        self._btn_commit = _btn(bx+2*(bw+gap),   "Commit Pose",  "#6a4c93", "#b565d9", self._on_commit_pose)

        # Row 2  (Plan / Random / Play-Stop)
        bb2 = self._ax_btns2.get_position()
        by2 = bb2.y0 + (bb2.height - bh)/2

        def _btn2(x, label, col, hcol, cb):
            ax = self.fig.add_axes([x, by2, bw, bh])
            b  = Button(ax, label, color=col, hovercolor=hcol)
            b.label.set_color("white"); b.label.set_fontsize(8)
            b.on_clicked(cb)
            return b

        self._btn_plan   = _btn2(bx,            "Plan Traj",    "#1b4332", "#40916c", self._on_plan_traj)
        self._btn_rand   = _btn2(bx+bw+gap,     "Randomise",    "#343a40", "#6c757d", self._on_randomise)
        self._btn_play   = _btn2(bx+2*(bw+gap), "Play ▶",       "#0077b6", "#023e8a", self._on_play_stop)

        # Hide path buttons until Path mode
        for b in (self._btn_plan, self._btn_rand, self._btn_play):
            b.ax.set_visible(False)
        self._path_buttons = [self._btn_plan, self._btn_rand, self._btn_play]

        # Hide IK button until IK mode
        for b in (self._btn_ik,):
            b.ax.set_visible(False)
        self._ik_buttons = [self._btn_ik]

    # ─────────────────────────────────────────────────────────────────────────
    # Slider visibility management
    # ─────────────────────────────────────────────────────────────────────────

    def _update_slider_visibility(self):
        fk   = self.mode == "FK"
        ik   = self.mode == "IK"
        path = self.mode == "Path"

        for sl in self._joint_sliders:
            sl.ax.set_visible(fk)
        for sl in self._goal_sliders:
            sl.ax.set_visible(ik)
        for sl in self._path_start_sliders + self._path_end_sliders:
            sl.ax.set_visible(path)
        for ax in self._ax_path_ex_axes:
            ax.set_visible(path)
        for b in self._path_buttons:
            b.ax.set_visible(path)
        for b in self._ik_buttons:
            b.ax.set_visible(ik)

    # ─────────────────────────────────────────────────────────────────────────
    # Mode callback
    # ─────────────────────────────────────────────────────────────────────────

    def _on_mode_change(self, label: str):
        self.mode = label
        if label != "Path":
            self._traj_playing = False
            self._btn_play.label.set_text("Play ▶")
        self._update_slider_visibility()
        self._redraw_flag = True
        self.fig.canvas.draw_idle()

    # ─────────────────────────────────────────────────────────────────────────
    # Slider callbacks  — only set dirty flag, never call _draw directly
    # ─────────────────────────────────────────────────────────────────────────

    def _on_joint_slider(self, _):
        if self.mode != "FK":
            return
        for i, sl in enumerate(self._joint_sliders):
            self.q_curr[i] = (np.deg2rad(sl.val)
                              if self.robot.joint_type[i] == 'r'
                              else sl.val)
        self._redraw_flag = True

    def _on_goal_slider(self, _):
        if self.mode != "IK":
            return
        self.goal_pos = _t([self._goal_sliders[0].val,
                            self._goal_sliders[1].val,
                            self._goal_sliders[2].val])
        self.goal_rpy = _t([self._goal_sliders[3].val,
                            self._goal_sliders[4].val,
                            self._goal_sliders[5].val])
        self._redraw_flag = True

    def _on_path_start_slider(self, _):
        self._path_start_pos = _t([self._path_start_sliders[0].val,
                                   self._path_start_sliders[1].val,
                                   self._path_start_sliders[2].val])
        self._path_start_rpy = _t([self._path_start_sliders[3].val,
                                   self._path_start_sliders[4].val,
                                   self._path_start_sliders[5].val])

    def _on_path_end_slider(self, _):
        self._path_end_pos = _t([self._path_end_sliders[0].val,
                                 self._path_end_sliders[1].val,
                                 self._path_end_sliders[2].val])
        self._path_end_rpy = _t([self._path_end_sliders[3].val,
                                 self._path_end_sliders[4].val,
                                 self._path_end_sliders[5].val])

    def _on_vel_lambda(self, val, idx):
        self.vel_lambdas[idx] = val

    def _on_acc_lambda(self, val, idx):
        self.acc_lambdas[idx] = val

    # ─────────────────────────────────────────────────────────────────────────
    # Button callbacks
    # ─────────────────────────────────────────────────────────────────────────

    def _on_apply_weights(self, _):
        try:
            self.pos_w   = float(self._textboxes["pos_w"].text)
            self.rot_w   = float(self._textboxes["rot_w"].text)
            self.crash_w = float(self._textboxes["crash_w"].text)
            self.dist_w  = float(self._textboxes["dist_w"].text)
            self.thresh_dist_val = float(self._textboxes["thresh_dist"].text)
            self.thresh_deg_val  = float(self._textboxes["thresh_deg"].text)
        except ValueError as e:
            self._status(f"Bad value: {e}", "#ff4444"); return
        self.oracle = self._make_oracle()
        self.oracle.L.set_pass_thresholds(self.thresh_dist_val, self.thresh_deg_val)
        self._status("Weights applied", "#00ff88")

    def _on_commit_pose(self, _):
        if self.q_new is None:
            self.q_prev = self.q_curr.clone()
        else:
            delta = self.q_new - self.q_curr
            self.q_prev = self.q_curr.clone()
            self.q_curr = self.q_new.clone()
            self.delta_q_prev = delta.clone()
            # Sync FK sliders
            for i, sl in enumerate(self._joint_sliders):
                sl.set_val(np.rad2deg(float(self.q_curr[i]))
                           if self.robot.joint_type[i] == 'r'
                           else float(self.q_curr[i]))
        self._status("Pose committed", "#00ff88")
        self._redraw_flag = True

    def _on_solve_ik(self, _):
        if self._worker_running:
            self._status("Solver running...", "#ffaa00"); return
        if self.mode != "IK":
            self._status("Switch to IK mode first.", "#ffaa00"); return
        self._status("Solving IK...", "#ffaa00")
        self._worker_running = True
        threading.Thread(target=self._worker_ik, daemon=True).start()

    def _on_plan_traj(self, _):
        if self._worker_running:
            self._status("Solver running...", "#ffaa00"); return
        self._traj_playing = False
        self._btn_play.label.set_text("Play ▶")
        self._status("Planning trajectory...", "#ffaa00")
        self._worker_running = True
        # Pass a flag so the worker knows whether to use slider endpoints or random ones.
        # _randomise_next_plan is set to True by _on_randomise and consumed once here.
        use_random = getattr(self, "_randomise_next_plan", False)
        self._randomise_next_plan = False
        threading.Thread(target=self._worker_plan,
                         args=(use_random,), daemon=True).start()

    def _on_randomise(self, _):
        """
        Generate guaranteed-reachable random start/end poses using
        planner.make_random_poses(), then reflect them back onto the
        path sliders so the user can see what was chosen.

        The old approach sampled random XYZ/RPY values in workspace which
        were frequently unreachable.  Sampling random joint angles and
        running FK guarantees both endpoints are within the robot's joint
        limits.
        """
        start_q = self.planner.make_random_poses()   # random joint angles within bounds
        end_q   = self.planner.make_random_poses()

        # FK-forward to workspace so we can show the values on the sliders
        self.robot.q_vect = start_q
        s_pos = _np(self.robot.give_ds()[-1])
        s_rpy = _np(self._6d_to_rpy_approx(to_6D_R(self.robot.give_Rs()[-1])))

        self.robot.q_vect = end_q
        e_pos = _np(self.robot.give_ds()[-1])
        e_rpy = _np(self._6d_to_rpy_approx(to_6D_R(self.robot.give_Rs()[-1])))

        # Restore current pose (sliders control q_curr, not robot.q_vect)
        self.robot.q_vect = self.q_curr.clone()

        # Push values onto the sliders (clamped to slider range)
        for sl, v in zip(self._path_start_sliders[:3], s_pos):
            sl.set_val(float(np.clip(v, sl.valmin, sl.valmax)))
        for sl, v in zip(self._path_start_sliders[3:], s_rpy):
            sl.set_val(float(np.clip(v, sl.valmin, sl.valmax)))
        for sl, v in zip(self._path_end_sliders[:3], e_pos):
            sl.set_val(float(np.clip(v, sl.valmin, sl.valmax)))
        for sl, v in zip(self._path_end_sliders[3:], e_rpy):
            sl.set_val(float(np.clip(v, sl.valmin, sl.valmax)))

        # Store the raw joint poses so _worker_plan can pass start_q to
        # follow_trajectory without re-solving IK for the first waypoint.
        self._random_start_q = start_q
        self._random_end_q   = end_q
        self._randomise_next_plan = True   # tell Plan Traj to use these

        self._status("Randomised — press Plan Traj", "#00ff88")

    def _on_play_stop(self, _):
        if not self._traj_qs:
            self._status("No trajectory — plan first.", "#ffaa00"); return
        self._traj_playing = not self._traj_playing
        self._btn_play.label.set_text("Stop ■" if self._traj_playing else "Play ▶")
        if self._traj_playing:
            self._traj_frame = 0
        self.fig.canvas.draw_idle()

    # ─────────────────────────────────────────────────────────────────────────
    # Worker threads
    # ─────────────────────────────────────────────────────────────────────────

    def _worker_ik(self):
        """Single-step IK solve."""
        try:
            goal_6d = _rpy_to_6d(self.goal_rpy)
            oracle  = self._make_oracle()
            oracle.L.set_pass_thresholds(self.thresh_dist_val, self.thresh_deg_val)
            oracle.reset_vars()
            oracle.delta_q_prev = self.delta_q_prev.clone()
            oracle.q_curr       = self.q_curr.clone()
            self.robot.q_vect   = self.q_curr.clone()

            res     = oracle.get_IK(Goal_Posi=self.goal_pos, Goal_Ori_6D=goal_6d)
            delta_q = _t(res.x)
            q_new   = self.q_curr + delta_q
            passed  = oracle.L.Pass
            crashed = oracle.L.crashed

            breakdown = self._compute_loss_breakdown(oracle.L, delta_q, goal_6d, crashed)

            self.q_new       = q_new
            self._ik_passed  = passed
            self.delta_q_prev = delta_q.clone()

            self._update_readout(breakdown, passed=passed)
            self._status(
                f"{'CRASHED' if crashed else ('PASS' if passed else 'FAIL')} "
                f"| converged={res.success}",
                "#ff4444" if crashed else ("#00ff88" if passed else "#ffaa00"))
        except Exception as e:
            self._status(f"IK error: {e}", "#ff4444")
        finally:
            self._worker_running = False
            self._redraw_flag = True

    def _worker_plan(self, use_random: bool):
        """
        Full trajectory planning via PathPlannerMath + Oracle.follow_trajectory.

        Two paths:
          use_random=True  — uses planner.random_MoveL() with the joint poses
                             already stored in self._random_start_q/end_q by
                             _on_randomise().  Endpoints are guaranteed reachable.
          use_random=False — builds a MoveL from the slider endpoint values (user
                             has set them manually in workspace coordinates).

        In both cases oracle.follow_trajectory() receives the packed tuple
        (trajectory_tensor, t_delay, start_q) and handles:
          • posing the robot to start_q before solving frame 0, so the solver
            never has to jump across joint space
          • threading the velocity / acceleration history through every frame
        """
        try:
            frames   = int(self._sl_frames.val)
            oracle   = self._make_oracle()
            oracle.L.set_pass_thresholds(self.thresh_dist_val, self.thresh_deg_val)
            oracle.rob.q_vect = self.q_curr.clone()

            # ── Build the workspace trajectory ────────────────────────────────
            if use_random and hasattr(self, "_random_start_q"):
                # random_MoveL needs the robot at start_q to read FK for start pose
                self.planner.robot.q_vect = self._random_start_q.clone()
                traj_packed = self.planner.random_MoveL(tot_time=1.0, frames=frames)
                # random_MoveL returns (traj_tensor, t_delay, start_q);
                # replace the start_q it returns (None from random path) with the
                # actual stored start joint pose so follow_trajectory can pose to it.
                traj_tensor, t_delay, _ = traj_packed
                traj_packed = (traj_tensor, t_delay, self._random_start_q.clone())
            else:
                # Manual: slider values are in workspace (pos + RPY).
                # Build the 9-D pos+6D vector from the path sliders.
                start_6d      = _rpy_to_6d(self._path_start_rpy)
                end_6d        = _rpy_to_6d(self._path_end_rpy)
                start_pos_ori = torch.cat([self._path_start_pos, start_6d])
                end_pos_ori   = torch.cat([self._path_end_pos,   end_6d])
                traj_packed   = self.planner.MoveL(
                    tot_time=1.0, frames=frames,
                    start_pos_ori=start_pos_ori,
                    end_pos_ori=end_pos_ori)
                # MoveL returns (tensor, t_delay, q_init=None); q_curr is
                # already set on oracle.rob above, so None means "start from
                # wherever the robot currently is" — which is correct here.

            self._status(f"Solving {frames} frames...", "#ffaa00")

            # ── Solve IK for every frame via follow_trajectory ────────────────
            # follow_trajectory unpacks (trajectory, t_delay, start_q), poses
            # the robot to start_q if provided, then solves frame-by-frame
            # while carrying velocity / acceleration history between steps.
            q_results, t_delay, any_crashed = oracle.follow_trajectory(traj_packed)
            # q_results is a list of scipy OptimizeResult, one per frame.
            # oracle.rob.q_vect is updated in-place each step, so we read the
            # solved joint poses directly from the stored results.

            # Rebuild the per-frame data the renderer needs
            traj_tensor = traj_packed[0]   # (frames, 9) workspace trajectory
            qs     = []
            goals  = []
            passes = []

            # Re-run pass/fail check per frame using the solved joint poses.
            # We can't rely on oracle.L.Pass (it reflects only the last frame),
            # so re-evaluate each result against its workspace goal.
            oracle_check = self._make_oracle()
            oracle_check.L.set_pass_thresholds(self.thresh_dist_val, self.thresh_deg_val)

            for fi, res in enumerate(q_results):
                delta_q = _t(res.x)
                # The joint pose after this step: oracle updated rob.q_vect
                # in-place, but we need per-frame poses.  Reconstruct from
                # the cumulative deltas stored in q_results.
                # Simpler: oracle stored final q in rob.q_vect after each step;
                # read q_curr + delta_q relative to whatever q_curr was at frame fi.
                # Because follow_trajectory updates rob.q_vect each step, the
                # cleanest way is to just trust res.x and the oracle's own
                # running q_curr — the result *is* the new joint pose.
                q_fi = _t(res.x)   # this is delta_q, not absolute pose
                # We need absolute pose.  follow_trajectory applies
                # rob.q_vect = q_curr + delta_q each step, so we can reconstruct:
                # We'll re-run a lightweight FK pass.
                # Actually: oracle.rob.q_vect after frame fi holds q_curr[fi+1].
                # The cleanest approach is to accumulate ourselves from start_q.
                # We stored q_results as OptimizeResults; each .x is the delta_q
                # from q_curr at *that* step.  Accumulate:
                if fi == 0:
                    start_q_abs = traj_packed[2]   # the start_q passed in
                    if start_q_abs is None:
                        start_q_abs = self.q_curr.clone()
                    running_q = _t(start_q_abs).clone()
                running_q = running_q + _t(res.x)

                frame  = traj_tensor[fi]
                gpos   = frame[:3].clone()
                g6d    = frame[3:].clone()
                grpy   = self._6d_to_rpy_approx(g6d)

                # Pass/fail: set robot to running_q and evaluate
                oracle_check.rob.q_vect = running_q.clone()
                pos_fi  = oracle_check.rob.give_ds()[-1]
                ori_fi  = oracle_check.rob.give_Rs()[-1]
                e_pos_fi = float(oracle_check.L.err_pos(pos_curr=pos_fi, pos_G_ws=gpos))
                e_ori_fi = float(oracle_check.L.err_ori(new_ori_SO3=ori_fi, G_SO3=to_SO3(g6d)))
                passed_fi = oracle_check.L.get_pass_or_fail(
                    e_pos=_t([e_pos_fi]), e_ori=_t([e_ori_fi]))

                qs.append(running_q.clone())
                goals.append((gpos, grpy))
                passes.append(passed_fi)

            self._traj_qs    = qs
            self._traj_goals = goals
            self._traj_pass  = passes
            self._traj_frame = 0

            n_pass = sum(passes)
            crash_tag = "  CRASHED" if any_crashed else ""
            self._status(
                f"Planned {frames} frames  ({n_pass}/{frames} pass){crash_tag}",
                "#ff4444" if any_crashed else "#00ff88")

        except Exception as e:
            self._status(f"Plan error: {e}", "#ff4444")
            import traceback; traceback.print_exc()
        finally:
            self._worker_running = False
            self._redraw_flag = True

    @staticmethod
    def _6d_to_rpy_approx(r6d: torch.Tensor) -> torch.Tensor:
        """Convert 6D rotation back to RPY degrees for display only."""
        R = to_SO3(r6d)
        pitch = float(torch.arcsin(-R[2, 0]).clamp(-1, 1))
        roll  = float(torch.arctan2(R[2, 1], R[2, 2]))
        yaw   = float(torch.arctan2(R[1, 0], R[0, 0]))
        return _t([np.rad2deg(roll), np.rad2deg(pitch), np.rad2deg(yaw)])

    # ─────────────────────────────────────────────────────────────────────────
    # Loss helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _compute_loss_breakdown(self, L, delta_q, goal_6d, crashed) -> dict:
        q_new = self.q_curr + delta_q
        self.robot.q_vect = q_new
        pos_new = self.robot.give_ds()[-1]
        ori_new = self.robot.give_Rs()[-1]
        R_goal  = to_SO3(goal_6d)
        e_pos  = float(L.err_pos(pos_curr=pos_new, pos_G_ws=self.goal_pos))
        e_ori  = float(L.err_ori(new_ori_SO3=ori_new, G_SO3=R_goal))
        e_vel  = float(L.err_vel(joint_velocity=delta_q))
        e_acc  = float(L.err_acc(joint_vel_new=delta_q, joint_vel_old=self.delta_q_prev))
        e_dist = float(L.err_dist(new_pose=q_new, m=-1.0, b=1.0))
        return dict(e_pos=e_pos, e_ori=e_ori, e_vel=e_vel, e_acc=e_acc,
                    e_dist=e_dist, total=e_pos+e_ori+e_vel+e_acc+e_dist,
                    crashed=crashed)

    def _update_readout(self, bd: dict, passed: bool | None = None):
        pf = "" if passed is None else ("  PASS" if passed else "  FAIL")
        lines = [
            f"{'e_pos':>7}: {bd['e_pos']:>9.4f}",
            f"{'e_ori':>7}: {bd['e_ori']:>9.4f}",
            f"{'e_vel':>7}: {bd['e_vel']:>9.4f}",
            f"{'e_acc':>7}: {bd['e_acc']:>9.4f}",
            f"{'e_dist':>7}: {bd['e_dist']:>9.4f}",
            f"{'─'*20}",
            f"{'TOTAL':>7}: {bd['total']:>9.4f}{pf}",
            f"{'CRASHED':>7}: {bd['crashed']}",
        ]
        self._readout_text.set_text("\n".join(lines))

    def _status(self, msg: str, color: str = "#00ff88"):
        self._readout_text.set_text(
            self._readout_text.get_text().split("\n")[0]  # keep loss lines
            if "\n" in self._readout_text.get_text() else "")
        # Append status as last line via ax title
        self._ax_readout.set_title(f"Loss / Status  —  {msg}",
                                   color=color, fontsize=8, pad=2)
        self.fig.canvas.draw_idle()

    # ─────────────────────────────────────────────────────────────────────────
    # 3-D rendering  — ONLY called on the main thread via _tick or _on_first_draw
    # ─────────────────────────────────────────────────────────────────────────

    def _draw(self):
        ax = self.ax3d
        ax.cla()
        self._style_3d(float(self.robot.max_reach))

        if self.mode == "Path" and self._traj_qs:
            self._draw_trajectory(ax)
        else:
            self._draw_three_robots(ax)
            self._draw_goal(ax)

    def _draw_three_robots(self, ax):
        self._draw_robot(ax, self._rob_prev, self.q_prev,
                         "#c77dff", "magenta", alpha=0.45)
        self._draw_robot(ax, self._rob_curr, self.q_curr,
                         "#00b4d8", "cyan",    alpha=1.0)
        if self.q_new is not None:
            self._draw_robot(ax, self._rob_new, self.q_new,
                             "#f4a261", "orange", alpha=0.75)

    def _draw_goal(self, ax):
        """Draw goal marker; colour depends on pass/fail."""
        # Show goal if IK mode, or if a result exists
        if not (self.mode == "IK" or self.q_new is not None):
            return
        # non-zero check: show only if goal was actually set
        if torch.allclose(self.goal_pos, torch.zeros(3, dtype=torch.float64)) and \
           torch.allclose(self.goal_rpy, torch.zeros(3, dtype=torch.float64)):
            return

        gp  = _np(self.goal_pos)
        Rn  = _np(_rpy_to_SO3(self.goal_rpy))
        col = ("#00ff88" if self._ik_passed
               else ("#ff4444" if self._ik_passed is False
               else "lime"))
        ax.scatter(*gp, color=col, s=70, zorder=6)
        ql = self.quiver_len
        for ci, c in enumerate(["red", "green", "blue"]):
            ax.quiver(gp[0], gp[1], gp[2],
                      Rn[0,ci], Rn[1,ci], Rn[2,ci],
                      length=ql, color=c, alpha=0.85)

    def _draw_trajectory(self, ax):
        """Render full trajectory ghost trail with opacity decay."""
        n = len(self._traj_qs)
        if n == 0:
            return

        # Which frame is "current" in the trail?
        cur = self._traj_frame if self._traj_playing else (n - 1)

        for fi in range(n):
            # Opacity: most recent frame fully opaque, oldest nearly invisible
            age   = (cur - fi) % n          # 0 = newest, n-1 = oldest
            alpha = max(0.08, 1.0 - age / max(n - 1, 1))

            q_fi   = self._traj_qs[fi]
            passed = self._traj_pass[fi] if self._traj_pass else None

            # Link / joint colour: cyan trail, red if collision
            self._draw_robot(ax, self._rob_new, q_fi,
                             "#00b4d8", "cyan", alpha=alpha * 0.9)

            # Goal marker
            if self._traj_goals:
                gpos, grpy = self._traj_goals[fi]
                gp  = _np(gpos)
                Rn  = _np(_rpy_to_SO3(grpy))
                gcol = ("#00ff88" if passed
                        else ("#ff4444" if passed is False
                        else "lime"))
                ax.scatter(*gp, color=gcol, s=30, alpha=alpha, zorder=6)
                ql = self.quiver_len * 0.6
                for ci, c in enumerate(["red", "green", "blue"]):
                    ax.quiver(gp[0], gp[1], gp[2],
                              Rn[0,ci], Rn[1,ci], Rn[2,ci],
                              length=ql, color=c, alpha=alpha * 0.7)

        # Always draw the "live" current robot on top in full opacity
        self._draw_robot(ax, self._rob_curr, self.q_curr,
                         "#00b4d8", "cyan", alpha=1.0)

    def _draw_robot(self, ax, rob, q_vect,
                    link_color: str, joint_color: str, alpha: float):
        rob.q_vect  = _t(q_vect)
        positions   = [_np(p) for p in rob.give_ds()]
        rotations   = [_np(R) for R in rob.give_Rs()]
        crashed, _  = rob.do_fk_and_check_crash()
        lc = "red" if crashed else link_color
        jc = "red" if crashed else joint_color

        xs, ys, zs = zip(*positions)
        ax.plot(xs, ys, zs, "-", color=lc, alpha=alpha, linewidth=1.8)
        ax.scatter(xs, ys, zs, color=jc, s=14, alpha=alpha, zorder=4)

        ql = self.quiver_len
        for pos, rot in zip(positions, rotations):
            for ci, c in enumerate(["red", "green", "blue"]):
                ax.quiver(pos[0], pos[1], pos[2],
                          rot[0,ci], rot[1,ci], rot[2,ci],
                          length=ql, color=c, alpha=alpha * 0.65)

        if crashed:
            ax.text2D(0.5, 0.97, "SELF-COLLISION",
                      transform=ax.transAxes, ha="center", va="top",
                      color="red", fontsize=10, fontweight="bold")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
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
    robot = Robot_math(a=my_a, alpha=my_alpha, d=my_d, theta=my_theta,
                       joint_type=["r"]*6, bounds=my_bounds,
                       fail_dist=[0.1]*6)
    robot.WT = robot.make_homogenous_transformation(
        yaw=0, pitch=0, roll=180, x=0, y=0, z=0)

    VirtualTwinLive(robot=robot,
                    pos_w=1.0, rot_w=0.5, vel_w=0.1, acc_w=0.1,
                    crash_w=2.0, dist_w=0.1,
                    quiver_len=1.5, unit_label="inches")


if __name__ == "__main__":
    main()