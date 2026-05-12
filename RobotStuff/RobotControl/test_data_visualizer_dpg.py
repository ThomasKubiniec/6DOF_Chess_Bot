'''
visualizer_app.py — DPG control panel + matplotlib robot visualizer.

Thread model
────────────
  MAIN THREAD  — matplotlib owns this. FuncAnimation drives redraws at
                 ~20 Hz via plt.show(block=True). No DPG or solver code
                 ever touches matplotlib from any other thread.

  DPG THREAD   — daemon thread running the DearPyGui event loop. All DPG
                 callbacks live here. When a callback wants to update the
                 3-D view it pushes a SceneSnapshot onto a thread-safe
                 queue; it never calls matplotlib directly.

  PLAY THREAD  — short-lived daemon spawned by the Play button. Advances
                 frame_idx and enqueues SceneSnapshots at the selected FPS.
                 Also calls dpg.set_value() which is DPG-thread-safe.

Layout:
  ┌─────────────────────────────────────────────┐
  │  FILE  ──  [Browse…]  path/to/file.pkl      │
  │  MOVE  ──  [Random Sample]  move 3 of 12    │
  │─────────────────────────────────────────────│
  │  SOLVER  ○ Oracle  ○ Simple IK  ○ Imitation │
  │  LEG     ○ 0  ○ 1  ○ 2  ○ 3                │
  │─────────────────────────────────────────────│
  │  [◀ Prev]  frame 14 / 100  [Next ▶]         │
  │  [▶ Play]  FPS ──●──────                    │
  │─────────────────────────────────────────────│
  │  Quality:  GOOD   e_pos 0.042  e_ori 0.011  │
  │─────────────────────────────────────────────│
  │  LEG SUMMARY  (all solvers, current leg)    │
  │  oracle     good 97/100  okay 3/100 bad 0   │
  │  simple_ik  good 88/100  okay 9/100 bad 3   │
  │  imit_ik    good 72/100  okay 21/100 bad 7  │
  └─────────────────────────────────────────────┘
'''

import sys
import time
import queue
import random
import threading
import pickle

import numpy as np
import torch

import dearpygui.dearpygui as dpg

# matplotlib MUST be imported and configured before anything else touches it,
# and plt.show(block=True) will be called from the main thread.
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401

from forward_kinematics import Robot_math

import train_simple_ik   # reuse build_robot()


# ─────────────────────────────────────────────────────────────────────────────
# Colour palette
# ─────────────────────────────────────────────────────────────────────────────
_PAL = {
    'bg'        : (18,  20,  28,  255),
    'panel'     : (26,  29,  42,  255),
    'border'    : (50,  55,  80,  255),
    'accent'    : (82, 160, 255,  255),
    'good'      : (80, 220,  80,  255),
    'okay'      : (255, 180,  40,  255),
    'bad'       : (230,  50,  50,  255),
    'text'      : (210, 215, 235,  255),
    'text_dim'  : (110, 115, 145,  255),
    'btn'       : (50,  90, 180,  255),
    'btn_hover' : (70, 110, 210,  255),
    'btn_active': (40,  70, 150,  255),
}

# Matplotlib draw colours
_MC = {
    'current'  : 'orange',
    'ghost'    : (0.4, 0.8, 1.0),
    'good'     : 'limegreen',
    'okay'     : 'orange',
    'bad'      : 'red',
    'axes'     : ['red', 'green', 'blue'],
}

_QUIVER_FRAC       = 0.15
_GHOST_ALPHA_START = 0.55
_GHOST_DISCOUNT    = 0.82
_ANIM_INTERVAL_MS  = 50    # FuncAnimation interval → ~20 Hz


# ─────────────────────────────────────────────────────────────────────────────
# Scene snapshot — the only thing that crosses the thread boundary
# ─────────────────────────────────────────────────────────────────────────────
class SceneSnapshot:
    '''
    Immutable description of what to draw next.
    Built by the DPG/play thread, consumed by FuncAnimation on the main thread.
    '''
    __slots__ = ('current_q', 'ghost_qs', 'markers')

    def __init__(self, current_q, ghost_qs, markers):
        self.current_q = current_q    # torch.Tensor (n,)  or None
        self.ghost_qs  = ghost_qs     # list of torch.Tensor (n,), oldest first
        self.markers   = markers      # list of {pos, R, quality}


# ─────────────────────────────────────────────────────────────────────────────
# App state
# ─────────────────────────────────────────────────────────────────────────────
class AppState:
    def __init__(self):
        self.moves    : list  = []
        self.move_idx : int   = -1
        self.solver   : str   = 'oracle'
        self.leg_idx  : int   = 0
        self.frame_idx: int   = 0
        self.playing  : bool  = False
        self.play_fps : float = 20.0
        self.current_leg_frames: list = []

    @property
    def n_moves(self):  return len(self.moves)
    @property
    def n_frames(self): return len(self.current_leg_frames)

    def current_frame(self):
        if not self.current_leg_frames:
            return None
        return self.current_leg_frames[
            max(0, min(self.frame_idx, self.n_frames - 1))]

    def resolve_leg(self):
        if self.move_idx < 0 or not self.moves:
            self.current_leg_frames = []
            return
        self.current_leg_frames = (
            self.moves[self.move_idx]['legs'][self.leg_idx]
            .get(self.solver, []))
        self.frame_idx = 0


# ─────────────────────────────────────────────────────────────────────────────
# Matplotlib renderer  (main-thread only)
# ─────────────────────────────────────────────────────────────────────────────
class MatplotlibRenderer:
    '''
    Owns the matplotlib figure and axes.
    _animate() is called by FuncAnimation on the main thread — it drains the
    scene_queue and redraws if there is a new snapshot.
    '''

    def __init__(self, robot: Robot_math, scene_queue: queue.Queue):
        self.robot       = robot
        self.scene_queue = scene_queue

        plt.style.use('dark_background')
        self._fig = plt.figure(figsize=(7, 6))
        self._ax  = self._fig.add_subplot(111, projection='3d')
        self._fig.canvas.manager.set_window_title('Robot Virtual Twin')

        self._last_snap: SceneSnapshot | None = None

        # FuncAnimation keeps a reference to itself to stay alive
        self._anim = FuncAnimation(
            self._fig,
            self._animate,
            interval=_ANIM_INTERVAL_MS,
            blit=False,
            cache_frame_data=False,
        )

    def _animate(self, _frame):
        '''Called by FuncAnimation. Drains queue, redraws on new snapshot.'''
        snap = None
        try:
            while True:                      # drain — use only the latest
                snap = self.scene_queue.get_nowait()
        except queue.Empty:
            pass

        if snap is None:
            return                           # nothing new — skip redraw

        self._last_snap = snap
        self._draw(snap)

    def _fk(self, q):
        '''FK without mutating robot.q_vect permanently.'''
        saved = self.robot.q_vect.clone()
        self.robot.q_vect = q.to(torch.float64)
        ds = [d.detach().cpu().numpy() for d in self.robot.give_ds()]
        Rs = [R.detach().cpu().numpy() for R in self.robot.give_Rs()]
        self.robot.q_vect = saved
        return ds, Rs

    def _draw_robot(self, ax, q, color, alpha):
        ds, Rs = self._fk(q)
        xs = [d[0] for d in ds]
        ys = [d[1] for d in ds]
        zs = [d[2] for d in ds]
        ax.plot(xs, ys, zs, '-', color=color, alpha=alpha, linewidth=1.8)
        ax.scatter(xs, ys, zs, color=color, s=14, alpha=alpha, zorder=4)
        ql = float(self.robot.max_reach) * _QUIVER_FRAC
        for pos, rot in zip(ds, Rs):
            for ci, c in enumerate(_MC['axes']):
                ax.quiver(pos[0], pos[1], pos[2],
                          rot[0, ci], rot[1, ci], rot[2, ci],
                          length=ql, color=c,
                          alpha=alpha * 0.55, normalize=True)

    def _draw_marker(self, ax, m):
        pos   = m['pos']
        R     = m['R']
        color = _MC.get(m.get('quality', 'bad'), _MC['bad'])
        ax.scatter([pos[0]], [pos[1]], [pos[2]],
                   color=color, s=60, zorder=5, depthshade=False)
        ql = float(self.robot.max_reach) * _QUIVER_FRAC * 0.8
        for ci, c in enumerate(_MC['axes']):
            ax.quiver(pos[0], pos[1], pos[2],
                      R[0, ci], R[1, ci], R[2, ci],
                      length=ql, color=c, alpha=0.9, normalize=True)

    def _draw(self, snap: SceneSnapshot):
        ax = self._ax
        ax.cla()

        mr = float(self.robot.max_reach)
        ax.set_xlim(-mr, mr); ax.set_ylim(-mr, mr); ax.set_zlim(-mr, mr)
        ax.set_xlabel('X');   ax.set_ylabel('Y');   ax.set_zlabel('Z')
        ax.set_title('Robot Virtual Twin', color='white')

        # Ghost trail — oldest most transparent
        ghosts = snap.ghost_qs
        if ghosts:
            a = _GHOST_ALPHA_START
            alphas = []
            for _ in ghosts:
                alphas.append(a)
                a *= _GHOST_DISCOUNT
            alphas = list(reversed(alphas))   # index 0 = oldest = lowest alpha
            for q_g, alpha in zip(ghosts, alphas):
                self._draw_robot(ax, q_g, _MC['ghost'], alpha)

        # Target markers
        for m in snap.markers:
            self._draw_marker(ax, m)

        # Current pose
        if snap.current_q is not None:
            self._draw_robot(ax, snap.current_q, _MC['current'], 0.95)

        self._fig.canvas.draw_idle()


# ─────────────────────────────────────────────────────────────────────────────
# Scene builder  (called from DPG / play thread — produces snapshots)
# ─────────────────────────────────────────────────────────────────────────────
def _build_snapshot(state: AppState, frame_idx: int | None = None) -> SceneSnapshot:
    if not state.current_leg_frames:
        return SceneSnapshot(current_q=None, ghost_qs=[], markers=[])

    idx   = frame_idx if frame_idx is not None else state.frame_idx
    idx   = max(0, min(idx, state.n_frames - 1))
    frame = state.current_leg_frames[idx]

    # Ghost trail: up to 8 previous frames
    trail_start = max(0, idx - 8)
    ghost_qs    = [state.current_leg_frames[i]['q']
                   for i in range(trail_start, idx)]

    # Target markers: every 5th frame + current
    markers = []
    for i, f in enumerate(state.current_leg_frames):
        if i % 5 == 0 or i == idx:
            markers.append({
                'pos'    : f['xyz'].detach().cpu().numpy(),
                'R'      : f['SO3'].detach().cpu().numpy(),
                'quality': f['quality'],
            })

    return SceneSnapshot(
        current_q = frame['q'],
        ghost_qs  = ghost_qs,
        markers   = markers,
    )


# ─────────────────────────────────────────────────────────────────────────────
# DPG helpers
# ─────────────────────────────────────────────────────────────────────────────
def _quality_color(q: str):
    return {'good': _PAL['good'], 'okay': _PAL['okay'], 'bad': _PAL['bad']
            }.get(q, _PAL['bad'])


def _leg_summary_text(move: dict, leg_idx: int) -> str:
    leg = move['legs'][leg_idx]
    lines = []
    for solver in ('oracle', 'simple_ik', 'imit_ik'):
        frames = leg.get(solver, [])
        if not frames:
            lines.append(f'{solver:<12} no data')
            continue
        n    = len(frames)
        good = sum(1 for f in frames if f['quality'] == 'good')
        okay = sum(1 for f in frames if f['quality'] == 'okay')
        bad  = sum(1 for f in frames if f['quality'] == 'bad')
        lines.append(
            f'{solver:<12} good {good}/{n}  okay {okay}/{n}  bad {bad}/{n}')
    return '\n'.join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# DPG application  (runs entirely on its daemon thread)
# ─────────────────────────────────────────────────────────────────────────────
class DpgApp:
    '''
    All DPG window construction and callbacks live here.
    Communicates with matplotlib exclusively via scene_queue.
    '''

    def __init__(self, state: AppState, scene_queue: queue.Queue):
        self.state       = state
        self.scene_queue = scene_queue

    def _enqueue(self, frame_idx: int | None = None):
        '''Build a snapshot from current state and push it to the render queue.'''
        snap = _build_snapshot(self.state, frame_idx)
        self.scene_queue.put(snap)

    # ── callbacks ─────────────────────────────────────────────────────────────

    def _cb_browse(self, s, a):
        dpg.show_item('file_dialog')

    def _cb_file_selected(self, s, app_data):
        path = app_data.get('file_path_name', '')
        if not path:
            return
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
            self.state.moves    = data
            self.state.move_idx = -1
            self.state.current_leg_frames = []
            dpg.set_value('lbl_file', path)
            dpg.set_value('lbl_move', f'0 of {self.state.n_moves} loaded')
            self._refresh_ui()
        except Exception as e:
            dpg.set_value('lbl_file', f'Error: {e}')

    def _cb_sample(self, s, a):
        if not self.state.moves:
            return
        self.state.move_idx = random.randint(0, self.state.n_moves - 1)
        self.state.resolve_leg()
        dpg.set_value('lbl_move',
                      f'Move {self.state.move_idx + 1} of {self.state.n_moves}')
        self._refresh_ui()
        self._enqueue()

    def _cb_solver(self, s, app_data):
        self.state.solver = app_data
        self.state.resolve_leg()
        self._refresh_ui()
        self._enqueue()

    def _cb_leg(self, s, app_data):
        self.state.leg_idx = int(app_data)
        self.state.resolve_leg()
        self._refresh_ui()
        self._enqueue()

    def _cb_prev_frame(self, s, a):
        if self.state.frame_idx > 0:
            self.state.frame_idx -= 1
        self._refresh_frame_ui()
        self._enqueue()

    def _cb_next_frame(self, s, a):
        if self.state.frame_idx < self.state.n_frames - 1:
            self.state.frame_idx += 1
        self._refresh_frame_ui()
        self._enqueue()

    def _cb_frame_slider(self, s, app_data):
        self.state.frame_idx = int(app_data)
        self._refresh_frame_ui()
        self._enqueue()

    def _cb_play(self, s, a):
        if self.state.playing:
            self.state.playing = False
            dpg.set_item_label('btn_play', '▶  Play')
            return
        if not self.state.current_leg_frames:
            return
        self.state.playing = True
        dpg.set_item_label('btn_play', '■  Stop')
        threading.Thread(target=self._play_loop, daemon=True).start()

    def _cb_fps(self, s, app_data):
        self.state.play_fps = float(app_data)

    # ── play loop (its own daemon thread) ────────────────────────────────────

    def _play_loop(self):
        while self.state.playing:
            n = self.state.n_frames
            if n == 0:
                break
            self.state.frame_idx = (self.state.frame_idx + 1) % n
            self._enqueue(self.state.frame_idx)
            dpg.set_value('slider_frame', self.state.frame_idx)
            self._refresh_frame_ui()
            time.sleep(1.0 / max(1.0, self.state.play_fps))
        dpg.set_item_label('btn_play', '▶  Play')
        self.state.playing = False

    # ── UI refresh (safe to call from DPG thread or play thread) ─────────────

    def _refresh_ui(self):
        self._refresh_frame_ui()
        self._refresh_summary()

    def _refresh_frame_ui(self):
        n     = self.state.n_frames
        idx   = self.state.frame_idx
        frame = self.state.current_frame()

        dpg.set_value('lbl_frame', f'Frame {idx + 1} / {max(n, 1)}')
        dpg.configure_item('slider_frame', max_value=max(n - 1, 0))
        dpg.set_value('slider_frame', idx)

        if frame:
            q_str = frame['quality']
            dpg.set_value('lbl_quality', f'Quality:  {q_str.upper()}')
            dpg.configure_item('lbl_quality', color=list(_quality_color(q_str)))
            dpg.set_value('lbl_errors',
                          f'e_pos {frame["e_pos"]:.4f}    '
                          f'e_ori {frame["e_ori"]:.4f}')
        else:
            dpg.set_value('lbl_quality', 'Quality:  —')
            dpg.set_value('lbl_errors',  'e_pos —    e_ori —')

    def _refresh_summary(self):
        if self.state.move_idx < 0 or not self.state.moves:
            dpg.set_value('lbl_summary', 'No move loaded.')
            return
        dpg.set_value('lbl_summary',
                      _leg_summary_text(
                          self.state.moves[self.state.move_idx],
                          self.state.leg_idx))

    # ── window construction ───────────────────────────────────────────────────

    def _build(self):
        dpg.create_context()

        with dpg.theme() as global_theme:
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_color(dpg.mvThemeCol_WindowBg,      _PAL['bg'])
                dpg.add_theme_color(dpg.mvThemeCol_ChildBg,       _PAL['panel'])
                dpg.add_theme_color(dpg.mvThemeCol_Border,        _PAL['border'])
                dpg.add_theme_color(dpg.mvThemeCol_Text,          _PAL['text'])
                dpg.add_theme_color(dpg.mvThemeCol_Button,        _PAL['btn'])
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, _PAL['btn_hover'])
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive,  _PAL['btn_active'])
                dpg.add_theme_color(dpg.mvThemeCol_FrameBg,       _PAL['panel'])
                dpg.add_theme_color(dpg.mvThemeCol_SliderGrab,    _PAL['accent'])
                dpg.add_theme_color(dpg.mvThemeCol_CheckMark,     _PAL['accent'])
                dpg.add_theme_style(dpg.mvStyleVar_WindowRounding, 8)
                dpg.add_theme_style(dpg.mvStyleVar_FrameRounding,  6)
                dpg.add_theme_style(dpg.mvStyleVar_ItemSpacing,    8, 6)
                dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 14, 12)
        dpg.bind_theme(global_theme)

        with dpg.file_dialog(label='Open trajectory dataset',
                             tag='file_dialog',
                             width=640, height=420,
                             show=False,
                             callback=self._cb_file_selected):
            dpg.add_file_extension('.pkl', color=list(_PAL['accent']))
            dpg.add_file_extension('.*')

        with dpg.window(label='IK Solver Visualizer',
                        tag='main_win',
                        width=460, height=680,
                        no_close=True):

            dpg.add_text('IK SOLVER VISUALIZER', color=list(_PAL['accent']))
            dpg.add_separator(); dpg.add_spacer(height=4)

            # File
            with dpg.group(horizontal=True):
                dpg.add_button(label='Browse…', width=90,
                               callback=self._cb_browse)
                dpg.add_text('No file loaded', tag='lbl_file',
                             color=list(_PAL['text_dim']))

            dpg.add_spacer(height=6); dpg.add_separator(); dpg.add_spacer(height=6)

            # Move
            with dpg.group(horizontal=True):
                dpg.add_button(label='Random Sample', width=130,
                               callback=self._cb_sample)
                dpg.add_text('—', tag='lbl_move',
                             color=list(_PAL['text_dim']))

            dpg.add_spacer(height=6); dpg.add_separator(); dpg.add_spacer(height=6)

            # Solver radio
            dpg.add_text('Solver', color=list(_PAL['text_dim']))
            dpg.add_radio_button(
                items=['oracle', 'simple_ik', 'imit_ik'],
                tag='radio_solver', default_value='oracle',
                horizontal=True, callback=self._cb_solver)

            dpg.add_spacer(height=4)

            # Leg radio
            dpg.add_text('Leg', color=list(_PAL['text_dim']))
            dpg.add_radio_button(
                items=['0', '1', '2', '3'],
                tag='radio_leg', default_value='0',
                horizontal=True, callback=self._cb_leg)

            dpg.add_spacer(height=6); dpg.add_separator(); dpg.add_spacer(height=6)

            # Frame nav
            dpg.add_text('Frame — / —', tag='lbl_frame',
                         color=list(_PAL['text']))
            dpg.add_slider_int(tag='slider_frame', label='',
                               width=-1, min_value=0, max_value=0,
                               callback=self._cb_frame_slider)
            with dpg.group(horizontal=True):
                dpg.add_button(label='◀ Prev', width=90,
                               callback=self._cb_prev_frame)
                dpg.add_button(label='Next ▶', width=90,
                               callback=self._cb_next_frame)

            dpg.add_spacer(height=6); dpg.add_separator(); dpg.add_spacer(height=6)

            # Playback
            with dpg.group(horizontal=True):
                dpg.add_button(label='▶  Play', tag='btn_play',
                               width=110, callback=self._cb_play)
                dpg.add_text('FPS', color=list(_PAL['text_dim']))
                dpg.add_slider_float(tag='slider_fps', width=160,
                                     min_value=1.0, max_value=60.0,
                                     default_value=20.0, format='%.0f',
                                     callback=self._cb_fps)

            dpg.add_spacer(height=6); dpg.add_separator(); dpg.add_spacer(height=6)

            # Frame quality
            dpg.add_text('FRAME', color=list(_PAL['text_dim']))
            dpg.add_text('Quality:  —', tag='lbl_quality',
                         color=list(_PAL['text']))
            dpg.add_text('e_pos —    e_ori —', tag='lbl_errors',
                         color=list(_PAL['text_dim']))

            dpg.add_spacer(height=6); dpg.add_separator(); dpg.add_spacer(height=6)

            # Leg summary
            dpg.add_text('LEG SUMMARY  (all solvers)', color=list(_PAL['text_dim']))
            dpg.add_text('Load a file and sample a move.',
                         tag='lbl_summary', color=list(_PAL['text']))

        dpg.create_viewport(title='IK Solver Visualizer',
                            width=480, height=700,
                            min_width=460, min_height=600)
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.set_primary_window('main_win', True)

    def run_loop(self, initial_path: str | None = None):
        '''
        Build the window, optionally auto-load a file, then run the DPG
        event loop.  Designed to be called from a daemon thread.
        '''
        self._build()

        if initial_path:
            try:
                with open(initial_path, 'rb') as f:
                    self.state.moves = pickle.load(f)
                dpg.set_value('lbl_file', initial_path)
                dpg.set_value('lbl_move',
                              f'0 of {self.state.n_moves} loaded')
            except Exception as e:
                print(f'Failed to auto-load {initial_path}: {e}')

        while dpg.is_dearpygui_running():
            dpg.render_dearpygui_frame()

        dpg.destroy_context()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point — matplotlib on main thread, DPG on daemon thread
# ─────────────────────────────────────────────────────────────────────────────
def main():
    device = torch.device('cpu')
    robot  = train_simple_ik.build_robot(device)

    initial_path = sys.argv[1] if len(sys.argv) > 1 else None

    # Shared state and the queue that crosses the thread boundary
    state       = AppState()
    scene_queue : queue.Queue = queue.Queue(maxsize=4)

    # Build the matplotlib figure on the main thread before DPG starts,
    # so Tk is initialised here and never touched from the DPG thread.
    renderer = MatplotlibRenderer(robot=robot, scene_queue=scene_queue)

    # Start DPG on a daemon thread
    dpg_app = DpgApp(state=state, scene_queue=scene_queue)
    dpg_thread = threading.Thread(
        target=dpg_app.run_loop,
        kwargs={'initial_path': initial_path},
        daemon=True,
        name='DpgThread',
    )
    dpg_thread.start()

    # Hand the main thread to matplotlib — blocks until the figure is closed
    plt.show(block=True)

    # Clean up: signal DPG to stop if the user closed the matplotlib window
    try:
        dpg.stop_dearpygui()
    except Exception:
        pass


if __name__ == '__main__':
    main()