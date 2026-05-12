"""
dataset_analysis.py
===================
Two independent analysis classes:

  ApproximateDatasetAnalyzer
  --------------------------
  Replicates the sampling logic of data_generator (no Oracle, no IK) to
  rapidly produce workspace coverage statistics without spending hours on
  scipy solves.  Useful for understanding the *intended* distribution
  before committing to a full dataset run.

  Plots produced:
    1. Radial distribution histogram  — how often each radius is sampled
    2. XY / XZ / YZ density heatmaps  — spatial coverage of trajectory
       waypoints across all three projected planes

  RealDatasetAnalyzer
  -------------------
  Loads a pickled dataset produced by my_dataframe.save_dataset() and
  computes:
    1. Target-weight histogram         — "good / okay / bad" split
    2. XY / XZ / YZ density heatmaps  — EE positions recovered via FK
       on every stored q_curr in the dataset

Both classes expose a .run() method that generates and saves all figures.

Usage
-----
  # Approximate (no dataset file needed):
  python dataset_analysis.py --mode approx --frames 200000

  # Real dataset:
  python dataset_analysis.py --mode real --load training_data

  # Both:
  python dataset_analysis.py --mode both --load training_data
"""

import argparse
import pickle
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")          # headless — safe for Colab and servers
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm

from forward_kinematics import Robot_math
from path_planning_math import PathPlannerMath
from rot_math import YPR_SO3, to_SO3


# ---------------------------------------------------------------------------
# Shared robot config  (copy from make_dataset.py if you change the robot)
# ---------------------------------------------------------------------------
def make_robot(device=None) -> Robot_math:
    device = device or torch.device("cpu")
    robot = Robot_math(
        a          = [0.0, 7.375, 0.0,  0.0,  0.0,  0.0],
        alpha      = [np.deg2rad(90),  np.deg2rad(180), np.deg2rad(90),
                      np.deg2rad(90),  np.deg2rad(-90), np.deg2rad(0)],
        d          = [-3.5, 0.0, 0.0, 8.25, 0.0, 5.1875],
        theta      = [np.deg2rad(0),   np.deg2rad(0),   np.deg2rad(90),
                      np.deg2rad(180), np.deg2rad(0),   np.deg2rad(-90)],
        joint_type = ["r"] * 6,
        bounds     = [
            (np.deg2rad(-90),  np.deg2rad(90)),
            (np.deg2rad(-180), np.deg2rad(0)),
            (np.deg2rad(-90),  np.deg2rad(90)),
            (np.deg2rad(-90),  np.deg2rad(90)),
            (np.deg2rad(-90),  np.deg2rad(90)),
            (np.deg2rad(-90),  np.deg2rad(90)),
        ],
        fail_dist  = [0.1] * 6,
        device     = device,
    )
    robot.WT = robot.make_homogenous_transformation(
        yaw=0, pitch=0, roll=180, x=0, y=0, z=0)
    return robot


# ---------------------------------------------------------------------------
# Plot styling  (paper-ready)
# ---------------------------------------------------------------------------
STYLE = {
    "figure.facecolor":  "white",
    "axes.facecolor":    "#f7f7f7",
    "axes.edgecolor":    "#cccccc",
    "axes.linewidth":    0.8,
    "axes.titlesize":    11,
    "axes.titleweight":  "bold",
    "axes.labelsize":    9,
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "grid.color":        "white",
    "grid.linewidth":    0.6,
    "font.family":       "DejaVu Sans",
    "figure.dpi":        150,
}

CMAP = "plasma"    # perceptually uniform, prints well in greyscale


def _apply_style():
    plt.rcParams.update(STYLE)


def _density_heatmap(ax, xs, ys, xlabel, ylabel, title,
                     bins=80, lognorm=True):
    """Render a 2-D histogram as a filled heatmap on *ax*."""
    norm = LogNorm() if lognorm else None
    h, xedges, yedges = np.histogram2d(xs, ys, bins=bins)
    im = ax.imshow(
        h.T,
        origin="lower",
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        aspect="auto",
        cmap=CMAP,
        norm=norm,
        interpolation="bilinear",
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True)
    return im


# ===========================================================================
# 1.  ApproximateDatasetAnalyzer
# ===========================================================================
class ApproximateDatasetAnalyzer:
    """
    Reproduce the *sampling geometry* of data_generator without running
    any IK solvers.  Fast — 200k waypoints take a few seconds.

    Parameters
    ----------
    robot      : Robot_math instance (used for max_reach and FK).
    low_frac   : mean_low  = max_reach * low_frac
    high_frac  : mean_high = max_reach * high_frac
    std_low    : std of the inner (near-workspace-centre) radius distribution
    std_high   : std of the outer (near-edge) radius distribution
    p_low      : probability of drawing from the inner distribution
    frame_low  : minimum frames per trajectory  (matched to make_dataset)
    frame_high : maximum frames per trajectory
    """

    def __init__(self,
                 robot: Robot_math,
                 low_frac:  float = 0.5,
                 high_frac: float = 5/6,
                 std_low:   float = 0.4,
                 std_high:  float = 0.4,
                 p_low:     float = 0.75,
                 frame_low: int   = 20,
                 frame_high: int  = 100):

        self.robot      = robot
        self.device     = robot.device
        self.max_reach  = float(robot.max_reach)

        self.mean_low   = self.max_reach * low_frac
        self.mean_high  = self.max_reach * high_frac
        self.std_low    = std_low * self.max_reach   # convert fraction → world units
        self.std_high   = std_high * self.max_reach
        self.p_low      = p_low

        self.frame_low  = frame_low
        self.frame_high = frame_high

        self.path_planner = PathPlannerMath(my_robot=robot)

        # Results populated by generate()
        self.endpoint_radii: np.ndarray | None = None   # (N_endpoints,)
        self.waypoint_xyz:   np.ndarray | None = None   # (N_waypoints, 3)

    # ------------------------------------------------------------------
    def _sample_radius(self, mean: float, std: float) -> float:
        r = torch.nn.init.trunc_normal_(
            torch.empty(1), mean=mean, std=std, a=0.0, b=self.max_reach
        ).item()
        return r

    def _sample_xyz(self) -> torch.Tensor:
        p = torch.rand(1).item()
        unit = torch.nn.functional.normalize(
            torch.randn(3, dtype=torch.float64, device=self.device), p=2, dim=0)
        if p <= self.p_low:
            r = self._sample_radius(self.mean_low, self.std_low)
        else:
            r = self._sample_radius(self.mean_high, self.std_high)
        return unit * r

    def generate(self, target_waypoints: int = 200_000,
                 report_every: int = 10_000):
        """
        Draw random endpoints and interpolate trajectories between them
        until *target_waypoints* have been collected.

        Stores results in self.endpoint_radii and self.waypoint_xyz.
        """
        print(f"[Approx] Generating ≥{target_waypoints:,} waypoints ...")
        t0 = time.time()

        radii_list    = []
        xyz_list      = []
        waypoint_count = 0

        frames_range = list(range(self.frame_low, self.frame_high + 1))

        while waypoint_count < target_waypoints:
            # Sample two random endpoints
            p1 = self._sample_xyz()
            p2 = self._sample_xyz()

            radii_list.append(torch.linalg.vector_norm(p1).item())
            radii_list.append(torch.linalg.vector_norm(p2).item())

            frames = int(np.random.choice(frames_range))

            # Interpolate a 3-D straight line (position only — ignore orientation)
            traj = self.path_planner.cubic_interp_vect(
                vect_0=p1, vect_1=p2, f=frames, t=1.0
            )   # (frames, 9) — first 3 cols are XYZ

            xyz_list.append(traj[:, :3].cpu().numpy())
            waypoint_count += frames

            if waypoint_count % report_every < frames:
                print(f"  {waypoint_count:>8,} / {target_waypoints:,} waypoints  "
                      f"({time.time()-t0:.1f}s)", end="\r", flush=True)

        print()

        self.endpoint_radii = np.array(radii_list)
        self.waypoint_xyz   = np.vstack(xyz_list)

        elapsed = time.time() - t0
        print(f"[Approx] Done. {waypoint_count:,} waypoints, "
              f"{len(self.endpoint_radii)//2:,} trajectories "
              f"in {elapsed:.1f}s")

    # ------------------------------------------------------------------
    def plot(self, out_dir: str = ".", prefix: str = "approx"):
        """Save all figures to *out_dir*."""
        if self.endpoint_radii is None:
            raise RuntimeError("Call generate() first.")

        _apply_style()
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)

        # ── Figure 1: Radial distribution ─────────────────────────────
        fig, ax = plt.subplots(figsize=(7, 4))
        bins = np.linspace(0, self.max_reach, 80)
        ax.hist(self.endpoint_radii, bins=bins, color="#e05a2b",
                edgecolor="white", linewidth=0.4)
        ax.axvline(self.mean_low,  color="#1a6faf", lw=1.5,
                   linestyle="--", label=f"mean_low  = {self.mean_low:.2f}")
        ax.axvline(self.mean_high, color="#2ca02c", lw=1.5,
                   linestyle="--", label=f"mean_high = {self.mean_high:.2f}")
        ax.axvline(self.max_reach, color="#7f7f7f", lw=1.0,
                   linestyle=":", label=f"max_reach = {self.max_reach:.2f}")
        ax.set_xlabel("Radius from base (world units)")
        ax.set_ylabel("Count of sampled endpoints")
        ax.set_title("Radial Distribution of Sampled Endpoints\n"
                     f"(bimodal, p_low={self.p_low})")
        ax.legend(fontsize=8)
        ax.grid(True)
        fig.tight_layout()
        path1 = out / f"{prefix}_radial_distribution.png"
        fig.savefig(path1, dpi=150)
        plt.close(fig)
        print(f"  Saved → {path1}")

        # ── Figure 2: XY / XZ / YZ density heatmaps ──────────────────
        xs = self.waypoint_xyz[:, 0]
        ys = self.waypoint_xyz[:, 1]
        zs = self.waypoint_xyz[:, 2]

        fig = plt.figure(figsize=(14, 4.5))
        gs  = gridspec.GridSpec(1, 4, width_ratios=[1, 1, 1, 0.05],
                                wspace=0.35, left=0.07, right=0.93)

        ax_xy = fig.add_subplot(gs[0])
        ax_xz = fig.add_subplot(gs[1])
        ax_yz = fig.add_subplot(gs[2])
        cax   = fig.add_subplot(gs[3])

        im = _density_heatmap(ax_xy, xs, ys, "X", "Y", "XY Plane")
        _density_heatmap(ax_xz, xs, zs, "X", "Z", "XZ Plane")
        _density_heatmap(ax_yz, ys, zs, "Y", "Z", "YZ Plane")

        fig.colorbar(im, cax=cax, label="Waypoint count (log scale)")
        fig.suptitle("Approximate Workspace Coverage — Trajectory Waypoints\n"
                     f"({len(self.waypoint_xyz):,} points, "
                     f"{len(self.endpoint_radii)//2:,} trajectories)",
                     fontsize=11, fontweight="bold", y=1.02)
        path2 = out / f"{prefix}_workspace_heatmaps.png"
        fig.savefig(path2, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved → {path2}")

    # ------------------------------------------------------------------
    def print_summary(self):
        """Print descriptive statistics to stdout."""
        if self.endpoint_radii is None:
            raise RuntimeError("Call generate() first.")

        r = self.endpoint_radii
        w = self.waypoint_xyz
        print("\n── Approximate Dataset Summary ─────────────────────────")
        print(f"  Endpoints sampled   : {len(r):>10,}")
        print(f"  Waypoints generated : {len(w):>10,}")
        print(f"  Max reach           : {self.max_reach:>10.4f}")
        print(f"  Radius  mean        : {r.mean():>10.4f}")
        print(f"  Radius  std         : {r.std():>10.4f}")
        print(f"  Radius  min / max   : {r.min():>8.4f} / {r.max():.4f}")
        for i, axis in enumerate("XYZ"):
            col = w[:, i]
            print(f"  {axis} range             : "
                  f"[{col.min():8.3f}, {col.max():8.3f}]  "
                  f"mean={col.mean():.3f}")
        print("─────────────────────────────────────────────────────────\n")

    def run(self, target_waypoints: int = 200_000,
            out_dir: str = ".", prefix: str = "approx"):
        self.generate(target_waypoints=target_waypoints)
        self.print_summary()
        self.plot(out_dir=out_dir, prefix=prefix)


# ===========================================================================
# 2.  RealDatasetAnalyzer
# ===========================================================================
class RealDatasetAnalyzer:
    """
    Load a pickled dataset and compute coverage + quality statistics.

    Frame format (as saved by my_dataframe):
        ((delta_q_N, q_N, dist_N, rot_6D), delta_q_out, target_weight)

    where
        q_N  = q / (high_bounds - low_bounds)   ← normalised joint angles

    Parameters
    ----------
    robot    : Robot_math — needed for FK and denormalisation.
    run_fk   : if True, denormalise q_N → q and run FK to get EE XYZ
               positions for the heatmaps.  Takes a few minutes for 200k
               frames but gives the true workspace coverage picture.
               If False, only the target-weight histogram is produced.
    """

    def __init__(self, robot: Robot_math, run_fk: bool = True):
        self.robot  = robot
        self.device = robot.device
        self.run_fk = run_fk

        self.dataset: deque | None = None

        # Results
        self.target_weights: np.ndarray | None = None
        self.ee_xyz:         np.ndarray | None = None

    # ------------------------------------------------------------------
    def load(self, filename: str):
        """Load dataset from *filename*.pkl."""
        path = Path(filename).with_suffix(".pkl")
        print(f"[Real] Loading {path} ...")
        with open(path, "rb") as f:
            self.dataset = pickle.load(f)
        print(f"[Real] Loaded {len(self.dataset):,} frames.")

    # ------------------------------------------------------------------
    def _denorm_q(self, q_N: torch.Tensor) -> torch.Tensor:
        """Invert get_normal_joint_value: q = q_N * (high - low)."""
        return q_N.to(self.device) * (self.robot.high_bounds - self.robot.low_bounds)

    # ------------------------------------------------------------------
    def analyse(self, report_every: int = 10_000):
        """Extract target weights and (optionally) run FK on every frame."""
        if self.dataset is None:
            raise RuntimeError("Call load() first.")

        n = len(self.dataset)
        print(f"[Real] Analysing {n:,} frames  (run_fk={self.run_fk}) ...")
        t0 = time.time()

        weights = np.empty(n, dtype=np.float32)
        xyz_rows = [] if self.run_fk else None

        for i, frame in enumerate(self.dataset):
            (delta_q_N, q_N, dist_N, rot_6D), delta_q_out, tw = frame

            weights[i] = float(tw)

            if self.run_fk:
                q_N_t = torch.tensor(q_N, dtype=torch.float64) \
                    if not isinstance(q_N, torch.Tensor) else q_N
                q = self._denorm_q(q_N_t)
                self.robot.q_vect = q
                ee = self.robot.give_ds()[-1].cpu().numpy()
                xyz_rows.append(ee)

            if (i + 1) % report_every == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                eta  = (n - i - 1) / rate
                print(f"  {i+1:>8,} / {n:,}  |  "
                      f"{elapsed:.0f}s elapsed  |  ETA {eta:.0f}s",
                      end="\r", flush=True)

        print()

        self.target_weights = weights

        if self.run_fk and xyz_rows:
            self.ee_xyz = np.vstack(xyz_rows)

        elapsed = time.time() - t0
        print(f"[Real] Analysis complete in {elapsed:.1f}s")

    # ------------------------------------------------------------------
    def plot(self, out_dir: str = ".", prefix: str = "real"):
        if self.target_weights is None:
            raise RuntimeError("Call analyse() first.")

        _apply_style()
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)

        # ── Figure 1: Target-weight histogram ─────────────────────────
        # The three canonical weights and their meaning
        weight_map = {
            1.0:  ("Good",  "#2ca02c"),
            0.3:  ("Okay",  "#ff7f0e"),
            0.05: ("Bad",   "#d62728"),
        }
        # Bin edges centred on each weight value
        boundaries = [0.0, 0.15, 0.65, 1.05]
        labels  = ["Bad (0.05)", "Okay (0.3)", "Good (1.0)"]
        colours = ["#d62728",    "#ff7f0e",    "#2ca02c"]

        counts, _ = np.histogram(self.target_weights, bins=boundaries)
        total      = counts.sum()
        pcts       = 100.0 * counts / total

        fig, ax = plt.subplots(figsize=(7, 4.5))
        bars = ax.bar(labels, counts, color=colours,
                      edgecolor="white", linewidth=0.6, width=0.55)

        for bar, pct, cnt in zip(bars, pcts, counts):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + total * 0.005,
                    f"{pct:.1f}%\n({cnt:,})",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")

        ax.set_ylabel("Frame count")
        ax.set_title("Dataset Quality Distribution\n"
                     "(target_weight used to scale training loss)")
        ax.set_ylim(0, counts.max() * 1.18)
        ax.grid(True, axis="y")
        fig.tight_layout()
        path1 = out / f"{prefix}_quality_histogram.png"
        fig.savefig(path1, dpi=150)
        plt.close(fig)
        print(f"  Saved → {path1}")

        # ── Figure 2: EE workspace heatmaps (only if FK was run) ──────
        if self.ee_xyz is not None:
            xs = self.ee_xyz[:, 0]
            ys = self.ee_xyz[:, 1]
            zs = self.ee_xyz[:, 2]

            fig = plt.figure(figsize=(14, 4.5))
            gs  = gridspec.GridSpec(1, 4, width_ratios=[1, 1, 1, 0.05],
                                    wspace=0.35, left=0.07, right=0.93)

            ax_xy = fig.add_subplot(gs[0])
            ax_xz = fig.add_subplot(gs[1])
            ax_yz = fig.add_subplot(gs[2])
            cax   = fig.add_subplot(gs[3])

            im = _density_heatmap(ax_xy, xs, ys, "X", "Y", "XY Plane")
            _density_heatmap(ax_xz, xs, zs, "X", "Z", "XZ Plane")
            _density_heatmap(ax_yz, ys, zs, "Y", "Z", "YZ Plane")

            fig.colorbar(im, cax=cax, label="Frame count (log scale)")
            fig.suptitle("Real Dataset — End-Effector Workspace Coverage\n"
                         f"({len(self.ee_xyz):,} frames via FK)",
                         fontsize=11, fontweight="bold", y=1.02)
            path2 = out / f"{prefix}_workspace_heatmaps.png"
            fig.savefig(path2, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved → {path2}")

    # ------------------------------------------------------------------
    def print_summary(self):
        if self.target_weights is None:
            raise RuntimeError("Call analyse() first.")

        w      = self.target_weights
        total  = len(w)
        n_good = (w == 1.0).sum()
        n_okay = (w == 0.3).sum()
        n_bad  = (w == 0.05).sum()

        print("\n── Real Dataset Summary ─────────────────────────────────")
        print(f"  Total frames        : {total:>10,}")
        print(f"  Good  (w=1.00)      : {n_good:>10,}  ({100*n_good/total:5.1f}%)")
        print(f"  Okay  (w=0.30)      : {n_okay:>10,}  ({100*n_okay/total:5.1f}%)")
        print(f"  Bad   (w=0.05)      : {n_bad:>10,}  ({100*n_bad/total:5.1f}%)")

        if self.ee_xyz is not None:
            for i, axis in enumerate("XYZ"):
                col = self.ee_xyz[:, i]
                print(f"  EE {axis} range         : "
                      f"[{col.min():8.3f}, {col.max():8.3f}]  "
                      f"mean={col.mean():.3f}")
        print("─────────────────────────────────────────────────────────\n")

    # ------------------------------------------------------------------
    def run(self, filename: str, out_dir: str = ".", prefix: str = "real"):
        self.load(filename)
        self.analyse()
        self.print_summary()
        self.plot(out_dir=out_dir, prefix=prefix)




# ===========================================================================
# 3.  SimpleIKDatasetAnalyzer
# ===========================================================================
class SimpleIKDatasetAnalyzer:
    """
    Analyses the simple IK dataset produced by make_simple_ik_dataset.py.

    Each frame is a (xyz (3,), SO3 (3,3)) tuple of raw end-effector
    coordinates stored from a random joint configuration.

    Plots produced:
      1. XY / XZ / YZ density heatmaps  — EE position coverage
      2. Per-axis position histograms    — distribution shape per axis
      3. Orientation spread              — histogram of ||R - I||_F per frame
                                           (proxy for how varied orientations are)
    """

    def __init__(self, robot: Robot_math):
        self.robot   = robot
        self.dataset = None          # list of (xyz, SO3) tuples after load()
        self.xyz_arr : np.ndarray | None = None   # (N, 3)
        self.ori_frob: np.ndarray | None = None   # (N,)  Frobenius norm vs identity

    # ------------------------------------------------------------------
    def load(self, filename: str):
        path = f"{filename}.pkl"
        with open(path, "rb") as f:
            raw = pickle.load(f)
        # deque or list — normalise to list
        self.dataset = list(raw)
        print(f"[SimpleIK] Loaded {len(self.dataset):,} frames ← {path}")

    # ------------------------------------------------------------------
    def analyse(self, report_every: int = 50_000):
        if self.dataset is None:
            raise RuntimeError("Call load() first.")

        n = len(self.dataset)
        print(f"[SimpleIK] Analysing {n:,} frames ...")
        t0 = time.time()

        xyz_rows  = np.empty((n, 3), dtype=np.float32)
        ori_frob  = np.empty(n,      dtype=np.float32)
        I3        = np.eye(3, dtype=np.float32)

        for i, (xyz, SO3) in enumerate(self.dataset):
            xyz_np = xyz.detach().cpu().numpy().astype(np.float32)                 if isinstance(xyz, torch.Tensor) else np.asarray(xyz, np.float32)
            SO3_np = SO3.detach().cpu().numpy().astype(np.float32)                 if isinstance(SO3, torch.Tensor) else np.asarray(SO3, np.float32)

            xyz_rows[i] = xyz_np
            ori_frob[i] = np.linalg.norm(SO3_np - I3, "fro")

            if (i + 1) % report_every == 0:
                elapsed = time.time() - t0
                eta     = (n - i - 1) / max((i + 1) / elapsed, 1e-9)
                print(f"  {i+1:>8,} / {n:,}  |  {elapsed:.0f}s  |  ETA {eta:.0f}s",
                      end="\r", flush=True)

        print()
        self.xyz_arr  = xyz_rows
        self.ori_frob = ori_frob
        print(f"[SimpleIK] Done in {time.time() - t0:.1f}s")

    # ------------------------------------------------------------------
    def plot(self, out_dir: str = ".", prefix: str = "simple_ik"):
        if self.xyz_arr is None:
            raise RuntimeError("Call analyse() first.")

        _apply_style()
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        xs, ys, zs = self.xyz_arr[:, 0], self.xyz_arr[:, 1], self.xyz_arr[:, 2]

        # ── Figure 1: workspace heatmaps ─────────────────────────────
        fig = plt.figure(figsize=(14, 4.5))
        gs  = gridspec.GridSpec(1, 4, width_ratios=[1, 1, 1, 0.05],
                                wspace=0.35, left=0.07, right=0.93)
        ax_xy = fig.add_subplot(gs[0])
        ax_xz = fig.add_subplot(gs[1])
        ax_yz = fig.add_subplot(gs[2])
        cax   = fig.add_subplot(gs[3])

        im = _density_heatmap(ax_xy, xs, ys, "X", "Y", "XY Plane")
        _density_heatmap(ax_xz, xs, zs, "X", "Z", "XZ Plane")
        _density_heatmap(ax_yz, ys, zs, "Y", "Z", "YZ Plane")
        fig.colorbar(im, cax=cax, label="Frame count (log scale)")
        fig.suptitle(f"Simple IK Dataset — EE Workspace Coverage\n"
                     f"({len(self.xyz_arr):,} frames)",
                     fontsize=11, fontweight="bold", y=1.02)
        p1 = out / f"{prefix}_workspace_heatmaps.png"
        fig.savefig(p1, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved → {p1}")

        # ── Figure 2: per-axis histograms ─────────────────────────────
        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        for ax, data, label, color in zip(
                axes, [xs, ys, zs], ["X", "Y", "Z"],
                ["#1f77b4", "#ff7f0e", "#2ca02c"]):
            ax.hist(data, bins=80, color=color, edgecolor="none", alpha=0.85)
            ax.set_xlabel(f"{label} position")
            ax.set_ylabel("Count")
            ax.set_title(f"{label}-axis Distribution  "
                         f"[{data.min():.1f}, {data.max():.1f}]")
            ax.grid(True, axis="y")
            # Annotate mean ± std
            ax.axvline(data.mean(), color="black", linewidth=1.2,
                       linestyle="--", label=f"mean={data.mean():.2f}")
            ax.axvline(data.mean() - data.std(), color="grey",
                       linewidth=0.8, linestyle=":")
            ax.axvline(data.mean() + data.std(), color="grey",
                       linewidth=0.8, linestyle=":", label=f"±σ={data.std():.2f}")
            ax.legend(fontsize=7)
        fig.suptitle("Simple IK Dataset — Per-Axis EE Position Distribution",
                     fontsize=11, fontweight="bold")
        fig.tight_layout()
        p2 = out / f"{prefix}_axis_histograms.png"
        fig.savefig(p2, dpi=150)
        plt.close(fig)
        print(f"  Saved → {p2}")

        # ── Figure 3: orientation spread ─────────────────────────────
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(self.ori_frob, bins=80, color="#9467bd",
                edgecolor="none", alpha=0.85)
        ax.set_xlabel("||R − I||_F  (Frobenius norm vs identity)")
        ax.set_ylabel("Count")
        ax.set_title("Simple IK Dataset — Orientation Spread\n"
                     "(0 = identity, larger = more rotated)")
        ax.axvline(self.ori_frob.mean(), color="black", linewidth=1.2,
                   linestyle="--", label=f"mean={self.ori_frob.mean():.3f}")
        ax.legend()
        ax.grid(True, axis="y")
        fig.tight_layout()
        p3 = out / f"{prefix}_orientation_spread.png"
        fig.savefig(p3, dpi=150)
        plt.close(fig)
        print(f"  Saved → {p3}")

    # ------------------------------------------------------------------
    def print_summary(self):
        if self.xyz_arr is None:
            raise RuntimeError("Call analyse() first.")
        print("\n── Simple IK Dataset Summary ────────────────────────────")
        print(f"  Total frames : {len(self.xyz_arr):>10,}")
        mr = float(self.robot.max_reach)
        for i, axis in enumerate("XYZ"):
            col = self.xyz_arr[:, i]
            print(f"  EE {axis}         : [{col.min():8.3f}, {col.max():8.3f}]"
                  f"  mean={col.mean():.3f}  σ={col.std():.3f}")
        radii = np.linalg.norm(self.xyz_arr, axis=1)
        print(f"  Radius       : [{radii.min():.3f}, {radii.max():.3f}]"
              f"  mean={radii.mean():.3f}  (max_reach={mr:.3f})")
        print(f"  Ori spread   : mean ||R-I||_F = {self.ori_frob.mean():.4f}"
              f"  σ={self.ori_frob.std():.4f}")
        pct_inner = 100.0 * (radii < mr * 0.5).mean()
        pct_outer = 100.0 * (radii > mr * 0.75).mean()
        print(f"  Inner 50%r   : {pct_inner:.1f}% of frames")
        print(f"  Outer 75%r   : {pct_outer:.1f}% of frames")
        print("─────────────────────────────────────────────────────────\n")

    # ------------------------------------------------------------------
    def run(self, filename: str, out_dir: str = ".", prefix: str = "simple_ik"):
        self.load(filename)
        self.analyse()
        self.print_summary()
        self.plot(out_dir=out_dir, prefix=prefix)


# ===========================================================================
# 4.  ChessboardDatasetAnalyzer
# ===========================================================================
class ChessboardDatasetAnalyzer:
    """
    Analyses the chessboard test dataset produced by test_data_maker.py.

    Structure: list of move_dicts, each with:
        move_dict['legs'][leg_idx][solver_name] = [frame_dict, ...]
        frame_dict = {q, xyz, SO3, quality, e_pos, e_ori}

    Plots produced per solver:
      1. Quality bar chart          — good / okay / bad counts per leg + total
      2. e_pos + e_ori violin plots — error distribution per leg
      3. EE trajectory scatter      — XY and XZ views, colour-coded by quality
    Plus a cross-solver comparison chart.
    """

    SOLVERS  = ("oracle", "simple_ik", "imit_ik")
    SOLVER_COLORS = {
        "oracle"    : "#1f77b4",
        "simple_ik" : "#ff7f0e",
        "imit_ik"   : "#2ca02c",
    }
    QUALITY_COLORS = {"good": "#2ca02c", "okay": "#ff7f0e", "bad": "#d62728"}
    LEG_NAMES = ["Leg 0\nHome→A", "Leg 1\nA→B", "Leg 2\nB→C", "Leg 3\nC→D"]

    def __init__(self, robot: Robot_math):
        self.robot   = robot
        self.dataset = None    # list of move_dicts after load()

        # populated by analyse()
        # stats[solver][leg_idx] = {'good': n, 'okay': n, 'bad': n,
        #                            'e_pos': [...], 'e_ori': [...],
        #                            'xyz': ndarray (N,3)}
        self.stats: dict | None = None

    # ------------------------------------------------------------------
    def load(self, filename: str):
        path = f"{filename}.pkl"
        with open(path, "rb") as f:
            self.dataset = pickle.load(f)
        print(f"[Chess] Loaded {len(self.dataset):,} moves ← {path}")

    # ------------------------------------------------------------------
    def analyse(self):
        if self.dataset is None:
            raise RuntimeError("Call load() first.")

        n_moves = len(self.dataset)
        n_legs  = 4

        # Initialise accumulators
        stats = {}
        for s in self.SOLVERS:
            stats[s] = []
            for _ in range(n_legs):
                stats[s].append({
                    "good": 0, "okay": 0, "bad": 0,
                    "e_pos": [], "e_ori": [], "xyz": [],
                })

        for move in self.dataset:
            for leg_idx in range(n_legs):
                leg = move["legs"][leg_idx]
                for solver in self.SOLVERS:
                    frames = leg.get(solver, [])
                    acc    = stats[solver][leg_idx]
                    for f in frames:
                        acc[f["quality"]] += 1
                        acc["e_pos"].append(f["e_pos"])
                        acc["e_ori"].append(f["e_ori"])
                        xyz = f["xyz"]
                        xyz_np = xyz.detach().cpu().numpy()                             if isinstance(xyz, torch.Tensor) else np.asarray(xyz)
                        acc["xyz"].append(xyz_np.astype(np.float32))

        # Convert lists to arrays
        for s in self.SOLVERS:
            for leg_idx in range(n_legs):
                acc = stats[s][leg_idx]
                acc["e_pos"] = np.array(acc["e_pos"], dtype=np.float32)
                acc["e_ori"] = np.array(acc["e_ori"], dtype=np.float32)
                acc["xyz"]   = np.array(acc["xyz"],   dtype=np.float32)                     if acc["xyz"] else np.empty((0, 3), dtype=np.float32)

        self.stats = stats
        total = sum(
            stats["oracle"][l]["good"] + stats["oracle"][l]["okay"] + stats["oracle"][l]["bad"]
            for l in range(n_legs))
        print(f"[Chess] Analysis complete — {n_moves} moves, {total:,} total oracle frames")

    # ------------------------------------------------------------------
    def _quality_counts(self, solver, leg_idx):
        acc = self.stats[solver][leg_idx]
        return acc["good"], acc["okay"], acc["bad"]

    # ------------------------------------------------------------------
    def plot(self, out_dir: str = ".", prefix: str = "chess"):
        if self.stats is None:
            raise RuntimeError("Call analyse() first.")

        _apply_style()
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)

        # ── Figure 1: quality bar chart per solver ────────────────────
        fig, axes = plt.subplots(1, len(self.SOLVERS),
                                 figsize=(5 * len(self.SOLVERS), 5),
                                 sharey=False)
        for ax, solver in zip(axes, self.SOLVERS):
            goods = [self.stats[solver][l]["good"] for l in range(4)]
            okays = [self.stats[solver][l]["okay"] for l in range(4)]
            bads  = [self.stats[solver][l]["bad"]  for l in range(4)]
            x     = np.arange(4)
            w     = 0.25
            ax.bar(x - w, goods, w, label="Good", color=self.QUALITY_COLORS["good"])
            ax.bar(x,     okays, w, label="Okay", color=self.QUALITY_COLORS["okay"])
            ax.bar(x + w, bads,  w, label="Bad",  color=self.QUALITY_COLORS["bad"])
            ax.set_xticks(x)
            ax.set_xticklabels(self.LEG_NAMES, fontsize=8)
            ax.set_title(solver.replace("_", " ").title())
            ax.set_ylabel("Frame count")
            ax.legend(fontsize=7)
            ax.grid(True, axis="y")
            # Annotate % good on top of each good bar
            total_per_leg = [g + o + b for g, o, b in zip(goods, okays, bads)]
            for xi, (g, t) in enumerate(zip(goods, total_per_leg)):
                if t > 0:
                    ax.text(xi - w, g + t * 0.01,
                            f"{100*g/t:.0f}%", ha="center",
                            va="bottom", fontsize=7, fontweight="bold")
        fig.suptitle("Chessboard Dataset — Frame Quality by Solver and Leg",
                     fontsize=12, fontweight="bold")
        fig.tight_layout()
        p1 = out / f"{prefix}_quality_bars.png"
        fig.savefig(p1, dpi=150)
        plt.close(fig)
        print(f"  Saved → {p1}")

        # ── Figure 2: e_pos violin per leg, all solvers overlaid ──────
        for err_key, err_label in [("e_pos", "Position error (e_pos)"),
                                   ("e_ori", "Orientation error (e_ori)")]:
            fig, axes = plt.subplots(1, 4, figsize=(14, 5), sharey=True)
            for leg_idx, ax in enumerate(axes):
                data    = []
                labels  = []
                colours = []
                for solver in self.SOLVERS:
                    arr = self.stats[solver][leg_idx][err_key]
                    if len(arr) > 0:
                        data.append(arr)
                        labels.append(solver.replace("_", "\n"))
                        colours.append(self.SOLVER_COLORS[solver])
                if data:
                    parts = ax.violinplot(data, showmedians=True, showextrema=False)
                    for pc, col in zip(parts["bodies"], colours):
                        pc.set_facecolor(col)
                        pc.set_alpha(0.7)
                    parts["cmedians"].set_color("black")
                    parts["cmedians"].set_linewidth(1.5)
                ax.set_xticks(range(1, len(labels) + 1))
                ax.set_xticklabels(labels, fontsize=8)
                ax.set_title(self.LEG_NAMES[leg_idx])
                ax.set_ylabel(err_label if leg_idx == 0 else "")
                ax.grid(True, axis="y")
            fig.suptitle(f"Chessboard Dataset — {err_label} Distribution by Leg",
                         fontsize=11, fontweight="bold")
            fig.tight_layout()
            p = out / f"{prefix}_{err_key}_violins.png"
            fig.savefig(p, dpi=150)
            plt.close(fig)
            print(f"  Saved → {p}")

        # ── Figure 3: EE trajectory scatter XY + XZ, per solver ──────
        fig, axes = plt.subplots(len(self.SOLVERS), 2,
                                 figsize=(12, 4 * len(self.SOLVERS)))
        for row, solver in enumerate(self.SOLVERS):
            for col, (dim_a, dim_b, xlabel, ylabel) in enumerate([
                    (0, 1, "X", "Y"), (0, 2, "X", "Z")]):
                ax = axes[row, col]
                for leg_idx in range(4):
                    xyz = self.stats[solver][leg_idx]["xyz"]
                    acc = self.stats[solver][leg_idx]
                    n   = len(acc["e_pos"])
                    if n == 0:
                        continue
                    # Colour each point by quality
                    good_n = acc["good"]
                    okay_n = acc["okay"]
                    bad_n  = acc["bad"]
                    # Reconstruct per-frame quality colour array:
                    # frames were appended in order — use cumulative counts
                    # (we don't store per-frame quality separately,
                    #  so approximate: plot in three slabs)
                    # Better: re-derive from e_pos threshold
                    # We use e_pos + e_ori vs thresholds stored on robot's Loss_Math
                    # but we don't have those here, so colour by sorted error quartiles.
                    err_total = acc["e_pos"] + acc["e_ori"]
                    q25, q75  = np.percentile(err_total, [25, 75])
                    colors    = np.where(err_total < q25, self.QUALITY_COLORS["good"],
                               np.where(err_total < q75, self.QUALITY_COLORS["okay"],
                                                         self.QUALITY_COLORS["bad"]))
                    if xyz.shape[0] > 0:
                        ax.scatter(xyz[:, dim_a], xyz[:, dim_b],
                                   c=colors, s=2, alpha=0.4,
                                   label=f"Leg {leg_idx}" if col == 0 else None)
                ax.set_xlabel(xlabel)
                ax.set_ylabel(ylabel)
                ax.set_title(f"{solver.replace('_', ' ').title()} — {xlabel}{ylabel}")
                ax.grid(True)
                if col == 0:
                    ax.legend(fontsize=7, markerscale=4)
        fig.suptitle("Chessboard Dataset — EE Trajectories (colour = error quartile)",
                     fontsize=11, fontweight="bold")
        fig.tight_layout()
        p3 = out / f"{prefix}_ee_trajectories.png"
        fig.savefig(p3, dpi=150)
        plt.close(fig)
        print(f"  Saved → {p3}")

        # ── Figure 4: cross-solver % good comparison (summary bar) ───
        fig, ax = plt.subplots(figsize=(9, 5))
        x       = np.arange(4)
        w       = 0.25
        offsets = [-w, 0, w]
        for offset, solver in zip(offsets, self.SOLVERS):
            pct_good = []
            for leg_idx in range(4):
                acc = self.stats[solver][leg_idx]
                total = acc["good"] + acc["okay"] + acc["bad"]
                pct_good.append(100.0 * acc["good"] / total if total > 0 else 0.0)
            bars = ax.bar(x + offset, pct_good, w,
                          label=solver.replace("_", " ").title(),
                          color=self.SOLVER_COLORS[solver],
                          alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(self.LEG_NAMES)
        ax.set_ylabel("% frames graded GOOD")
        ax.set_ylim(0, 110)
        ax.axhline(100, color="grey", linewidth=0.6, linestyle="--")
        ax.set_title("Solver Comparison — % Good Frames per Leg",
                     fontsize=11, fontweight="bold")
        ax.legend()
        ax.grid(True, axis="y")
        fig.tight_layout()
        p4 = out / f"{prefix}_solver_comparison.png"
        fig.savefig(p4, dpi=150)
        plt.close(fig)
        print(f"  Saved → {p4}")

    # ------------------------------------------------------------------
    def print_summary(self):
        if self.stats is None:
            raise RuntimeError("Call analyse() first.")

        print("\n── Chessboard Dataset Summary ───────────────────────────")
        print(f"  Moves loaded : {len(self.dataset):,}")
        for solver in self.SOLVERS:
            print(f"\n  {solver.replace('_', ' ').upper()}")
            total_g = total_o = total_b = 0
            for leg_idx in range(4):
                acc = self.stats[solver][leg_idx]
                g, o, b = acc["good"], acc["okay"], acc["bad"]
                t = g + o + b
                total_g += g; total_o += o; total_b += b
                pct = 100*g/t if t > 0 else 0
                e_p = acc["e_pos"]
                e_o = acc["e_ori"]
                print(f"    Leg {leg_idx}: good {g:>5,}/{t:,} ({pct:.0f}%)  "
                      f"e_pos μ={e_p.mean():.4f} σ={e_p.std():.4f}  "
                      f"e_ori μ={e_o.mean():.4f} σ={e_o.std():.4f}")
            tt = total_g + total_o + total_b
            print(f"    TOTAL: good {total_g:,}/{tt:,} "
                  f"({100*total_g/tt:.1f}%)  "
                  f"okay {total_o:,}  bad {total_b:,}")
        print("─────────────────────────────────────────────────────────\n")

    # ------------------------------------------------------------------
    def run(self, filename: str, out_dir: str = ".", prefix: str = "chess"):
        self.load(filename)
        self.analyse()
        self.print_summary()
        self.plot(out_dir=out_dir, prefix=prefix)

# ===========================================================================
# Entry point
# ===========================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataset analysis tools")
    parser.add_argument("--mode",
                        choices=["approx", "real", "both", "simple_ik", "chess"],
                        default="both",
                        help="Which analyser to run  "
                             "(approx | real | both | simple_ik | chess)")
    parser.add_argument("--load",    type=str, default="training_data",
                        help="Dataset filename (no .pkl extension).  "
                             "For --mode both/real/approx this is the imitation "
                             "learning dataset; for simple_ik the simple IK "
                             "dataset; for chess the chessboard test dataset.")
    parser.add_argument("--frames",  type=int, default=200_000,
                        help="Target waypoints for approximate analysis "
                             "(default: 200000)")
    parser.add_argument("--no-fk",   action="store_true",
                        help="Skip FK in real analysis (only quality histogram)")
    parser.add_argument("--out",     type=str, default=".",
                        help="Output directory for figures (default: .)")
    args = parser.parse_args()

    robot = make_robot(torch.device("cpu"))

    if args.mode in ("approx", "both"):
        approx = ApproximateDatasetAnalyzer(robot=robot)
        approx.run(target_waypoints=args.frames,
                   out_dir=args.out, prefix="approx")

    if args.mode in ("real", "both"):
        real = RealDatasetAnalyzer(robot=robot, run_fk=not args.no_fk)
        real.run(filename=args.load,
                 out_dir=args.out, prefix="real")

    if args.mode == "simple_ik":
        sik = SimpleIKDatasetAnalyzer(robot=robot)
        sik.run(filename=args.load,
                out_dir=args.out, prefix="simple_ik")

    if args.mode == "chess":
        chess = ChessboardDatasetAnalyzer(robot=robot)
        chess.run(filename=args.load,
                  out_dir=args.out, prefix="chess")
        


# ── Usage examples ────────────────────────────────────────────────────────
# Imitation learning dataset (quality histogram only, fast):
#   python dataset_analysis.py --mode real --load training_data --no-fk
#
# Imitation learning dataset (full with FK heatmaps):
#   python dataset_analysis.py --mode real --load training_data
#
# Approximate workspace coverage (no dataset file needed):
#   python dataset_analysis.py --mode approx --frames 200000
#
# Both imitation analyses:
#   python dataset_analysis.py --mode both --load training_data --out figs/
#
# Simple IK dataset:
#   python dataset_analysis.py --mode simple_ik --load simple_ik_training_data
#
# Chessboard test dataset:
#   python dataset_analysis.py --mode chess --load chessboard_test_dataset --out figs/