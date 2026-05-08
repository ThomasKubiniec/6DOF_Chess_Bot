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
# Entry point
# ===========================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataset analysis tools")
    parser.add_argument("--mode",    choices=["approx", "real", "both"],
                        default="both",
                        help="Which analyser to run (default: both)")
    parser.add_argument("--load",    type=str, default="training_data",
                        help="Dataset filename to load for real analysis "
                             "(no .pkl extension)")
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
        


# # Just the quality histogram, no FK (fast):
# python dataset_analysis.py --mode real --load training_data --no-fk

# # Full real analysis with FK heatmaps:
# python dataset_analysis.py --mode real --load training_data

# # Approximate only (no dataset file needed):
# python dataset_analysis.py --mode approx --frames 200000

# # Both, save figures to a figs/ folder:
# python dataset_analysis.py --mode both --load training_data --out figs/