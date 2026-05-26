"""
HER Replay Buffer for the TD3 Robot Arm Agent — optimised for GPU training.

Key design changes vs previous version
---------------------------------------
1. Pre-allocated tensor storage (GPU-resident)
   Transitions are stored as columns in large pre-allocated float32 tensors
   on the training device rather than as a deque of namedtuples.
   This eliminates list(deque) conversion and torch.stack on every sample.

2. O(1) sampling via index array
   random.sample on a deque is O(n).  We maintain a flat index array and
   sample indices directly, then gather from the pre-allocated tensors.

3. CPU->GPU transfer eliminated
   All stored tensors live on the target device.  Batches are sliced
   directly — zero copy, zero transfer overhead at sample time.

4. FK-free reward recomputation in finish_episode
   raw transitions already carry pos_next / R_next from the environment
   step, so reward() no longer needs to re-run FK internally.  We call
   the reward terms directly using the cached poses.

5. float32 storage
   Stored as float32 (matching network dtype) to halve VRAM and improve
   memory bandwidth during sampling.  FK / reward math still uses float64
   at episode commit time, cast once before storage.

Buffer layout (all on self.device, dtype=torch.float32)
  _s       : (capacity, state_dim)
  _a       : (capacity, action_dim)
  _r       : (capacity,)
  _s_prime : (capacity, state_dim)
  _done    : (capacity,)

capacity is buffer_size.  A write pointer wraps around modulo capacity
(ring buffer), matching the old deque(maxlen=...) semantics.
"""

from __future__ import annotations
import random
import torch

from rewards_math import Reward_Math


class Replay_Buffer:
    """
    GPU-resident replay buffer with HER (FUTURE strategy) and decaying ratio.

    Parameters
    ----------
    buffer_size      : maximum transitions stored  (ring buffer)
    reward_math      : Reward_Math instance
    device           : torch.device for stored tensors (should match TD3 device)
    state_dim        : dimension of state vector
    action_dim       : number of joints
    her_ratio_start  : HER probability at episode 0           (default 0.80)
    her_ratio_end    : HER probability floor                   (default 0.20)
    her_decay_steps  : episodes over which ratio decays        (default 1000)
    """

    def __init__(self,
                 buffer_size:     int,
                 reward_math:     Reward_Math,
                 device:          torch.device,
                 state_dim:       int,
                 action_dim:      int,
                 her_ratio_start: float = 0.80,
                 her_ratio_end:   float = 0.20,
                 her_decay_steps: int   = 1000):

        self.capacity        = buffer_size
        self.reward_math     = reward_math
        self.device          = device
        self.state_dim       = state_dim
        self.action_dim      = action_dim
        self.her_ratio_start = her_ratio_start
        self.her_ratio_end   = her_ratio_end
        self.her_decay_steps = her_decay_steps

        self._ptr            = 0      # next write position
        self._size           = 0      # current number of valid entries
        self._episodes_done  = 0

        # Pre-allocated GPU tensors
        self._s       = torch.zeros(buffer_size, state_dim,  dtype=torch.float32, device=device)
        self._a       = torch.zeros(buffer_size, action_dim, dtype=torch.float32, device=device)
        self._r       = torch.zeros(buffer_size,             dtype=torch.float32, device=device)
        self._s_prime = torch.zeros(buffer_size, state_dim,  dtype=torch.float32, device=device)
        self._done    = torch.zeros(buffer_size,             dtype=torch.float32, device=device)

        # Episode cache: list of raw step dicts (CPU float64, never touches GPU)
        self._episode_cache: list[dict] = []

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def her_ratio(self) -> float:
        t = min(self._episodes_done / max(self.her_decay_steps, 1), 1.0)
        return self.her_ratio_start + t * (self.her_ratio_end - self.her_ratio_start)

    # -------------------------------------------------------------------------
    # Episode-level API
    # -------------------------------------------------------------------------

    def add_step(self,
                 q_new:        torch.Tensor,
                 delta_q_new:  torch.Tensor,
                 delta_q_prev: torch.Tensor,
                 pos_curr:     torch.Tensor,
                 R_curr:       torch.Tensor,
                 pos_next:     torch.Tensor,
                 R_next:       torch.Tensor,
                 a_norm:       torch.Tensor,
                 done:         bool):
        """Cache one step.  Call finish_episode() at episode end."""
        self._episode_cache.append({
            "q_new":        self._cpu64(q_new),
            "delta_q_new":  self._cpu64(delta_q_new),
            "delta_q_prev": self._cpu64(delta_q_prev),
            "pos_curr":     self._cpu64(pos_curr),
            "R_curr":       self._cpu64(R_curr),
            "pos_next":     self._cpu64(pos_next),
            "R_next":       self._cpu64(R_next),
            "a_norm":       self._cpu64(a_norm),
            "done":         done,
        })

    def finish_episode(self,
                       pos_goal:   torch.Tensor,
                       R_goal_SO3: torch.Tensor,
                       use_focal:  bool = True):
        """
        Commit cached episode to the GPU buffer.

        For each step:
          1. Real transition  (actual goal, reward computed from cached poses).
          2. HER transition   (FUTURE goal, with probability her_ratio).

        Reward is computed WITHOUT re-running FK: pos_next / R_next are
        already in the cache, so we call reward terms directly.
        """
        episode  = self._episode_cache
        T        = len(episode)
        rm       = self.reward_math
        ratio    = self.her_ratio

        pos_goal = self._cpu64(pos_goal)
        R_goal   = self._cpu64(R_goal_SO3)

        for t, raw in enumerate(episode):

            # ── 1. Real transition ──────────────────────────────────────────
            s = rm.build_state(
                pos_curr     = raw["pos_curr"],
                pos_goal     = pos_goal,
                R_goal_SO3   = R_goal,
                delta_q_prev = raw["delta_q_prev"],
            )
            s_prime = rm.build_state(
                pos_curr     = raw["pos_next"],
                pos_goal     = pos_goal,
                R_goal_SO3   = R_goal,
                delta_q_prev = raw["delta_q_new"],
            )
            r_real = self._compute_reward_no_fk(
                raw=raw, pos_goal=pos_goal, R_goal=R_goal,
                use_focal=use_focal, rm=rm,
            )
            self._write(s, raw["a_norm"], r_real, s_prime, raw["done"])

            # ── 2. HER transition ───────────────────────────────────────────
            if random.random() < ratio and t < T - 1:
                fi           = random.randint(t + 1, T - 1)
                future       = episode[fi]
                her_pos_goal = future["pos_next"]
                her_R_goal   = future["R_next"]

                s_her = rm.build_state(
                    pos_curr     = raw["pos_curr"],
                    pos_goal     = her_pos_goal,
                    R_goal_SO3   = her_R_goal,
                    delta_q_prev = raw["delta_q_prev"],
                )
                s_prime_her = rm.build_state(
                    pos_curr     = raw["pos_next"],
                    pos_goal     = her_pos_goal,
                    R_goal_SO3   = her_R_goal,
                    delta_q_prev = raw["delta_q_new"],
                )
                r_her = self._compute_reward_no_fk(
                    raw=raw, pos_goal=her_pos_goal, R_goal=her_R_goal,
                    use_focal=use_focal, rm=rm,
                )
                her_done = (t == fi) or raw["done"]
                self._write(s_her, raw["a_norm"], r_her, s_prime_her, her_done)

        self._episode_cache = []
        self._episodes_done += 1

    # -------------------------------------------------------------------------
    # Sampling API
    # -------------------------------------------------------------------------

    def sample(self, batch_size: int) -> dict | None:
        """
        Sample a random mini-batch.  O(batch_size) — no deque conversion.

        Returns dict of float32 tensors on self.device:
            s, a, r, s_prime, done  —  shapes (B, state_dim), (B, action_dim),
                                        (B,), (B, state_dim), (B,)
        Returns None if buffer has fewer than batch_size transitions.
        """
        if self._size < batch_size:
            return None

        idx = torch.randint(0, self._size, (batch_size,), device=self.device)

        return {
            "s":       self._s      [idx],
            "a":       self._a      [idx],
            "r":       self._r      [idx],
            "s_prime": self._s_prime[idx],
            "done":    self._done   [idx],
        }

    def __len__(self) -> int:
        return self._size

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _write(self,
               s:       torch.Tensor,
               a:       torch.Tensor,
               r:       torch.Tensor | float,
               s_prime: torch.Tensor,
               done:    bool | float):
        """Write one transition into the ring buffer at the current pointer."""
        i = self._ptr

        self._s      [i] = s.to(dtype=torch.float32, device=self.device)
        self._a      [i] = a.to(dtype=torch.float32, device=self.device)
        self._r      [i] = float(r.item() if isinstance(r, torch.Tensor) else r)
        self._s_prime[i] = s_prime.to(dtype=torch.float32, device=self.device)
        self._done   [i] = float(done)

        self._ptr  = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def _compute_reward_no_fk(self,
                               raw:       dict,
                               pos_goal:  torch.Tensor,
                               R_goal:    torch.Tensor,
                               use_focal: bool,
                               rm:        Reward_Math) -> torch.Tensor:
        """
        Compute scalar reward from cached poses, skipping FK.

        Uses pos_next / R_next already in the raw transition dict instead
        of re-running forward kinematics inside rm.reward().
        """
        rm.reset_episode()

        rv_pos, e_pos_raw = rm.r_pos(raw["pos_next"], pos_goal)
        rv_ori, e_ori_raw = rm.r_ori(raw["R_next"],   R_goal,
                                      e_pos_raw if use_focal else None)
        rv_pass           = rm.r_pass(e_pos_raw, e_ori_raw)
        rv_vel            = rm.r_vel(raw["delta_q_new"])
        rv_acc            = rm.r_acc(raw["delta_q_new"], raw["delta_q_prev"])
        rv_crash          = rm.r_crash(raw["q_new"])
        rv_time           = torch.tensor(-1.0, dtype=torch.float64,
                                         device=rm.device)

        return rv_pass + rv_pos + rv_ori + rv_vel + rv_acc + rv_crash + rv_time

    @staticmethod
    def _cpu64(x) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            return x.detach().to(dtype=torch.float64, device="cpu")
        return torch.tensor(x, dtype=torch.float64)