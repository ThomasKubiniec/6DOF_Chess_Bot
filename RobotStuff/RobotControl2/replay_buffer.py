"""
HER Replay Buffer — vectorised episode commit for the TD3 Robot Arm Agent.

Key design principles
---------------------
1. Pre-allocated GPU ring buffer (from previous version — unchanged).
   O(1) sample via torch.randint; zero CPU->GPU transfer at sample time.

2. Vectorised finish_episode  (NEW — the main speedup)
   The old implementation called build_state() and individual reward terms
   T times in a Python loop (T = episode length, typically 200).  With HER
   at ratio 0.8 that was ~360 Python-level tensor ops per episode commit.

   The new implementation:
     a) Stacks the T cached steps into episode tensors in one list-comp.
     b) Calls build_state_batch() once  ->  (T, S) states in one op.
     c) Calls reward_batch() once       ->  (T,) rewards in one op.
     d) For HER: samples future indices for all steps simultaneously,
        calls build_state_batch() twice more, reward_batch() once more.
     e) Writes all T (or 2T) transitions with a single vectorised
        _write_batch() call that fills contiguous ring-buffer slots.

   Python-loop overhead per episode: O(1) instead of O(T).

3. Episode cache stores raw numpy-like CPU tensors (unchanged interface
   for the training loop's add_step() calls).
"""

from __future__ import annotations
import random
import torch
from rewards_math import Reward_Math


class Replay_Buffer:
    """
    GPU-resident replay buffer with vectorised HER (FUTURE strategy).

    Parameters
    ----------
    buffer_size      : maximum transitions stored
    reward_math      : Reward_Math instance
    device           : torch.device for stored tensors
    state_dim        : dimension of the state vector
    action_dim       : number of joints
    her_ratio_start  : HER probability at episode 0      (default 0.80)
    her_ratio_end    : HER probability floor              (default 0.20)
    her_decay_steps  : episodes over which ratio decays  (default 1000)
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
        self.rm              = reward_math
        self.device          = device
        self.state_dim       = state_dim
        self.action_dim      = action_dim
        self.her_ratio_start = her_ratio_start
        self.her_ratio_end   = her_ratio_end
        self.her_decay_steps = her_decay_steps

        self._ptr            = 0
        self._size           = 0
        self._episodes_done  = 0

        # Pre-allocated GPU tensors (float32 — matches network dtype)
        self._s       = torch.zeros(buffer_size, state_dim,  dtype=torch.float32, device=device)
        self._a       = torch.zeros(buffer_size, action_dim, dtype=torch.float32, device=device)
        self._r       = torch.zeros(buffer_size,             dtype=torch.float32, device=device)
        self._s_prime = torch.zeros(buffer_size, state_dim,  dtype=torch.float32, device=device)
        self._done    = torch.zeros(buffer_size,             dtype=torch.float32, device=device)

        self._episode_cache: list[dict] = []

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def her_ratio(self) -> float:
        t = min(self._episodes_done / max(self.her_decay_steps, 1), 1.0)
        return self.her_ratio_start + t * (self.her_ratio_end - self.her_ratio_start)

    # -------------------------------------------------------------------------
    # Episode step cache  (called once per env step from train.py)
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
        """Cache one environment step.  Call finish_episode() at episode end."""
        self._episode_cache.append({
            "q_new":        _cpu64(q_new),
            "delta_q_new":  _cpu64(delta_q_new),
            "delta_q_prev": _cpu64(delta_q_prev),
            "pos_curr":     _cpu64(pos_curr),
            "R_curr":       _cpu64(R_curr),
            "pos_next":     _cpu64(pos_next),
            "R_next":       _cpu64(R_next),
            "a_norm":       _cpu64(a_norm),
            "done":         float(done),
        })

    # -------------------------------------------------------------------------
    # Vectorised episode commit
    # -------------------------------------------------------------------------

    def finish_episode(self,
                       pos_goal:   torch.Tensor,
                       R_goal_SO3: torch.Tensor,
                       use_focal:  bool = True):
        """
        Commit the cached episode to the GPU ring buffer.

        All T steps are processed in two batched calls (build_state_batch +
        reward_batch) rather than T individual Python calls.  HER transitions
        are similarly generated in one additional pair of batched calls.

        Parameters
        ----------
        pos_goal   : (3,)   actual goal EE position for this episode
        R_goal_SO3 : (3,3)  actual goal rotation matrix
        use_focal  : passed through to reward_batch focal coupling
        """
        cache = self._episode_cache
        T     = len(cache)
        if T == 0:
            self._episode_cache = []
            return

        rm    = self.rm
        ratio = self.her_ratio

        # ── Stack episode cache and move to rm.device ────────────────────
        # Cache entries are CPU float64. All reward_batch / build_state_batch
        # calls operate on rm.device (CUDA when training on GPU), so we cast
        # once here to avoid cross-device errors inside those calls.
        dev = rm.device
        pos_curr_T     = torch.stack([c["pos_curr"]     for c in cache]).to(dev)  # (T, 3)
        pos_next_T     = torch.stack([c["pos_next"]     for c in cache]).to(dev)  # (T, 3)
        R_curr_T       = torch.stack([c["R_curr"]       for c in cache]).to(dev)  # (T, 3, 3)
        R_next_T       = torch.stack([c["R_next"]       for c in cache]).to(dev)  # (T, 3, 3)
        delta_q_new_T  = torch.stack([c["delta_q_new"]  for c in cache]).to(dev)  # (T, n)
        delta_q_prev_T = torch.stack([c["delta_q_prev"] for c in cache]).to(dev)  # (T, n)
        a_norm_T       = torch.stack([c["a_norm"]       for c in cache]).to(dev)  # (T, n)
        done_T         = torch.tensor([c["done"] for c in cache],
                                      dtype=torch.float32, device=dev)            # (T,)
        pos_goal   = _cpu64(pos_goal).to(dev)
        R_goal_SO3 = _cpu64(R_goal_SO3).to(dev)

        # ── Real transitions ────────────────────────────────────────────────
        # State at t: uses pos_curr[t] and delta_q_prev[t]
        s_T = rm.build_state_batch(
            pos_curr_batch     = pos_curr_T,
            pos_goal           = pos_goal,
            R_goal_SO3         = R_goal_SO3,
            delta_q_prev_batch = delta_q_prev_T,
        )                                                    # (T, S)

        # State at t+1: uses pos_next[t] and delta_q_new[t]
        sp_T = rm.build_state_batch(
            pos_curr_batch     = pos_next_T,
            pos_goal           = pos_goal,
            R_goal_SO3         = R_goal_SO3,
            delta_q_prev_batch = delta_q_new_T,
        )                                                    # (T, S)

        # Rewards: reward_batch uses pos_next as "current" after the step
        r_T, _ = rm.reward_batch(
            pos_curr_batch     = pos_next_T,
            R_curr_SO3_batch   = R_next_T,
            delta_q_new_batch  = delta_q_new_T,
            delta_q_prev_batch = delta_q_prev_T,
            pos_goal_batch     = pos_goal.unsqueeze(0).expand(T, -1),
            R_goal_SO3_batch   = R_goal_SO3.unsqueeze(0).expand(T, -1, -1),
            use_focal          = use_focal,
        )                                                    # (T,)

        self._write_batch(s_T, a_norm_T, r_T, sp_T, done_T)

        # ── HER transitions (FUTURE strategy) ──────────────────────────────
        if ratio > 0.0 and T > 1:
            # For each step t, decide whether to generate a HER transition
            # Sample future goal indices: for step t, goal comes from [t+1, T-1]
            # Steps that have no future (t == T-1) are excluded.
            eligible = torch.arange(T - 1, device=dev)       # steps 0..T-2
            n_eligible = len(eligible)

            # Bernoulli mask over eligible steps
            mask = torch.rand(n_eligible, device=dev) < ratio  # (T-1,) bool

            if mask.any():
                t_idx = eligible[mask]                       # which steps get HER

                # For each selected step, sample a random future index
                # future in [t+1, T-1] — use uniform random offset
                range_sizes  = T - 1 - t_idx                # (M,)
                rand_offsets = (torch.rand(len(t_idx), device=dev) * range_sizes).long()
                future_idx   = t_idx + 1 + rand_offsets     # (M,)

                # HER goal = achieved pose at the future step
                her_pos_goal = pos_next_T[future_idx]        # (M, 3)
                her_R_goal   = R_next_T[future_idx]          # (M, 3, 3)

                # HER done: True when t reaches the future goal step or ep ended
                her_done_T = done_T[t_idx].clone()
                at_goal    = (t_idx == future_idx)
                her_done_T[at_goal] = 1.0

                s_her = rm.build_state_batch(
                    pos_curr_batch     = pos_curr_T[t_idx],
                    pos_goal           = her_pos_goal,
                    R_goal_SO3         = her_R_goal,
                    delta_q_prev_batch = delta_q_prev_T[t_idx],
                )                                            # (M, S)

                sp_her = rm.build_state_batch(
                    pos_curr_batch     = pos_next_T[t_idx],
                    pos_goal           = her_pos_goal,
                    R_goal_SO3         = her_R_goal,
                    delta_q_prev_batch = delta_q_new_T[t_idx],
                )                                            # (M, S)

                r_her, _ = rm.reward_batch(
                    pos_curr_batch     = pos_next_T[t_idx],
                    R_curr_SO3_batch   = R_next_T[t_idx],
                    delta_q_new_batch  = delta_q_new_T[t_idx],
                    delta_q_prev_batch = delta_q_prev_T[t_idx],
                    pos_goal_batch     = her_pos_goal,
                    R_goal_SO3_batch   = her_R_goal,
                    use_focal          = use_focal,
                )                                            # (M,)

                self._write_batch(s_her, a_norm_T[t_idx],
                                   r_her, sp_her, her_done_T)

        self._episode_cache = []
        self._episodes_done += 1

    # -------------------------------------------------------------------------
    # Sampling  (unchanged interface — O(batch_size), no deque conversion)
    # -------------------------------------------------------------------------

    def sample(self, batch_size: int) -> dict | None:
        """
        Sample a random mini-batch from the GPU ring buffer.

        Returns dict of float32 tensors already on self.device:
            s, a, r, s_prime, done  —  (B, state_dim), (B, n), (B,), (B, state_dim), (B,)
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
    # Internal write helpers
    # -------------------------------------------------------------------------

    def _write_batch(self,
                     s:       torch.Tensor,   # (B, S)
                     a:       torch.Tensor,   # (B, n)
                     r:       torch.Tensor,   # (B,)
                     s_prime: torch.Tensor,   # (B, S)
                     done:    torch.Tensor):  # (B,)
        """
        Write B transitions into contiguous ring-buffer slots.
        Wraps around the ring modulo capacity.
        """
        B = s.shape[0]

        # Cast to float32 and move to device in one go
        s       = s.to(dtype=torch.float32, device=self.device)
        a       = a.to(dtype=torch.float32, device=self.device)
        r       = r.to(dtype=torch.float32, device=self.device)
        s_prime = s_prime.to(dtype=torch.float32, device=self.device)
        done    = done.to(dtype=torch.float32, device=self.device)

        # Build index array wrapping around capacity
        idx = torch.arange(self._ptr, self._ptr + B, device=self.device) % self.capacity

        self._s      [idx] = s
        self._a      [idx] = a
        self._r      [idx] = r
        self._s_prime[idx] = s_prime
        self._done   [idx] = done

        self._ptr  = (self._ptr + B) % self.capacity
        self._size = min(self._size + B, self.capacity)


# ---------------------------------------------------------------------------
# Module-level helper (avoids repeated isinstance checks)
# ---------------------------------------------------------------------------

def _cpu64(x) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.detach().to(dtype=torch.float64, device="cpu")
    return torch.tensor(x, dtype=torch.float64)