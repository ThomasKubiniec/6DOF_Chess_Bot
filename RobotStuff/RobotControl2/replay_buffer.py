"""
HER Replay Buffer for the TD3 Robot Arm Agent.

Episodes are stored in a temporary episode buffer as raw transitions.
At the end of each episode, finish_episode() is called. For every step t
in the episode it:

  1. Computes the REAL reward using the actual goal and stores that
     transition in the main replay buffer.

  2. With probability her_ratio, also stores a HER transition: a future
     achieved state from the same episode is picked as the relabelled goal
     (FUTURE strategy), the reward is recomputed as if that were always the
     goal, and done=True is set for the relabelled terminal step.

her_ratio decays linearly from her_ratio_start to her_ratio_end over
her_decay_steps calls to finish_episode(), giving dense relabelling early
in training when the buffer is sparse, tapering to light relabelling once
the policy is competent.

HER goal selection strategy: FUTURE (Andrychowicz et al., 2017) --
the relabelled goal is sampled uniformly from steps {t+1, ..., T-1} of
the same episode.
"""

from collections import deque, namedtuple
import random
import torch

# Ready-to-train transition stored in the main buffer
Transition = namedtuple("Transition", ["s", "a", "r", "s_prime", "done"])

# Per-step data cached during a live episode (all CPU float64 tensors)
RawTransition = namedtuple("RawTransition", [
    "q_new",         # (n,)   joint angles after action
    "delta_q_new",   # (n,)   action applied this step
    "delta_q_prev",  # (n,)   action applied previous step
    "pos_curr",      # (3,)   EE position BEFORE this step (= s observation)
    "R_curr",        # (3,3)  EE rotation BEFORE this step
    "pos_next",      # (3,)   EE position AFTER this step  (= s' observation)
    "R_next",        # (3,3)  EE rotation AFTER this step
    "a_norm",        # (n,)   normalised action in [-1, 1]
    "done",          # bool
])


class Replay_Buffer:
    """
    Replay buffer with Hindsight Experience Replay (HER, FUTURE strategy)
    and a linearly decaying HER ratio.

    Parameters
    ----------
    buffer_size      : maximum transitions in the main replay buffer
    reward_math      : Reward_Math instance (used to recompute rewards / states)
    her_ratio_start  : HER probability at the start of training  (default 0.80)
    her_ratio_end    : HER probability floor after decay          (default 0.20)
    her_decay_steps  : number of finish_episode() calls over which to decay
    """

    def __init__(self,
                 buffer_size:     int,
                 reward_math,
                 her_ratio_start: float = 0.80,
                 her_ratio_end:   float = 0.20,
                 her_decay_steps: int   = 1000):

        self.buffer          = deque(maxlen=buffer_size)
        self.reward_math     = reward_math
        self.her_ratio_start = her_ratio_start
        self.her_ratio_end   = her_ratio_end
        self.her_decay_steps = her_decay_steps
        self._episodes_done  = 0

        self._episode_cache: list = []   # list[RawTransition]

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def her_ratio(self) -> float:
        """Current HER probability, decaying linearly with episodes completed."""
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
        """
        Cache one environment step during a live episode.
        Call every step; call finish_episode() once the episode ends.

        Parameters
        ----------
        q_new        : joint angles after applying action          (n,)
        delta_q_new  : joint delta applied this step               (n,)
        delta_q_prev : joint delta applied the previous step       (n,)
        pos_curr     : EE position before this step                (3,)
        R_curr       : EE rotation matrix before this step         (3,3)
        pos_next     : EE position after this step                 (3,)
        R_next       : EE rotation matrix after this step          (3,3)
        a_norm       : normalised action in [-1, 1]                (n,)
        done         : True if this step ends the episode
        """
        self._episode_cache.append(RawTransition(
            q_new        = self._t(q_new),
            delta_q_new  = self._t(delta_q_new),
            delta_q_prev = self._t(delta_q_prev),
            pos_curr     = self._t(pos_curr),
            R_curr       = self._t(R_curr),
            pos_next     = self._t(pos_next),
            R_next       = self._t(R_next),
            a_norm       = self._t(a_norm),
            done         = done,
        ))

    def finish_episode(self,
                       pos_goal:   torch.Tensor,
                       R_goal_SO3: torch.Tensor,
                       use_focal:  bool = True):
        """
        Process the cached episode and commit transitions to the main buffer.

        For each step t:
          1. Stores the real transition with the actual goal and reward.
          2. With probability her_ratio, stores a HER transition using a
             randomly selected future achieved pose as the relabelled goal.

        Clears the episode cache and increments the episode counter.

        Parameters
        ----------
        pos_goal   : actual goal EE position for this episode    (3,)
        R_goal_SO3 : actual goal rotation matrix for this episode (3,3)
        use_focal  : passed through to Reward_Math.reward()
        """
        episode  = self._episode_cache
        T        = len(episode)
        pos_goal = self._t(pos_goal)
        R_goal   = self._t(R_goal_SO3)
        rm       = self.reward_math
        ratio    = self.her_ratio

        for t, raw in enumerate(episode):

            # ------------------------------------------------------------------
            # 1. Real transition  (actual goal, actual reward)
            # ------------------------------------------------------------------
            s = rm.build_state(
                pos_curr     = raw.pos_curr,
                pos_goal     = pos_goal,
                R_goal_SO3   = R_goal,
                delta_q_prev = raw.delta_q_prev,
            )
            s_prime = rm.build_state(
                pos_curr     = raw.pos_next,
                pos_goal     = pos_goal,
                R_goal_SO3   = R_goal,
                delta_q_prev = raw.delta_q_new,
            )

            rm.reset_episode()
            r_real, _ = rm.reward(
                q_new        = raw.q_new,
                delta_q_new  = raw.delta_q_new,
                delta_q_prev = raw.delta_q_prev,
                pos_goal     = pos_goal,
                R_goal_SO3   = R_goal,
                use_focal    = use_focal,
            )

            self._store(s, raw.a_norm, r_real, s_prime, raw.done)

            # ------------------------------------------------------------------
            # 2. HER transition  (FUTURE strategy)
            # ------------------------------------------------------------------
            if random.random() < ratio and t < T - 1:
                # Sample a random future step's achieved pose as the new goal
                future_idx   = random.randint(t + 1, T - 1)
                future       = episode[future_idx]
                her_pos_goal = future.pos_next    # (3,)  achieved EE position
                her_R_goal   = future.R_next      # (3,3) achieved EE rotation

                s_her = rm.build_state(
                    pos_curr     = raw.pos_curr,
                    pos_goal     = her_pos_goal,
                    R_goal_SO3   = her_R_goal,
                    delta_q_prev = raw.delta_q_prev,
                )
                s_prime_her = rm.build_state(
                    pos_curr     = raw.pos_next,
                    pos_goal     = her_pos_goal,
                    R_goal_SO3   = her_R_goal,
                    delta_q_prev = raw.delta_q_new,
                )

                # Terminal for the HER transition: this step reaches the
                # relabelled goal (t == future_idx) or the episode ended.
                her_done = (t == future_idx) or raw.done

                rm.reset_episode()
                r_her, _ = rm.reward(
                    q_new        = raw.q_new,
                    delta_q_new  = raw.delta_q_new,
                    delta_q_prev = raw.delta_q_prev,
                    pos_goal     = her_pos_goal,
                    R_goal_SO3   = her_R_goal,
                    use_focal    = use_focal,
                )

                self._store(s_her, raw.a_norm, r_her, s_prime_her, her_done)

        # Housekeeping
        self._episode_cache = []
        self._episodes_done += 1

    # -------------------------------------------------------------------------
    # Sampling API  (unchanged interface for TD3.train_step)
    # -------------------------------------------------------------------------

    def sample(self, batch_size: int) -> dict | None:
        """
        Sample a random mini-batch of ready-to-train transitions.

        Returns a dict with keys 's', 'a', 'r', 's_prime', 'done'
        (all (B, *) float64 tensors), or None if the buffer is too small.
        """
        if batch_size > len(self.buffer):
            return None

        batch = random.sample(list(self.buffer), batch_size)

        return {
            "s":       torch.stack([t.s       for t in batch]),
            "a":       torch.stack([t.a       for t in batch]),
            "r":       torch.stack([t.r       for t in batch]),
            "s_prime": torch.stack([t.s_prime for t in batch]),
            "done":    torch.stack([t.done    for t in batch]),
        }

    def __len__(self) -> int:
        return len(self.buffer)

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _store(self,
               s:       torch.Tensor,
               a:       torch.Tensor,
               r:       torch.Tensor,
               s_prime: torch.Tensor,
               done):
        self.buffer.append(Transition(
            s       = s.detach().cpu(),
            a       = a.detach().cpu(),
            r       = r.detach().cpu() if isinstance(r, torch.Tensor)
                      else torch.tensor(r, dtype=torch.float64),
            s_prime = s_prime.detach().cpu(),
            done    = torch.tensor(float(done), dtype=torch.float64),
        ))

    @staticmethod
    def _t(x) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            return x.detach().to(dtype=torch.float64, device="cpu")
        return torch.tensor(x, dtype=torch.float64)