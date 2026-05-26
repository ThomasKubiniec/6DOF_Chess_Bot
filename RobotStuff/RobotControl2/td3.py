"""
Twin Delayed Deep Deterministic Policy Gradient (TD3)
for robot inverse kinematics via reinforcement learning.

────────────────────────────────────────────────────────────────────────────
Algorithm overview
────────────────────────────────────────────────────────────────────────────

Networks
  Actor   π_φ(s)          → ã ∈ [-1, 1]^n   (normalised joint deltas)
  Critic₁ Q_θ₁(s ‖ a)     → scalar value
  Critic₂ Q_θ₂(s ‖ a)     → scalar value
  + delayed target copies of all three.

Training loop (caller's responsibility)
  1. Roll out π_φ in the environment; store (s, a, r, s', d) in replay buffer D.
     Optionally apply Hindsight Experience Replay (HER) before storing.
  2. Sample a mini-batch from D.
  3. Compute TD targets y via compute_targets().
  4. Update local critics via update_critics().
  5. Every policy_delay steps: update actor via update_actor(), then
     soft-update all target networks via soft_update_targets().

Action normalisation convention
  The actor outputs values in [-1, 1] (via Tanh).
  Denormalise to actual joint deltas before applying to the robot:
      delta_q = reward_math.denormalize_from_range(a_norm, low, high)
  where low = q_l - q_h and high = q_h - q_l.

Bugs fixed vs original
  1. DQN no longer inherits from nothing — see dqn.py (now nn.Module).
  2. output_function passed as instances (nn.Tanh(), nn.Identity()) not classes.
  3. compute_target_actions() indexed batch_sample[:,3] which is wrong for a
     dict-of-tensors buffer — now uses batch["s_prime"] correctly.
  4. compute_targets() called torch.cat([s_prime, targ_a_prime]) without dim=1
     — fixed to torch.cat([s_prime, targ_a_prime], dim=1) for (B, S+n) input.
  5. Update_Target_Networks() called self.soft_update() which didn't exist —
     renamed to soft_update_foreach() consistently.
  6. Actor loss was never implemented — added update_actor().
  7. Optimisers are now created inside TD3 so the caller doesn't have to.
"""

import torch
import torch.nn as nn
from torch.distributions import Normal

import copy

from dqn import DQN


class TD3:
    """
    TD3 agent for continuous robot control.

    Parameters
    ----------
    state_dim    : dimension of the state vector  (output of Reward_Math.build_state)
    action_dim   : number of joints  n  (actor outputs n values in [-1, 1])
    hidden_width : neurons per hidden layer
    hidden_depth : number of additional hidden layers (beyond the first)
    actor_lr     : learning rate for the actor optimiser    (default 1e-4)
    critic_lr    : learning rate for the critic optimisers  (default 1e-3)
    dtype        : torch dtype shared by all networks (default torch.float32)
    """

    def __init__(self,
                 state_dim:    int,
                 action_dim:   int,
                 hidden_width: int,
                 hidden_depth: int,
                 actor_lr:     float       = 1e-4,
                 critic_lr:    float       = 1e-3,
                 dtype:        torch.dtype = torch.float32):

        self.state_dim    = state_dim
        self.action_dim   = action_dim
        self.hidden_width = hidden_width
        self.hidden_depth = hidden_depth
        self.dtype        = dtype

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._build_networks()
        self._build_optimisers(actor_lr, critic_lr)

    # ─────────────────────────────────────────────────────────────────────────
    # Initialisation
    # ─────────────────────────────────────────────────────────────────────────

    def _build_networks(self):
        kw = dict(hidden_width=self.hidden_width,
                  hidden_depth=self.hidden_depth,
                  dtype=self.dtype)

        # Actor: state → joint delta (normalised to [-1, 1] via Tanh)
        self.actor = DQN(input_size=self.state_dim,
                         output_size=self.action_dim,
                         output_activation=nn.Tanh(),
                         **kw)

        # Critics: (state ‖ action) → scalar Q-value  (no output nonlinearity)
        self.critic_1 = DQN(input_size=self.state_dim + self.action_dim,
                            output_size=1,
                            output_activation=nn.Identity(),
                            **kw)
        self.critic_2 = DQN(input_size=self.state_dim + self.action_dim,
                            output_size=1,
                            output_activation=nn.Identity(),
                            **kw)

        # Target networks — deep copies, frozen from the optimiser's perspective
        self.target_actor    = copy.deepcopy(self.actor)
        self.target_critic_1 = copy.deepcopy(self.critic_1)
        self.target_critic_2 = copy.deepcopy(self.critic_2)

        # Target networks never receive gradient updates directly
        for net in (self.target_actor, self.target_critic_1, self.target_critic_2):
            for p in net.parameters():
                p.requires_grad_(False)

    def _build_optimisers(self, actor_lr: float, critic_lr: float):
        self.actor_optim    = torch.optim.Adam(self.actor.parameters(),    lr=actor_lr)
        self.critic_1_optim = torch.optim.Adam(self.critic_1.parameters(), lr=critic_lr)
        self.critic_2_optim = torch.optim.Adam(self.critic_2.parameters(), lr=critic_lr)

    # ─────────────────────────────────────────────────────────────────────────
    # Noise helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _clip_noise(self, sigma: float, clip: float) -> torch.Tensor:
        """
        Sample clipped Gaussian noise for target policy smoothing.

            ε ~ N(0, σ)  clipped to [-clip, clip]
        """
        eps = Normal(0.0, sigma).sample(torch.Size([self.action_dim]))
        return torch.clamp(eps, -clip, clip).to(device=self.device, dtype=self.dtype)

    # ─────────────────────────────────────────────────────────────────────────
    # Action selection  (used during rollout and target computation)
    # ─────────────────────────────────────────────────────────────────────────

    def select_action(self,
                      state:      torch.Tensor,
                      expl_sigma: float = 0.1,
                      expl_clip:  float = 0.5,
                      a_low:      float = -1.0,
                      a_high:     float =  1.0) -> torch.Tensor:
        """
        Compute an action from the local actor with exploration noise.

            a = clip( π_φ(s) + clip(N(0,σ), ±expl_clip), a_low, a_high )

        Parameters
        ----------
        state      : (S,) state tensor
        expl_sigma : std of exploration noise
        expl_clip  : noise clipping bound
        a_low/high : action space bounds (default [-1, 1] — normalised space)
        """
        state = state.to(device=self.device, dtype=self.dtype)
        with torch.no_grad():
            a = self.actor(state)
        noise = self._clip_noise(sigma=expl_sigma, clip=expl_clip)
        return torch.clamp(a + noise, a_low, a_high)

    def _target_action(self,
                       s_prime:      torch.Tensor,
                       target_sigma: float = 0.2,
                       target_clip:  float = 0.5,
                       a_low:        float = -1.0,
                       a_high:       float =  1.0) -> torch.Tensor:
        """
        Compute smoothed target action for TD target computation.

            ã' = clip( π_φ'(s') + clip(N(0,σ), ±clip), a_low, a_high )

        Parameters
        ----------
        s_prime      : (B, S) next-state batch
        target_sigma : std of target policy smoothing noise
        target_clip  : noise clipping bound for smoothing
        """
        with torch.no_grad():
            a_prime = self.target_actor(s_prime)            # (B, n)
        noise = self._clip_noise(sigma=target_sigma,
                                 clip=target_clip).unsqueeze(0)   # (1, n) → broadcasts
        return torch.clamp(a_prime + noise, a_low, a_high)  # (B, n)

    # ─────────────────────────────────────────────────────────────────────────
    # TD target computation
    # ─────────────────────────────────────────────────────────────────────────

    def compute_targets(self,
                        batch:        dict[str, torch.Tensor],
                        gamma:        float = 0.99,
                        target_sigma: float = 0.2,
                        target_clip:  float = 0.5) -> torch.Tensor:
        """
        Compute the Bellman TD target:

            y = r + γ (1 − d) · min( Q_θ'₁(s', ã'), Q_θ'₂(s', ã') )

        Parameters
        ----------
        batch  : dict from Replay_Buffer.sample()
                 keys: 's', 'a', 'r', 's_prime', 'done'   all (B, *)
        gamma  : discount factor

        Returns
        -------
        y : (B, 1) TD targets (detached — no gradient flows through targets)
        """
        r       = batch["r"].to(device=self.device, dtype=self.dtype)        # (B,)
        s_prime = batch["s_prime"].to(device=self.device, dtype=self.dtype)  # (B, S)
        done    = batch["done"].to(device=self.device, dtype=self.dtype)     # (B,)

        a_prime = self._target_action(s_prime,
                                      target_sigma=target_sigma,
                                      target_clip=target_clip)               # (B, n)

        sa_prime = torch.cat([s_prime, a_prime], dim=1)                      # (B, S+n)

        with torch.no_grad():
            q1_targ = self.target_critic_1(sa_prime)   # (B, 1)
            q2_targ = self.target_critic_2(sa_prime)   # (B, 1)

        q_min = torch.minimum(q1_targ, q2_targ)        # (B, 1)

        # r and done are (B,); unsqueeze to (B,1) to broadcast with q_min (B,1)
        y = r.unsqueeze(1) + gamma * (1.0 - done.unsqueeze(1)) * q_min

        return y.detach()   # (B, 1)

    # ─────────────────────────────────────────────────────────────────────────
    # Critic update
    # ─────────────────────────────────────────────────────────────────────────

    def update_critics(self,
                       batch: dict[str, torch.Tensor],
                       gamma: float = 0.99,
                       target_sigma: float = 0.2,
                       target_clip:  float = 0.5
                       ) -> tuple[float, float]:
        """
        Compute critic losses and perform one gradient step on each critic.

            L_i = MSE( Q_θᵢ(s, a),  y )

        Returns
        -------
        (loss_1, loss_2) : Python floats for logging
        """
        s = batch["s"].to(device=self.device, dtype=self.dtype)  # (B, S)
        a = batch["a"].to(device=self.device, dtype=self.dtype)  # (B, n)

        sa = torch.cat([s, a], dim=1)                            # (B, S+n)

        y = self.compute_targets(batch, gamma=gamma,
                                 target_sigma=target_sigma,
                                 target_clip=target_clip)        # (B, 1)

        q1 = self.critic_1(sa)   # (B, 1)
        q2 = self.critic_2(sa)   # (B, 1)

        loss_1 = nn.functional.mse_loss(q1, y)
        loss_2 = nn.functional.mse_loss(q2, y)

        self.critic_1_optim.zero_grad()
        loss_1.backward()
        self.critic_1_optim.step()

        self.critic_2_optim.zero_grad()
        loss_2.backward()
        self.critic_2_optim.step()

        return loss_1.item(), loss_2.item()

    # ─────────────────────────────────────────────────────────────────────────
    # Actor update  (delayed — call every `policy_delay` critic updates)
    # ─────────────────────────────────────────────────────────────────────────

    def update_actor(self, batch: dict[str, torch.Tensor]) -> float:
        """
        Update the actor by maximising Q_θ₁(s, π_φ(s)).

            L_actor = -mean( Q_θ₁(s, π_φ(s)) )

        Critic parameters are frozen during this step.

        Returns
        -------
        actor_loss : Python float for logging
        """
        s = batch["s"].to(device=self.device, dtype=self.dtype)  # (B, S)

        # Freeze critic params — no need to backprop through them
        for p in self.critic_1.parameters():
            p.requires_grad_(False)

        a_pred = self.actor(s)                          # (B, n)
        sa     = torch.cat([s, a_pred], dim=1)          # (B, S+n)
        actor_loss = -self.critic_1(sa).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        # Re-enable critic gradients
        for p in self.critic_1.parameters():
            p.requires_grad_(True)

        return actor_loss.item()

    # ─────────────────────────────────────────────────────────────────────────
    # Soft target update  (Polyak averaging)
    # ─────────────────────────────────────────────────────────────────────────

    def _soft_update(self, target_net: nn.Module,
                     source_net: nn.Module,
                     tau: float):
        """
        θ_target ← τ · θ_source + (1 − τ) · θ_target

        Uses torch._foreach for efficiency (fused CUDA kernels when available).
        """
        target_params = list(target_net.parameters())
        source_params = list(source_net.parameters())

        with torch.no_grad():
            torch._foreach_mul_(target_params, 1.0 - tau)
            torch._foreach_add_(target_params, source_params, alpha=tau)

    def soft_update_targets(self, tau: float = 0.005):
        """Polyak-average all three target networks."""
        self._soft_update(self.target_critic_1, self.critic_1, tau)
        self._soft_update(self.target_critic_2, self.critic_2, tau)
        self._soft_update(self.target_actor,    self.actor,    tau)

    # ─────────────────────────────────────────────────────────────────────────
    # Convenience: full training step
    # ─────────────────────────────────────────────────────────────────────────

    def train_step(self,
                   batch:        dict[str, torch.Tensor],
                   gamma:        float = 0.99,
                   tau:          float = 0.005,
                   policy_delay: int   = 2,
                   step:         int   = 0,
                   target_sigma: float = 0.2,
                   target_clip:  float = 0.5,
                   ) -> dict[str, float]:
        """
        One complete TD3 training step.

        Parameters
        ----------
        batch        : mini-batch from Replay_Buffer.sample()
        gamma        : discount factor
        tau          : Polyak averaging coefficient
        policy_delay : update actor + targets every this many critic steps
        step         : current global training step index (for delay logic)
        target_sigma : std of target policy smoothing noise
        target_clip  : clipping bound for target smoothing noise

        Returns
        -------
        info : dict with loss values for logging
        """
        l1, l2 = self.update_critics(batch, gamma=gamma,
                                     target_sigma=target_sigma,
                                     target_clip=target_clip)
        info = {"critic_loss_1": l1, "critic_loss_2": l2}

        if step % policy_delay == 0:
            actor_loss = self.update_actor(batch)
            self.soft_update_targets(tau)
            info["actor_loss"] = actor_loss

        return info