"""
agent/policy_network.py
=======================
Neural network policies for the trading RL agent.

Two architectures:
  1. DQNPolicy      — for Deep Q-Network (off-policy, replay buffer)
  2. ActorCritic    — for PPO (on-policy, trajectory buffer)

Both take STATE_DIM inputs and output action logits over [hold, buy, sell].

Architecture design choices:
  - Small networks (64-128 hidden) to avoid overfitting on limited trading data
  - LayerNorm instead of BatchNorm (works with single samples at inference)
  - Residual connections in the deeper version for gradient flow
  - Action masking: if already in position, buy/sell logits are set to -inf
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple

from agent.state_encoder import STATE_DIM

N_ACTIONS = 3    # 0=hold, 1=buy, 2=sell
ACTION_NAMES = ["hold", "buy", "sell"]


# ─────────────────────────────────────────────────────────────────────────────
# DQN Policy (off-policy, used with ReplayBuffer)
# ─────────────────────────────────────────────────────────────────────────────

class DQNPolicy(nn.Module):
    """
    Dueling DQN architecture:
      - Shared feature extractor
      - Separate value stream V(s) and advantage stream A(s,a)
      - Q(s,a) = V(s) + A(s,a) - mean(A)

    Dueling helps because the agent can learn state value
    independently of action advantages — useful when HOLD is
    often the right answer (state value is low regardless of action).
    """

    def __init__(self, state_dim: int = STATE_DIM, hidden: int = 128):
        super().__init__()

        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
        )

        # Value stream: V(s) → scalar
        self.value = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        # Advantage stream: A(s,a) → N_ACTIONS
        self.advantage = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Linear(64, N_ACTIONS),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)

    def forward(
        self,
        state: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            state: (batch, STATE_DIM)
            action_mask: (batch, N_ACTIONS) bool, True=valid action

        Returns:
            q_values: (batch, N_ACTIONS)
        """
        feat = self.feature(state)
        V    = self.value(feat)
        A    = self.advantage(feat)
        Q    = V + A - A.mean(dim=-1, keepdim=True)

        # Mask invalid actions (e.g. can't buy when already long)
        if action_mask is not None:
            Q = Q.masked_fill(~action_mask, float("-inf"))

        return Q

    @torch.no_grad()
    def act(
        self,
        state: np.ndarray,
        epsilon: float = 0.0,
        in_position: bool = False,
    ) -> Tuple[int, np.ndarray]:
        """
        Epsilon-greedy action selection with action masking.

        Args:
            state: STATE_DIM float array
            epsilon: exploration rate (0=greedy, 1=random)
            in_position: if True, mask out buy/sell (must hold)

        Returns:
            (action_idx, q_values)
        """
        # Action masking
        mask = torch.ones(1, N_ACTIONS, dtype=torch.bool)
        if in_position:
            mask[0, 1] = False   # can't buy
            mask[0, 2] = False   # can't sell

        if np.random.random() < epsilon:
            valid = mask[0].numpy()
            action = int(np.random.choice(np.where(valid)[0]))
            q_vals = np.zeros(N_ACTIONS)
        else:
            st = torch.FloatTensor(state).unsqueeze(0)
            q  = self.forward(st, mask).squeeze(0)
            action  = int(q.argmax().item())
            q_vals  = q.numpy()

        return action, q_vals


# ─────────────────────────────────────────────────────────────────────────────
# Actor-Critic for PPO
# ─────────────────────────────────────────────────────────────────────────────

class ActorCritic(nn.Module):
    """
    Shared-backbone actor-critic for PPO.

    Actor  → action probability distribution π(a|s)
    Critic → state value estimate V(s)

    The shared backbone means features learned for value estimation
    also improve the policy, and vice versa.
    """

    def __init__(self, state_dim: int = STATE_DIM, hidden: int = 128):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.LayerNorm(hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.Tanh(),
        )

        self.actor  = nn.Linear(hidden, N_ACTIONS)
        self.critic = nn.Linear(hidden, 1)

        self._init_weights()

    def _init_weights(self):
        nn.init.orthogonal_(self.actor.weight,  gain=0.01)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.zeros_(self.actor.bias)
        nn.init.zeros_(self.critic.bias)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (action_logits, state_value)."""
        feat   = self.shared(state)
        logits = self.actor(feat)
        value  = self.critic(feat).squeeze(-1)
        return logits, value

    def get_action(
        self,
        state: np.ndarray,
        in_position: bool = False,
        deterministic: bool = False,
    ) -> Tuple[int, float, float]:
        """
        Sample action from policy.

        Returns:
            (action, log_prob, value)
        """
        st     = torch.FloatTensor(state).unsqueeze(0)
        logits, value = self.forward(st)

        # Action masking
        if in_position:
            logits[0, 1] = float("-inf")
            logits[0, 2] = float("-inf")

        probs = F.softmax(logits, dim=-1)
        dist  = torch.distributions.Categorical(probs)

        if deterministic:
            action = int(probs.argmax().item())
        else:
            action = int(dist.sample().item())

        log_prob = float(dist.log_prob(torch.tensor(action)).item())
        val      = float(value.item())
        return action, log_prob, val

    def evaluate(
        self,
        states  : torch.Tensor,
        actions : torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """For PPO update: return log_probs, values, entropy."""
        logits, values = self.forward(states)
        probs    = F.softmax(logits, dim=-1)
        dist     = torch.distributions.Categorical(probs)
        log_probs = dist.log_prob(actions)
        entropy   = dist.entropy()
        return log_probs, values, entropy
