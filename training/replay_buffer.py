"""
training/replay_buffer.py
=========================
Experience replay buffer for offline RL training.

Stores transitions: (state, action, reward, next_state, done)
Supports:
  - Random sampling (DQN-style)
  - Prioritized sampling (PER) — samples high-TD-error transitions more
  - Oracle imitation data injection — pre-fill with expert demonstrations
  - Episode-level storage for trajectory-based methods (PPO)
"""

import random
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any


# ─────────────────────────────────────────────────────────────────────────────
# Transition
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Transition:
    state      : np.ndarray
    action     : int          # 0=hold, 1=buy, 2=sell
    reward     : float
    next_state : np.ndarray
    done       : bool
    info       : Dict[str, Any] = field(default_factory=dict)
    priority   : float = 1.0   # for PER


# ─────────────────────────────────────────────────────────────────────────────
# Replay Buffer
# ─────────────────────────────────────────────────────────────────────────────

class ReplayBuffer:
    """
    Uniform experience replay buffer.

    Usage:
        buf = ReplayBuffer(capacity=50_000)
        buf.push(state, action, reward, next_state, done)
        batch = buf.sample(64)
    """

    def __init__(self, capacity: int = 50_000):
        self.capacity = capacity
        self._buf: deque = deque(maxlen=capacity)

    def push(
        self,
        state      : np.ndarray,
        action     : int,
        reward     : float,
        next_state : np.ndarray,
        done       : bool,
        info       : Optional[Dict] = None,
    ) -> None:
        self._buf.append(Transition(
            state      = np.array(state,      dtype=np.float32),
            action     = int(action),
            reward     = float(reward),
            next_state = np.array(next_state, dtype=np.float32),
            done       = bool(done),
            info       = info or {},
        ))

    def sample(self, batch_size: int) -> List[Transition]:
        return random.sample(self._buf, min(batch_size, len(self._buf)))

    def sample_arrays(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        """Return numpy arrays ready for torch.from_numpy()."""
        batch = self.sample(batch_size)
        states      = np.stack([t.state      for t in batch])
        actions     = np.array([t.action     for t in batch], dtype=np.int64)
        rewards     = np.array([t.reward     for t in batch], dtype=np.float32)
        next_states = np.stack([t.next_state for t in batch])
        dones       = np.array([t.done       for t in batch], dtype=np.float32)
        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        return len(self._buf)

    @property
    def is_ready(self) -> bool:
        return len(self._buf) >= 256

    def stats(self) -> Dict:
        if not self._buf:
            return {"size": 0}
        rewards = [t.reward for t in self._buf]
        return {
            "size"       : len(self._buf),
            "capacity"   : self.capacity,
            "pct_full"   : round(len(self._buf) / self.capacity * 100, 1),
            "avg_reward" : round(float(np.mean(rewards)), 4),
            "max_reward" : round(float(np.max(rewards)),  4),
            "min_reward" : round(float(np.min(rewards)),  4),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Prioritized Experience Replay (PER)
# ─────────────────────────────────────────────────────────────────────────────

class PrioritizedReplayBuffer(ReplayBuffer):
    """
    PER buffer: transitions with higher TD-error get sampled more often.
    This helps the agent learn faster from surprising/informative transitions.

    After each training step, call update_priorities() with the new TD errors.
    """

    def __init__(self, capacity: int = 50_000, alpha: float = 0.6, beta: float = 0.4):
        super().__init__(capacity)
        self.alpha     = alpha   # how much prioritization to use (0=uniform)
        self.beta      = beta    # importance sampling correction
        self._indices  : List[int] = []
        self._priorities: deque = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done, info=None) -> None:
        max_p = max(self._priorities) if self._priorities else 1.0
        super().push(state, action, reward, next_state, done, info)
        self._priorities.append(max_p)

    def sample(self, batch_size: int) -> List[Transition]:
        buf_list = list(self._buf)
        pri_list = np.array(list(self._priorities), dtype=np.float32)
        probs    = pri_list ** self.alpha
        probs   /= probs.sum()
        n        = min(batch_size, len(buf_list))
        self._indices = np.random.choice(len(buf_list), size=n, replace=False, p=probs).tolist()
        return [buf_list[i] for i in self._indices]

    def update_priorities(self, td_errors: np.ndarray) -> None:
        buf_list = list(self._buf)
        pri_list = list(self._priorities)
        for idx, err in zip(self._indices, td_errors):
            if idx < len(pri_list):
                pri_list[idx] = float(abs(err)) + 1e-6
        self._priorities = deque(pri_list, maxlen=self.capacity)


# ─────────────────────────────────────────────────────────────────────────────
# Episode trajectory buffer (for PPO)
# ─────────────────────────────────────────────────────────────────────────────

class TrajectoryBuffer:
    """
    Stores complete episode trajectories for on-policy methods (PPO).
    Cleared after each policy update.
    """

    def __init__(self):
        self.states     : List[np.ndarray] = []
        self.actions    : List[int]        = []
        self.rewards    : List[float]      = []
        self.log_probs  : List[float]      = []
        self.values     : List[float]      = []
        self.dones      : List[bool]       = []

    def push(self, state, action, reward, log_prob, value, done):
        self.states.append(np.array(state, dtype=np.float32))
        self.actions.append(int(action))
        self.rewards.append(float(reward))
        self.log_probs.append(float(log_prob))
        self.values.append(float(value))
        self.dones.append(bool(done))

    def compute_returns(self, gamma: float = 0.99, gae_lambda: float = 0.95) -> np.ndarray:
        """Compute GAE (Generalized Advantage Estimation) returns."""
        rewards  = np.array(self.rewards,  dtype=np.float32)
        values   = np.array(self.values,   dtype=np.float32)
        dones    = np.array(self.dones,    dtype=np.float32)
        T        = len(rewards)
        returns  = np.zeros(T, dtype=np.float32)
        adv      = 0.0
        last_val = 0.0

        for t in reversed(range(T)):
            next_val = last_val if t == T - 1 else values[t + 1]
            mask     = 1.0 - dones[t]
            delta    = rewards[t] + gamma * next_val * mask - values[t]
            adv      = delta + gamma * gae_lambda * mask * adv
            returns[t] = adv + values[t]

        return returns

    def get_arrays(self) -> Tuple[np.ndarray, ...]:
        returns = self.compute_returns()
        states  = np.stack(self.states)
        actions = np.array(self.actions,   dtype=np.int64)
        lp      = np.array(self.log_probs, dtype=np.float32)
        adv     = returns - np.array(self.values, dtype=np.float32)
        adv     = (adv - adv.mean()) / (adv.std() + 1e-8)
        return states, actions, lp, returns, adv

    def clear(self):
        self.__init__()

    def __len__(self):
        return len(self.rewards)
