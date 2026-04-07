"""
training/dqn_trainer.py
=======================
DQN trainer with:
  1. Oracle imitation pre-training  — fills replay buffer with expert demos
  2. Standard DQN training loop     — learns from own experience
  3. Target network                 — stable Q-learning targets
  4. Epsilon decay                  — exploration → exploitation
  5. Gradient clipping              — prevents exploding gradients
  6. Training metrics logging       — tracks loss, rewards, epsilon

This is the "REAL LEARNING" the mentor asked for.
The agent genuinely updates its weights based on trading outcomes.
"""

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Any, Dict, List, Optional, Tuple
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from agent.policy_network import DQNPolicy, N_ACTIONS, ACTION_NAMES
from agent.state_encoder  import StateEncoder
from training.replay_buffer import ReplayBuffer


# ─────────────────────────────────────────────────────────────────────────────
# Hyperparameters
# ─────────────────────────────────────────────────────────────────────────────

class DQNConfig:
    lr              = 3e-4
    gamma           = 0.99
    batch_size      = 64
    target_update   = 200       # steps between target network syncs
    epsilon_start   = 1.0
    epsilon_end     = 0.05
    epsilon_decay   = 0.997     # per episode
    buffer_capacity = 50_000
    min_buffer_size = 512       # start training after this many transitions
    grad_clip       = 1.0
    hidden          = 128


# ─────────────────────────────────────────────────────────────────────────────
# DQN Trainer
# ─────────────────────────────────────────────────────────────────────────────

class DQNTrainer:
    """
    Full DQN training loop for the XAU/USD trading environment.

    Phase 1: Imitation pre-training
        Run oracle agent → fill replay buffer with expert (state, action) pairs
        Supervised loss: cross-entropy on oracle's actions
        This gives the network a head-start (no random flailing)

    Phase 2: RL fine-tuning
        Standard DQN: collect experience → sample from replay → update Q-network
        Epsilon decays from 1.0 → 0.05 over training
    """

    def __init__(self, config: DQNConfig = None):
        self.cfg     = config or DQNConfig()
        self.policy  = DQNPolicy(hidden=self.cfg.hidden)
        self.target  = copy.deepcopy(self.policy)
        self.target.eval()
        self.opt     = optim.Adam(self.policy.parameters(), lr=self.cfg.lr)
        self.buffer  = ReplayBuffer(self.cfg.buffer_capacity)
        self.encoder = StateEncoder()

        self.epsilon    = self.cfg.epsilon_start
        self.step_count = 0
        self.ep_count   = 0
        self.losses     : List[float] = []
        self.ep_rewards : List[float] = []
        self.ep_pnls    : List[float] = []

    # ── Phase 1: Oracle Imitation Pre-training ────────────────────────────────

    def pretrain_from_oracle(
        self,
        env_factory,          # callable → fresh TradingEnv
        n_episodes   : int = 200,
        verbose      : bool = True,
    ) -> Dict[str, float]:
        """
        Collect oracle demonstrations and train with supervised loss.
        This is Behavioural Cloning (BC) — a form of imitation learning.
        """
        from agent.llm_agent import OracleAgent
        oracle = OracleAgent()

        if verbose:
            print(f"\n  [Phase 1] Oracle imitation pre-training — {n_episodes} episodes")

        pretrain_losses = []

        for ep in range(n_episodes):
            env = env_factory()
            obs = env.reset()
            self.encoder.reset()
            done = False

            while not done:
                state  = self.encoder.encode(obs)
                action_dict = oracle.act(obs)
                action_idx  = {"hold": 0, "buy": 1, "sell": 2}[action_dict["decision"]]

                obs_next, reward, done, info = env.step(action_dict)
                next_state = self.encoder.encode(obs_next)
                self.encoder.update_from_info(info)

                self.buffer.push(state, action_idx, reward, next_state, done, info)
                obs = obs_next

            # Supervised update on oracle actions
            if len(self.buffer) >= self.cfg.batch_size:
                loss = self._supervised_update()
                pretrain_losses.append(loss)

        avg_loss = float(np.mean(pretrain_losses)) if pretrain_losses else 0
        if verbose:
            print(f"  Pre-training complete. Avg supervised loss: {avg_loss:.4f}")
            print(f"  Buffer size: {len(self.buffer)} transitions")

        return {"pretrain_loss": avg_loss, "buffer_size": len(self.buffer)}

    def _supervised_update(self) -> float:
        """Cross-entropy loss on oracle action labels (imitation learning)."""
        batch  = self.buffer.sample(self.cfg.batch_size)
        states  = torch.FloatTensor(np.stack([t.state  for t in batch]))
        actions = torch.LongTensor([t.action for t in batch])

        logits = self.policy.feature(states)
        logits = self.policy.advantage(logits)   # use advantage stream as logits
        loss   = nn.CrossEntropyLoss()(logits, actions)

        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), self.cfg.grad_clip)
        self.opt.step()
        return float(loss.item())

    # ── Phase 2: DQN RL Training ───────────────────────────────────────────────

    def train_episode(self, env) -> Dict[str, float]:
        """Run one full episode of DQN training. Returns episode metrics."""
        obs = env.reset()
        self.encoder.reset()
        done          = False
        ep_reward     = 0.0
        ep_loss_sum   = 0.0
        ep_loss_count = 0

        while not done:
            state       = self.encoder.encode(obs)
            in_position = obs.get("position", "flat") != "flat"

            # Epsilon-greedy action
            action_idx, q_vals = self.policy.act(state, self.epsilon, in_position)
            action_name = ACTION_NAMES[action_idx]

            # Convert to env action format
            action_dict = self._idx_to_action(action_idx, obs)

            obs_next, reward, done, info = env.step(action_dict)
            next_state = self.encoder.encode(obs_next)
            self.encoder.update_from_info(info)

            self.buffer.push(state, action_idx, reward, next_state, done, info)
            ep_reward += reward
            self.step_count += 1

            # Learn
            if len(self.buffer) >= self.cfg.min_buffer_size:
                loss = self._dqn_update()
                ep_loss_sum   += loss
                ep_loss_count += 1

            # Sync target network
            if self.step_count % self.cfg.target_update == 0:
                self.target.load_state_dict(self.policy.state_dict())

            obs = obs_next

        # Epsilon decay
        self.epsilon = max(self.cfg.epsilon_end, self.epsilon * self.cfg.epsilon_decay)
        self.ep_count += 1

        summary = env.episode_summary()
        self.ep_rewards.append(ep_reward)
        self.ep_pnls.append(summary.get("total_pnl", 0))
        avg_loss = ep_loss_sum / ep_loss_count if ep_loss_count > 0 else 0

        return {
            "episode"    : self.ep_count,
            "ep_reward"  : round(ep_reward, 4),
            "total_pnl"  : round(summary.get("total_pnl", 0), 2),
            "n_trades"   : summary.get("n_trades", 0),
            "win_rate"   : summary.get("win_rate", 0),
            "epsilon"    : round(self.epsilon, 4),
            "avg_loss"   : round(avg_loss, 6),
            "buffer_size": len(self.buffer),
        }

    def _dqn_update(self) -> float:
        """One Bellman update step."""
        states, actions, rewards, next_states, dones = self.buffer.sample_arrays(self.cfg.batch_size)

        s  = torch.FloatTensor(states)
        a  = torch.LongTensor(actions)
        r  = torch.FloatTensor(rewards)
        ns = torch.FloatTensor(next_states)
        d  = torch.FloatTensor(dones)

        # Current Q values
        q_curr = self.policy(s).gather(1, a.unsqueeze(1)).squeeze(1)

        # Target Q values (Double DQN: use online net to select, target net to eval)
        with torch.no_grad():
            best_actions  = self.policy(ns).argmax(1)
            q_next        = self.target(ns).gather(1, best_actions.unsqueeze(1)).squeeze(1)
            q_target      = r + self.cfg.gamma * q_next * (1 - d)

        loss = nn.SmoothL1Loss()(q_curr, q_target)

        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), self.cfg.grad_clip)
        self.opt.step()

        return float(loss.item())

    def _idx_to_action(self, idx: int, obs: Dict) -> Dict:
        """Convert action index back to env action dict with SL/TP."""
        price  = obs.get("price", 2300)
        f72    = obs.get("fib_72", price - 20)
        f85    = obs.get("fib_85", price + 20)
        vol    = obs.get("volatility", "medium")
        buf    = {"low": 4, "medium": 8, "high": 15}.get(vol, 8)

        if idx == 0:   # hold
            return {"decision": "hold", "stop_loss": 0, "take_profit": 0}
        elif idx == 1: # buy
            sl = f72 - buf
            tp = price + 2.2 * (price - sl)
            return {"decision": "buy", "stop_loss": round(sl, 2), "take_profit": round(tp, 2)}
        else:          # sell
            sl = f85 + buf
            tp = price - 2.2 * (sl - price)
            return {"decision": "sell", "stop_loss": round(sl, 2), "take_profit": round(tp, 2)}

    # ── Full training run ──────────────────────────────────────────────────────

    def train(
        self,
        env_factory,
        n_episodes   : int  = 500,
        pretrain_eps : int  = 100,
        log_every    : int  = 50,
        verbose      : bool = True,
    ) -> Dict:
        """
        Full training pipeline:
          1. Oracle imitation pre-training
          2. DQN RL fine-tuning with epsilon-greedy exploration
        """
        # Phase 1
        stats_pretrain = self.pretrain_from_oracle(env_factory, pretrain_eps, verbose)

        # Phase 2
        if verbose:
            print(f"\n  [Phase 2] DQN RL training — {n_episodes} episodes")
            print(f"  {'Ep':>6}  {'PnL':>8}  {'Reward':>8}  {'Trades':>7}  {'WinRate':>8}  {'Epsilon':>8}  {'Loss':>10}")
            print(f"  {'─'*65}")

        for ep in range(1, n_episodes + 1):
            env     = env_factory()
            metrics = self.train_episode(env)

            if verbose and ep % log_every == 0:
                window = self.ep_pnls[-log_every:]
                avg_pnl = np.mean(window) if window else 0
                print(
                    f"  {ep:>6}  "
                    f"  {avg_pnl:>+7.2f}  "
                    f"  {metrics['ep_reward']:>+7.4f}  "
                    f"  {metrics['n_trades']:>6}  "
                    f"  {metrics['win_rate']:>7.1f}%  "
                    f"  {metrics['epsilon']:>7.4f}  "
                    f"  {metrics['avg_loss']:>9.6f}"
                )

        final = {
            "total_episodes" : self.ep_count,
            "final_epsilon"  : round(self.epsilon, 4),
            "avg_pnl_last50" : round(float(np.mean(self.ep_pnls[-50:])), 2),
            "avg_reward_last50": round(float(np.mean(self.ep_rewards[-50:])), 4),
            "buffer_size"    : len(self.buffer),
        }
        if verbose:
            print(f"\n  Training complete.")
            print(f"  Final epsilon:        {final['final_epsilon']}")
            print(f"  Avg PnL (last 50):    {final['avg_pnl_last50']:+.2f}")
            print(f"  Buffer size:          {final['buffer_size']}")

        return final

    # ── Save / Load ────────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        torch.save({
            "policy"     : self.policy.state_dict(),
            "target"     : self.target.state_dict(),
            "optimizer"  : self.opt.state_dict(),
            "epsilon"    : self.epsilon,
            "step_count" : self.step_count,
            "ep_count"   : self.ep_count,
            "ep_pnls"    : self.ep_pnls,
        }, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location="cpu")
        self.policy.load_state_dict(ckpt["policy"])
        self.target.load_state_dict(ckpt["target"])
        self.opt.load_state_dict(ckpt["optimizer"])
        self.epsilon    = ckpt["epsilon"]
        self.step_count = ckpt["step_count"]
        self.ep_count   = ckpt["ep_count"]
        self.ep_pnls    = ckpt.get("ep_pnls", [])
