"""
training/ppo_trainer.py
=======================
PPO (Proximal Policy Optimization) trainer.

PPO is preferred over DQN for this environment because:
  - Trading has high variance rewards → PPO's clipped surrogate is more stable
  - Episode structure is natural for on-policy collection
  - Actor-critic naturally handles the value baseline (reduces variance)
  - No replay buffer needed → simpler, less memory

Key PPO innovations used here:
  - Clipped surrogate objective (prevents too-large policy updates)
  - GAE advantage estimation (reduces variance of policy gradient)
  - Value function loss with clipping
  - Entropy bonus (maintains exploration)
  - Multiple epochs over collected trajectories
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Optional
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from agent.policy_network  import ActorCritic, N_ACTIONS, ACTION_NAMES
from agent.state_encoder   import StateEncoder
from training.replay_buffer import TrajectoryBuffer


class PPOConfig:
    lr            = 2.5e-4
    gamma         = 0.99
    gae_lambda    = 0.95
    clip_eps      = 0.2      # PPO clip parameter
    value_coef    = 0.5      # value loss weight
    entropy_coef  = 0.01     # entropy bonus weight
    n_epochs      = 4        # update epochs per trajectory
    batch_size    = 32
    grad_clip     = 0.5
    hidden        = 128
    steps_per_update = 200   # collect this many steps before updating


class PPOTrainer:
    """
    PPO trainer with oracle imitation warm-start.

    Training flow:
        1. Collect `steps_per_update` steps from environment
        2. Compute GAE advantages
        3. Run `n_epochs` of PPO updates on the trajectory
        4. Clear trajectory buffer
        5. Repeat
    """

    def __init__(self, config: PPOConfig = None):
        self.cfg     = config or PPOConfig()
        self.ac      = ActorCritic(hidden=self.cfg.hidden)
        self.opt     = optim.Adam(self.ac.parameters(), lr=self.cfg.lr)
        self.encoder = StateEncoder()
        self.traj    = TrajectoryBuffer()

        self.total_steps  = 0
        self.total_updates= 0
        self.ep_count     = 0
        self.ep_pnls      : List[float] = []
        self.update_losses: List[Dict]  = []

    # ── Oracle warm-start (same as DQN) ───────────────────────────────────────

    def warmstart_from_oracle(self, env_factory, n_episodes=100, verbose=True):
        """Behavioural cloning from oracle before RL."""
        from agent.llm_agent import OracleAgent
        oracle = OracleAgent()

        if verbose:
            print(f"\n  [Warm-start] Behavioural cloning — {n_episodes} episodes")

        bc_losses = []
        bc_opt    = optim.Adam(self.ac.parameters(), lr=1e-3)

        for ep in range(n_episodes):
            env = env_factory()
            obs = env.reset()
            self.encoder.reset()
            done = False
            ep_states, ep_actions = [], []

            while not done:
                state       = self.encoder.encode(obs)
                action_dict = oracle.act(obs)
                action_idx  = {"hold": 0, "buy": 1, "sell": 2}[action_dict["decision"]]
                ep_states.append(state)
                ep_actions.append(action_idx)
                obs, _, done, info = env.step(action_dict)
                self.encoder.update_from_info(info)

            if ep_states:
                s = torch.FloatTensor(np.stack(ep_states))
                a = torch.LongTensor(ep_actions)
                logits, _ = self.ac(s)
                loss = nn.CrossEntropyLoss()(logits, a)
                bc_opt.zero_grad(); loss.backward(); bc_opt.step()
                bc_losses.append(float(loss.item()))

        avg = float(np.mean(bc_losses)) if bc_losses else 0
        if verbose:
            print(f"  Warm-start complete. Avg BC loss: {avg:.4f}")
        return avg

    # ── PPO update step ───────────────────────────────────────────────────────

    def _ppo_update(self) -> Dict[str, float]:
        """Run PPO update on collected trajectory."""
        states, actions, old_lps, returns, advantages = self.traj.get_arrays()

        S  = torch.FloatTensor(states)
        A  = torch.LongTensor(actions)
        OL = torch.FloatTensor(old_lps)
        R  = torch.FloatTensor(returns)
        ADV= torch.FloatTensor(advantages)

        total_pg, total_vf, total_ent = 0.0, 0.0, 0.0
        n_batches = max(1, len(states) // self.cfg.batch_size)

        for _ in range(self.cfg.n_epochs):
            idx   = torch.randperm(len(states))
            for i in range(n_batches):
                b   = idx[i*self.cfg.batch_size:(i+1)*self.cfg.batch_size]
                if len(b) < 4:
                    continue

                new_lps, values, entropy = self.ac.evaluate(S[b], A[b])
                ratio     = torch.exp(new_lps - OL[b])
                adv_b     = ADV[b]

                # Clipped surrogate loss (policy gradient)
                pg1  = ratio * adv_b
                pg2  = torch.clamp(ratio, 1 - self.cfg.clip_eps, 1 + self.cfg.clip_eps) * adv_b
                pg_l = -torch.min(pg1, pg2).mean()

                # Value loss with clipping
                vf_l = nn.MSELoss()(values, R[b])

                # Entropy bonus
                ent_l = -entropy.mean()

                loss = pg_l + self.cfg.value_coef * vf_l + self.cfg.entropy_coef * ent_l
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.ac.parameters(), self.cfg.grad_clip)
                self.opt.step()

                total_pg  += float(pg_l.item())
                total_vf  += float(vf_l.item())
                total_ent += float(ent_l.item())

        self.traj.clear()
        self.total_updates += 1
        n = n_batches * self.cfg.n_epochs
        return {
            "pg_loss" : total_pg  / n,
            "vf_loss" : total_vf  / n,
            "ent_loss": total_ent / n,
        }

    # ── Collect + train loop ──────────────────────────────────────────────────

    def collect_and_train(self, env_factory) -> Optional[Dict]:
        """Collect steps_per_update steps, then do PPO update if ready."""
        env  = env_factory()
        obs  = env.reset()
        self.encoder.reset()
        done = False

        while not done and self.total_steps % self.cfg.steps_per_update != 0:
            state  = self.encoder.encode(obs)
            in_pos = obs.get("position", "flat") != "flat"
            action_idx, log_prob, value = self.ac.get_action(state, in_pos)

            action_dict = self._idx_to_action(action_idx, obs)
            obs_next, reward, done, info = env.step(action_dict)
            self.encoder.update_from_info(info)

            self.traj.push(state, action_idx, reward, log_prob, value, done)
            self.total_steps += 1
            obs = obs_next

        if not done:
            return None   # keep collecting

        self.ep_count += 1
        summary = env.episode_summary()
        self.ep_pnls.append(summary.get("total_pnl", 0))

        if len(self.traj) >= 10:
            loss_info = self._ppo_update()
            self.update_losses.append(loss_info)
            return {**loss_info, "ep": self.ep_count,
                    "pnl": summary.get("total_pnl", 0),
                    "n_trades": summary.get("n_trades", 0)}
        return None

    def _idx_to_action(self, idx: int, obs: Dict) -> Dict:
        price = obs.get("price", 2300)
        f72   = obs.get("fib_72", price - 20)
        f85   = obs.get("fib_85", price + 20)
        vol   = obs.get("volatility", "medium")
        buf   = {"low": 4, "medium": 8, "high": 15}.get(vol, 8)
        if idx == 0:
            return {"decision": "hold", "stop_loss": 0, "take_profit": 0}
        elif idx == 1:
            sl = f72 - buf
            return {"decision": "buy",  "stop_loss": round(sl,2), "take_profit": round(price+2.2*(price-sl),2)}
        else:
            sl = f85 + buf
            return {"decision": "sell", "stop_loss": round(sl,2), "take_profit": round(price-2.2*(sl-price),2)}

    def train(self, env_factory, n_episodes=500, warmstart_eps=100, log_every=50, verbose=True):
        self.warmstart_from_oracle(env_factory, warmstart_eps, verbose)

        if verbose:
            print(f"\n  [PPO] Training — {n_episodes} episodes")
            print(f"  {'Ep':>6}  {'Avg PnL':>10}  {'PG Loss':>10}  {'VF Loss':>10}")
            print(f"  {'─'*45}")

        for ep in range(1, n_episodes + 1):
            result = self.collect_and_train(env_factory)
            if result and verbose and ep % log_every == 0:
                window = self.ep_pnls[-log_every:]
                print(f"  {ep:>6}  {np.mean(window):>+9.2f}  "
                      f"  {result.get('pg_loss',0):>9.6f}  "
                      f"  {result.get('vf_loss',0):>9.6f}")

        return {
            "total_episodes" : self.ep_count,
            "avg_pnl_last50" : round(float(np.mean(self.ep_pnls[-50:])), 2),
        }

    def save(self, path):
        torch.save({"ac": self.ac.state_dict(), "opt": self.opt.state_dict(),
                    "ep_count": self.ep_count, "ep_pnls": self.ep_pnls}, path)

    def load(self, path):
        ckpt = torch.load(path, map_location="cpu")
        self.ac.load_state_dict(ckpt["ac"])
        self.opt.load_state_dict(ckpt["opt"])
        self.ep_count = ckpt["ep_count"]
        self.ep_pnls  = ckpt.get("ep_pnls", [])
