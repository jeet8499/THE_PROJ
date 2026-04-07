"""
train.py — Main training entry point
=====================================

Usage:
    python train.py --algo dqn --episodes 500 --pretrain 100
    python train.py --algo ppo --episodes 300 --pretrain 50
    python train.py --algo dqn --eval --checkpoint checkpoints/dqn_ep500.pt

What this script does:
  1. Creates trading environment factory
  2. Runs oracle imitation pre-training (behavioural cloning)
  3. Runs RL fine-tuning (DQN or PPO)
  4. Evaluates trained policy vs oracle and random baselines
  5. Saves checkpoint
"""

import argparse, os, sys, json
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from env.trading_env    import TradingEnv
from env.reward         import episode_score
from agent.state_encoder import StateEncoder
from agent.policy_network import DQNPolicy, ACTION_NAMES

# Lazy import HYBRID_STATE_DIM
from agent.hybrid_agent import HYBRID_STATE_DIM


def make_env(difficulty="medium", episode_len=20, seed=None):
    """Factory function — called fresh each episode."""
    return TradingEnv(difficulty=difficulty, episode_len=episode_len, seed=seed)


def evaluate_policy(policy, n_episodes=50, difficulty="medium", verbose=False):
    """Evaluate a trained DQN policy. Returns avg score and avg pnl."""
    encoder = StateEncoder()
    from agent.hybrid_agent import LLMBiasExtractor
    llm = LLMBiasExtractor()
    scores, pnls = [], []

    for ep in range(n_episodes):
        env = make_env(difficulty=difficulty)
        obs, _ = env.reset()
        encoder.reset()
        done = False

        while not done:
            base  = encoder.encode(obs)
            bias  = llm._rule_based_bias(obs)
            state = np.concatenate([base, bias])
            in_pos = obs.get("position", "flat") != "flat"
            action_idx, _ = policy.act(state, epsilon=0.0, in_position=in_pos)

            # Convert to env action
            price = obs.get("price", 2300)
            f72   = obs.get("fib_72", price-20)
            f85   = obs.get("fib_85", price+20)
            vol   = obs.get("volatility", "medium")
            buf   = {"low":4,"medium":8,"high":15}.get(vol,8)
            if action_idx == 0:
                action = {"decision":"hold","stop_loss":0,"take_profit":0}
            elif action_idx == 1:
                sl=f72-buf; action={"decision":"buy","stop_loss":round(sl,2),"take_profit":round(price+2.2*(price-sl),2)}
            else:
                sl=f85+buf; action={"decision":"sell","stop_loss":round(sl,2),"take_profit":round(price-2.2*(sl-price),2)}

            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            encoder.update_from_info(info)

        summary = env.episode_summary()
        scores.append(episode_score(summary))
        pnls.append(summary.get("total_pnl", 0))
        if verbose:
            print(f"  ep{ep+1:03d}  score={scores[-1]:.4f}  pnl={pnls[-1]:+.2f}")

    return {
        "avg_score": round(float(np.mean(scores)), 4),
        "avg_pnl"  : round(float(np.mean(pnls)),   2),
        "min_score": round(float(np.min(scores)),   4),
        "max_score": round(float(np.max(scores)),   4),
    }


def main():
    parser = argparse.ArgumentParser(description="Train RL trading agent")
    parser.add_argument("--algo",       choices=["dqn","ppo"],  default="dqn")
    parser.add_argument("--episodes",   type=int,  default=300)
    parser.add_argument("--pretrain",   type=int,  default=100)
    parser.add_argument("--difficulty", choices=["easy","medium","hard"], default="medium")
    parser.add_argument("--episode-len",type=int,  default=20)
    parser.add_argument("--eval",       action="store_true")
    parser.add_argument("--checkpoint", type=str,  default=None)
    parser.add_argument("--save-dir",   type=str,  default="checkpoints")
    parser.add_argument("--seed",       type=int,  default=42)
    parser.add_argument("--quiet",      action="store_true")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    verbose = not args.quiet
    env_factory = lambda: make_env(args.difficulty, args.episode_len)

    if args.algo == "dqn":
        from training.dqn_trainer import DQNTrainer, DQNConfig
        cfg = DQNConfig()
        trainer = DQNTrainer(cfg)
        if args.checkpoint:
            trainer.load(args.checkpoint)
            print(f"  Loaded checkpoint: {args.checkpoint}")

        if not args.eval:
            result = trainer.train(
                env_factory,
                n_episodes   = args.episodes,
                pretrain_eps = args.pretrain,
                verbose      = verbose,
            )
            ckpt_path = os.path.join(args.save_dir, f"dqn_ep{args.episodes}.pt")
            trainer.save(ckpt_path)
            print(f"\n  Checkpoint saved → {ckpt_path}")
            print(f"  Final stats: {json.dumps(result, indent=2)}")

        print(f"\n  Evaluating trained DQN policy (50 episodes)...")
        eval_result = evaluate_policy(trainer.policy, n_episodes=50, difficulty=args.difficulty)
        print(f"  Eval result: {json.dumps(eval_result, indent=2)}")

    elif args.algo == "ppo":
        from training.ppo_trainer import PPOTrainer, PPOConfig
        cfg = PPOConfig()
        trainer = PPOTrainer(cfg)
        if args.checkpoint:
            trainer.load(args.checkpoint)

        if not args.eval:
            result = trainer.train(
                env_factory,
                n_episodes    = args.episodes,
                warmstart_eps = args.pretrain,
                verbose       = verbose,
            )
            ckpt_path = os.path.join(args.save_dir, f"ppo_ep{args.episodes}.pt")
            trainer.save(ckpt_path)
            print(f"\n  Checkpoint saved → {ckpt_path}")


if __name__ == "__main__":
    main()
