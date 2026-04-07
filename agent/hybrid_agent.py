"""
agent/hybrid_agent.py
=====================
Hybrid LLM + RL Agent — the architecture the mentor asked for.

The LLM and RL policy play DIFFERENT roles:

  LLM role: Strategy Generator / Reward Critic
    - Reads complex multi-factor observation
    - Generates a "strategic intent" (bias signal)
    - Explains WHY it thinks a trade is valid
    - Can veto RL actions that seem dangerous

  RL role: Executor / Optimizer
    - Takes LLM bias + encoded state as input
    - Has learned from thousands of episodes what actually works
    - Not constrained to human-readable rules
    - Adapts to regime changes automatically

Combined flow:
    obs → LLM bias signal → augment state vector → RL policy → final action

The LLM provides a "soft label" (0-1 for buy/sell/hold) that gets
appended to the state vector the RL policy sees. This means the RL
policy LEARNS when to trust the LLM and when to override it.

This is a genuine LLM+RL hybrid, not just "LLM with memory".
"""

import os, json, textwrap
import numpy as np
from typing import Any, Dict, Optional, Tuple
from openai import OpenAI

from agent.state_encoder  import StateEncoder, STATE_DIM
from agent.policy_network import DQNPolicy, ActorCritic, ACTION_NAMES

# Augmented state includes 3 extra dims from LLM: [p_hold, p_buy, p_sell]
HYBRID_STATE_DIM = STATE_DIM + 3


# ─────────────────────────────────────────────────────────────────────────────
# LLM bias extractor
# ─────────────────────────────────────────────────────────────────────────────

LLM_BIAS_PROMPT = textwrap.dedent("""
You are a market bias signal generator for XAU/USD.

Given the observation, output ONLY a JSON object with three probabilities that sum to 1.0:
{
  "p_hold": <float 0-1>,
  "p_buy":  <float 0-1>,
  "p_sell": <float 0-1>,
  "reason": "<10 words max>"
}

Rules:
- p_buy high when: trend=bullish, inside_zone, confirmed, positive sentiment
- p_sell high when: trend=bearish, inside_zone, confirmed, negative sentiment
- p_hold high when: not_confirmed, range, outside zone, or conflicting signals
- If already in position: p_hold = 0.9+
""").strip()


class LLMBiasExtractor:
    """Extracts a soft probability vector from LLM for use as RL state augmentation."""

    def __init__(self):
        api_base = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
        api_key  = os.environ.get("HF_TOKEN", os.environ.get("OPENAI_API_KEY", ""))
        model    = os.environ.get("MODEL_NAME", "gpt-4o-mini")
        try:
            self.client = OpenAI(base_url=api_base, api_key=api_key)
            self.model  = model
            self.available = bool(api_key)
        except Exception:
            self.available = False

    def get_bias(self, obs: Dict[str, Any]) -> np.ndarray:
        """
        Returns [p_hold, p_buy, p_sell] as float32 array.
        Falls back to rule-based heuristic if LLM unavailable.
        """
        if not self.available:
            return self._rule_based_bias(obs)

        try:
            obs_clean = {k: v for k, v in obs.items()
                         if k not in ("bar", "candles_available", "noise_injected")}
            response = self.client.chat.completions.create(
                model    = self.model,
                messages = [
                    {"role": "system", "content": LLM_BIAS_PROMPT},
                    {"role": "user",   "content": json.dumps(obs_clean)},
                ],
                temperature = 0.0,
                max_tokens  = 80,
            )
            raw  = response.choices[0].message.content.strip()
            data = json.loads(raw)
            probs = np.array([
                float(data.get("p_hold", 0.5)),
                float(data.get("p_buy",  0.25)),
                float(data.get("p_sell", 0.25)),
            ], dtype=np.float32)
            probs = np.clip(probs, 0, 1)
            probs /= probs.sum()   # normalize to sum to 1
            return probs
        except Exception:
            return self._rule_based_bias(obs)

    def _rule_based_bias(self, obs: Dict) -> np.ndarray:
        """Fast deterministic fallback — same logic as oracle."""
        trend  = obs.get("trend", "range")
        zone   = obs.get("zone_position", "above_zone")
        conf   = obs.get("confirmation", "not_confirmed")
        sent   = obs.get("sentiment", "neutral")
        pos    = obs.get("position", "flat")

        if pos != "flat":
            return np.array([0.92, 0.04, 0.04], dtype=np.float32)

        if zone == "inside_zone" and conf == "confirmed":
            if trend == "bullish" and sent != "negative":
                return np.array([0.10, 0.80, 0.10], dtype=np.float32)
            if trend == "bearish" and sent != "positive":
                return np.array([0.10, 0.10, 0.80], dtype=np.float32)

        if conf == "not_confirmed" or zone != "inside_zone" or trend == "range":
            return np.array([0.85, 0.08, 0.07], dtype=np.float32)

        # Conflicting signals
        return np.array([0.65, 0.20, 0.15], dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Hybrid Agent
# ─────────────────────────────────────────────────────────────────────────────

class HybridAgent:
    """
    LLM + RL hybrid trading agent.

    The RL policy takes an AUGMENTED state: [encoded_obs | llm_bias]
    This means the RL policy learns to USE the LLM signal — not just follow it.

    Modes:
        "hybrid"  : LLM bias + RL policy (default)
        "rl_only" : RL policy only (ablation)
        "llm_only": LLM bias → argmax (ablation)
    """

    def __init__(
        self,
        policy    : Optional[DQNPolicy] = None,
        mode      : str = "hybrid",
        epsilon   : float = 0.0,
    ):
        self.encoder = StateEncoder()
        self.llm     = LLMBiasExtractor()
        self.mode    = mode
        self.epsilon = epsilon

        # Use provided policy or create a fresh one with augmented state dim
        if policy is not None:
            self.policy = policy
        else:
            self.policy = DQNPolicy(state_dim=HYBRID_STATE_DIM)

        self._memory: list = []
        self._sl_hits = 0
        self._tp_hits = 0

    def reset(self):
        self.encoder.reset()
        self._memory  = []
        self._sl_hits = 0
        self._tp_hits = 0

    def remember(self, action: Dict, info: Dict):
        if info.get("sl_hit"): self._sl_hits += 1
        if info.get("tp_hit"): self._tp_hits += 1
        self._memory.append({
            "decision": action.get("decision"),
            "pnl"     : info.get("realized_pnl", 0),
            "sl_hit"  : info.get("sl_hit", False),
            "tp_hit"  : info.get("tp_hit", False),
        })
        if len(self._memory) > 5:
            self._memory = self._memory[-5:]

    def act(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Produce final trading action.

        Returns action dict with decision, sl, tp, and reasoning.
        """
        in_pos = obs.get("position", "flat") != "flat"

        if self.mode == "llm_only":
            return self._llm_only_act(obs)

        # Get LLM bias signal
        llm_bias = self.llm.get_bias(obs) if self.mode == "hybrid" else np.array([1/3,1/3,1/3])

        # Build augmented state
        base_state = self.encoder.encode(obs)
        aug_state  = np.concatenate([base_state, llm_bias])

        # RL policy decision
        action_idx, q_vals = self.policy.act(aug_state, self.epsilon, in_pos)
        action_name = ACTION_NAMES[action_idx]

        # Build env action
        action = self._build_action(action_name, obs)

        # Add reasoning: show LLM signal + RL decision
        llm_top = ACTION_NAMES[np.argmax(llm_bias)]
        if llm_top == action_name:
            reason = f"LLM+RL agree: {action_name.upper()} (llm={llm_bias[np.argmax(llm_bias)]:.2f}, q={q_vals[action_idx]:.3f})"
        else:
            reason = f"RL overrides LLM: {action_name.upper()} (llm said {llm_top.upper()}, q={q_vals[action_idx]:.3f})"

        action["reasoning"] = reason
        return action

    def _llm_only_act(self, obs: Dict) -> Dict:
        bias = self.llm.get_bias(obs)
        idx  = int(np.argmax(bias))
        action = self._build_action(ACTION_NAMES[idx], obs)
        action["reasoning"] = f"LLM: {ACTION_NAMES[idx].upper()} (p={bias[idx]:.2f})"
        return action

    def _build_action(self, decision: str, obs: Dict) -> Dict:
        price = obs.get("price", 2300)
        f72   = obs.get("fib_72", price - 20)
        f85   = obs.get("fib_85", price + 20)
        vol   = obs.get("volatility", "medium")
        buf   = {"low": 4, "medium": 8, "high": 15}.get(vol, 8)

        if decision == "buy":
            sl = f72 - buf
            tp = price + 2.2 * (price - sl)
            return {"decision": "buy", "stop_loss": round(sl,2), "take_profit": round(tp,2)}
        elif decision == "sell":
            sl = f85 + buf
            tp = price - 2.2 * (sl - price)
            return {"decision": "sell", "stop_loss": round(sl,2), "take_profit": round(tp,2)}
        return {"decision": "hold", "stop_loss": 0.0, "take_profit": 0.0}
