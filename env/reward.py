"""
env/reward.py  (v2)
===================
RL reward function — shaped on REALIZED PnL, not classification.

Reward components:
  1. Realized PnL (primary)       — normalized to account size
  2. Risk discipline bonus        — good RR ratio, proper SL placement
  3. Patience bonus               — holding through TP hit rewarded
  4. Impulsive trade penalty      — trading against trend/noise
  5. Drawdown penalty             — losing too much equity
  6. Hold penalty                 — sitting flat when signal is clear
  7. Episode-end equity bonus     — overall episode performance
"""

from typing import Any, Dict, Optional

ACCOUNT_SIZE = 10_000.0

# ─────────────────────────────────────────────────────────────────────────────
# Main reward function
# ─────────────────────────────────────────────────────────────────────────────

def compute_step_reward(
    action         : Dict[str, Any],
    bar_result     : Dict[str, Any],
    obs            : Dict[str, Any],
    position_before: Optional[str],
    is_last        : bool,
    position_mgr   : Any,
) -> float:
    """
    Compute reward for one step.

    Returns a float reward. Typical range: [-1.0, +1.0]
    Large trades can push beyond ±1 if RR is excellent.
    """
    reward = 0.0

    decision     = action["decision"]
    sl           = action["stop_loss"]
    tp           = action["take_profit"]
    price        = obs["price"]
    trend        = obs["trend"]
    confirmation = obs["confirmation"]
    zone_pos     = obs["zone_position"]
    volatility   = obs["volatility"]
    noise        = obs.get("noise_injected", False)

    realized_pnl = bar_result.get("realized_pnl", 0.0)
    sl_hit       = bar_result.get("sl_hit", False)
    tp_hit       = bar_result.get("tp_hit", False)
    was_flat     = position_before is None

    # ─────────────────────────────────────────────────────────────────────────
    # 1. Realized PnL — normalize to ±1 range
    # ─────────────────────────────────────────────────────────────────────────
    if abs(realized_pnl) > 0.001:
        reward += realized_pnl / (ACCOUNT_SIZE * 0.02)   # 2% account = ±1.0

    # ─────────────────────────────────────────────────────────────────────────
    # 2. Risk / Reward discipline bonus (on trade entry)
    # ─────────────────────────────────────────────────────────────────────────
    if decision in ("buy", "sell") and was_flat and sl > 0 and tp > 0:
        risk     = abs(price - sl)
        reward_r = abs(tp - price)
        rr       = reward_r / risk if risk > 0 else 0

        if rr >= 2.0:
            reward += 0.05    # good RR
        elif rr >= 1.5:
            reward += 0.02
        elif rr < 1.0:
            reward -= 0.08    # bad RR — punish

        # SL placement sanity: SL inside fib zone = bad
        fib_72, fib_85 = obs["fib_72"], obs["fib_85"]
        if decision == "sell" and sl < fib_85:
            reward -= 0.05    # SL too tight for sell
        if decision == "buy"  and sl > fib_72:
            reward -= 0.05    # SL too tight for buy

    # ─────────────────────────────────────────────────────────────────────────
    # 3. TP hit bonus (patience rewarded)
    # ─────────────────────────────────────────────────────────────────────────
    if tp_hit:
        reward += 0.10

    # ─────────────────────────────────────────────────────────────────────────
    # 4. SL hit penalty (bad entry or SL too tight)
    # ─────────────────────────────────────────────────────────────────────────
    if sl_hit:
        reward -= 0.08

    # ─────────────────────────────────────────────────────────────────────────
    # 5. Impulsive trade penalty — buying/selling against signal
    # ─────────────────────────────────────────────────────────────────────────
    if decision in ("buy", "sell") and was_flat:
        against_trend = (
            (decision == "buy"  and trend == "bearish") or
            (decision == "sell" and trend == "bullish")
        )
        if against_trend:
            reward -= 0.12

        if confirmation == "not_confirmed":
            reward -= 0.08

        if zone_pos != "inside_zone":
            reward -= 0.06

        # Entering in high volatility without confirmation = extra risky
        if volatility == "high" and confirmation == "not_confirmed":
            reward -= 0.06

    # ─────────────────────────────────────────────────────────────────────────
    # 6. Drawdown penalty
    # ─────────────────────────────────────────────────────────────────────────
    dd = position_mgr.max_drawdown
    if dd > 0.05:
        reward -= dd * 0.5    # proportional penalty

    # ─────────────────────────────────────────────────────────────────────────
    # 7. Missed opportunity penalty — clear signal but chose HOLD while flat
    # ─────────────────────────────────────────────────────────────────────────
    if decision == "hold" and was_flat:
        clear_bull = trend == "bullish" and confirmation == "confirmed" and zone_pos == "inside_zone"
        clear_bear = trend == "bearish" and confirmation == "confirmed" and zone_pos == "inside_zone"
        if clear_bull or clear_bear:
            reward -= 0.04    # gentle nudge: "you should have acted"

    # ─────────────────────────────────────────────────────────────────────────
    # 8. Episode-end bonus — overall equity performance
    # ─────────────────────────────────────────────────────────────────────────
    if is_last:
        pnl_pct = position_mgr.total_pnl / ACCOUNT_SIZE
        if pnl_pct > 0.01:       # > 1% gain
            reward += pnl_pct * 2
        elif pnl_pct < -0.02:    # > 2% loss
            reward += pnl_pct    # negative

    return round(reward, 6)


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: episode-level score (0–1 scale, for benchmarking)
# ─────────────────────────────────────────────────────────────────────────────

def episode_score(summary: Dict[str, Any]) -> float:
    """
    Normalize episode performance to 0–1 for comparison.

    Designed so: Oracle > Random > Always-hold
    Key insight: random agents have high PnL variance + low RR quality.
    We reward consistency (positive PnL) + quality (RR ratio) over raw PnL.
    """
    pnl_pct  = summary.get("pnl_pct", 0.0)
    win_rate = summary.get("win_rate", 0.0) / 100
    avg_rr   = summary.get("avg_rr", 0.0)
    dd       = summary.get("max_drawdown", 0.0) / 100
    n_trades = summary.get("n_trades", 0)

    # No trades = flat base (patient, not penalized, not rewarded)
    if n_trades == 0:
        return 0.35

    # PnL sign matters more than magnitude for stability
    # Positive PnL: score scales up with quality
    # Negative PnL: score is capped low regardless of magnitude
    pnl_positive = pnl_pct > 0

    # Sharpe-proxy: reward positive PnL that came from disciplined RR
    rr_quality = max(0.0, min(1.0, avg_rr / 2.5))          # 0..2.5+ RR → 0..1
    dd_penalty = max(0.0, 1.0 - dd * 6)                     # 16% DD → 0.0

    if pnl_positive:
        # Disciplined profitable trade
        pnl_component = min(1.0, (pnl_pct / 2.0))           # 2% gain = 1.0
        score = (
            0.40 * pnl_component +
            0.30 * win_rate      +
            0.20 * rr_quality    +
            0.10 * dd_penalty
        )
        # Bonus for both profitable AND good RR (oracle hallmark)
        if rr_quality > 0.7 and win_rate > 0.4:
            score += 0.08
    else:
        # Losing trade — cap score, but give partial credit for good RR attempt
        score = (
            0.10 * rr_quality +
            0.05 * dd_penalty
        )

    return round(min(1.0, max(0.0, score)), 4)
