"""
data/market_generator.py
========================
Procedural XAU/USD market generator — infinite, noisy, realistic.

Replaces the static 15-scenario dataset with a stream of varied episodes:
  - Trending / ranging / reversal regimes
  - Volatility clustering (GARCH-like vol persistence)
  - News sentiment shocks (sudden spikes/drops)
  - Conflicting signals (e.g. bullish structure + bearish sentiment)
  - Incomplete / uncertain observations (some fields randomly degraded)
  - Fibonacci zone auto-computed from recent swing high/low
"""

import random
import math
from typing import Any, Dict, List, Optional, Tuple

GOLD_BASE   = 2300.0   # approximate anchor price
TICK        = 0.25     # minimum price movement

# ─────────────────────────────────────────────────────────────────────────────
# Low-level price engine
# ─────────────────────────────────────────────────────────────────────────────

class PriceEngine:
    """
    Generates a realistic OHLCV candle series using:
      - Geometric Brownian Motion base
      - GARCH(1,1)-style volatility clustering
      - Regime drift (trend / range / reversal)
      - Sentiment shocks injected at random bars
    """

    def __init__(
        self,
        seed: Optional[int] = None,
        n_candles: int = 60,
        regime: Optional[str] = None,
    ):
        self.rng = random.Random(seed)
        self.n   = n_candles

        # Regime: bullish | bearish | range | reversal
        self.regime = regime or self.rng.choice(
            ["bullish", "bullish", "bearish", "bearish", "range", "reversal"]
        )

        # Base price ±15% from anchor
        self.start_price = GOLD_BASE * (0.85 + self.rng.random() * 0.30)

        # Vol parameters
        self.base_vol   = self.rng.uniform(0.0008, 0.0025)   # daily vol
        self.vol        = self.base_vol
        self.alpha      = 0.15   # ARCH coefficient
        self.beta       = 0.80   # GARCH coefficient

        # Drift by regime
        drift_map = {
            "bullish"  : self.rng.uniform( 0.0003,  0.0012),
            "bearish"  : self.rng.uniform(-0.0012, -0.0003),
            "range"    : self.rng.uniform(-0.0001,  0.0001),
            "reversal" : None,   # handled separately
        }
        self.drift = drift_map[self.regime]

        # Sentiment shock: random bar, random magnitude
        self.shock_bar = self.rng.randint(5, n_candles - 5)
        self.shock_mag = self.rng.choice([-1, 1]) * self.rng.uniform(0.003, 0.012)

    # ── GARCH vol update ──────────────────────────────────────────────────────

    def _update_vol(self, ret: float) -> None:
        self.vol = math.sqrt(
            self.base_vol**2 * (1 - self.alpha - self.beta)
            + self.alpha * ret**2
            + self.beta * self.vol**2
        )
        self.vol = max(0.0003, min(self.vol, 0.008))

    # ── Single candle ─────────────────────────────────────────────────────────

    def _candle(self, price: float, bar: int) -> Dict[str, float]:
        # Reversal: bullish first half, bearish second
        if self.regime == "reversal":
            drift = 0.0006 if bar < self.n // 2 else -0.0006
        else:
            drift = self.drift  # type: ignore[assignment]

        # Inject sentiment shock
        shock = self.shock_mag if bar == self.shock_bar else 0.0

        ret    = drift + self.rng.gauss(0, self.vol) + shock
        close  = round(max(price * (1 + ret), 1.0) / TICK) * TICK
        self._update_vol(ret)

        # Build OHLC around close
        wick   = abs(close - price) * self.rng.uniform(0.5, 2.5)
        high   = round((max(price, close) + wick * self.rng.random()) / TICK) * TICK
        low    = round((min(price, close) - wick * self.rng.random()) / TICK) * TICK
        volume = int(self.rng.uniform(800, 4000) * (1 + abs(ret) / self.base_vol))

        return {"open": price, "high": high, "low": low, "close": close, "volume": volume}

    # ── Full series ───────────────────────────────────────────────────────────

    def generate(self) -> List[Dict[str, float]]:
        candles = []
        price   = self.start_price
        for i in range(self.n):
            c = self._candle(price, i)
            candles.append(c)
            price = c["close"]
        return candles


# ─────────────────────────────────────────────────────────────────────────────
# Fibonacci zone
# ─────────────────────────────────────────────────────────────────────────────

def compute_fibonacci_zone(
    candles: List[Dict[str, float]],
    lookback: int = 20,
) -> Tuple[float, float]:
    """
    Compute 72%–85% Fibonacci retracement zone from recent swing high/low.
    Returns (fib_72, fib_85) — fib_72 < fib_85 always.
    """
    recent = candles[-lookback:]
    swing_high = max(c["high"]  for c in recent)
    swing_low  = min(c["low"]   for c in recent)
    span = swing_high - swing_low

    fib_72 = swing_low  + 0.72 * span
    fib_85 = swing_low  + 0.85 * span
    return round(fib_72, 2), round(fib_85, 2)


# ─────────────────────────────────────────────────────────────────────────────
# Noisy observation builder
# ─────────────────────────────────────────────────────────────────────────────

_SENTIMENTS  = ["positive", "negative", "neutral"]
_VOLATILITIES = ["low", "medium", "high"]
_CONFIRMS    = ["confirmed", "not_confirmed"]


def _vol_label(vol: float) -> str:
    if vol < 0.001:  return "low"
    if vol < 0.002:  return "medium"
    return "high"


def _zone_position(price: float, fib_72: float, fib_85: float) -> str:
    if fib_72 <= price <= fib_85:  return "inside_zone"
    if price > fib_85:             return "above_zone"
    return "below_zone"


def build_observation(
    candles: List[Dict[str, float]],
    engine: PriceEngine,
    bar: int,
    noise_level: float = 0.3,   # 0 = clean, 1 = max noise
    rng: Optional[random.Random] = None,
) -> Dict[str, Any]:
    """
    Build a noisy observation dict from candle history up to `bar`.

    Noise mechanisms:
      1. Sentiment can conflict with structural trend (probability = noise_level)
      2. Confirmation can be randomly degraded to not_confirmed
      3. Volatility label can be off-by-one tier
      4. Fib levels have small jitter (± 0.3%)
    """
    if rng is None:
        rng = random.Random()

    history  = candles[:bar + 1]
    current  = history[-1]
    price    = round(current["close"], 2)
    fib_72, fib_85 = compute_fibonacci_zone(history)

    # ── Structural trend from last 10 bars ────────────────────────────────────
    if len(history) >= 10:
        trend_closes = [c["close"] for c in history[-10:]]
        slope = (trend_closes[-1] - trend_closes[0]) / trend_closes[0]
        if slope >  0.004: structural_trend = "bullish"
        elif slope < -0.004: structural_trend = "bearish"
        else: structural_trend = "range"
    else:
        structural_trend = engine.regime if engine.regime != "reversal" else "range"

    # ── True sentiment aligned with trend ────────────────────────────────────
    true_sentiment = {
        "bullish" : "positive",
        "bearish" : "negative",
        "range"   : "neutral",
    }.get(structural_trend, "neutral")

    # ── Noise injections ──────────────────────────────────────────────────────

    # 1. Sentiment conflict (most common real-world confusion)
    if rng.random() < noise_level * 0.7:
        sentiment = rng.choice(_SENTIMENTS)         # conflicting news
    else:
        sentiment = true_sentiment

    # 2. Confirmation noise
    if rng.random() < noise_level * 0.5:
        confirmation = rng.choice(_CONFIRMS)
    else:
        # True confirmation: inside zone + vol not extreme
        vol_val = engine.vol
        true_confirm = (
            "confirmed"
            if _zone_position(price, fib_72, fib_85) == "inside_zone"
               and vol_val < 0.004
            else "not_confirmed"
        )
        confirmation = true_confirm

    # 3. Volatility label noise (off by one tier sometimes)
    true_vol = _vol_label(engine.vol)
    if rng.random() < noise_level * 0.3:
        vol_label = rng.choice(_VOLATILITIES)
    else:
        vol_label = true_vol

    # 4. Fib jitter
    jitter = price * 0.003 * noise_level
    fib_72_noisy = round(fib_72 + rng.uniform(-jitter, jitter), 2)
    fib_85_noisy = round(fib_85 + rng.uniform(-jitter, jitter), 2)
    fib_72_noisy, fib_85_noisy = min(fib_72_noisy, fib_85_noisy), max(fib_72_noisy, fib_85_noisy)

    return {
        "pair"          : "XAU/USD",
        "price"         : price,
        "trend"         : structural_trend,
        "fib_72"        : fib_72_noisy,
        "fib_85"        : fib_85_noisy,
        "zone_position" : _zone_position(price, fib_72_noisy, fib_85_noisy),
        "sentiment"     : sentiment,
        "volatility"    : vol_label,
        "confirmation"  : confirmation,
        # Extra context for RL agent
        "bar"           : bar,
        "candles_available": len(history),
        "true_trend"    : structural_trend,   # agent can't see this directly
        "noise_injected": noise_level > 0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Public API: generate a full episode's worth of data
# ─────────────────────────────────────────────────────────────────────────────

def generate_episode(
    seed: Optional[int] = None,
    n_candles: int = 60,
    episode_len: int = 10,
    noise_level: float = 0.3,
    regime: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Generate one full trading episode:
      - n_candles of price history (warm-up)
      - episode_len decision points (bars where agent must act)

    Returns dict with full candle series + per-step observations.
    """
    rng    = random.Random(seed)
    engine = PriceEngine(seed=seed, n_candles=n_candles + episode_len, regime=regime)
    all_candles = engine.generate()

    # Warm-up candles (context), then episode bars
    warm_candles = all_candles[:n_candles]
    ep_candles   = all_candles[n_candles : n_candles + episode_len]

    steps = []
    for i, candle in enumerate(ep_candles):
        bar     = n_candles + i
        obs     = build_observation(all_candles, engine, bar, noise_level, rng)
        next_c  = all_candles[bar + 1] if bar + 1 < len(all_candles) else candle
        steps.append({
            "bar"       : bar,
            "candle"    : candle,
            "next_open" : next_c["open"],
            "observation": obs,
        })

    return {
        "regime"      : engine.regime,
        "noise_level" : noise_level,
        "warm_candles": warm_candles,
        "steps"       : steps,
        "start_price" : warm_candles[0]["open"],
        "end_price"   : ep_candles[-1]["close"] if ep_candles else warm_candles[-1]["close"],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Bulk dataset generation (for benchmark / eval)
# ─────────────────────────────────────────────────────────────────────────────

def generate_dataset(
    n_episodes: int = 500,
    episode_len: int = 10,
    noise_level: float = 0.3,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Generate a reproducible dataset of N episodes across all regimes.
    noise_level varies per episode for diversity.
    """
    rng = random.Random(seed)
    regimes = ["bullish", "bearish", "range", "reversal"]
    episodes = []

    for i in range(n_episodes):
        ep_seed    = rng.randint(0, 999_999)
        regime     = regimes[i % len(regimes)]
        ep_noise   = rng.uniform(0.1, 0.5)   # vary noise per episode
        ep = generate_episode(
            seed=ep_seed,
            episode_len=episode_len,
            noise_level=ep_noise,
            regime=regime,
        )
        ep["episode_id"] = f"ep_{i:04d}"
        ep["seed"]       = ep_seed
        episodes.append(ep)

    return episodes
