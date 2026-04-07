"""
agent/state_encoder.py
======================
Converts raw observation dicts → fixed-size float32 numpy vectors.

This is a critical piece: the neural network needs a fixed-size input.
We encode all categorical fields as one-hot and normalize all floats.

Output vector (STATE_DIM = 32):
  [0]    price (normalized to [0,1] via log-scale over 1000–5000 range)
  [1]    fib_72 (normalized)
  [2]    fib_85 (normalized)
  [3]    zone_spread (fib_85 - fib_72, normalized)
  [4]    price_in_zone (how far price is through the zone, 0=at72, 1=at85)
  [5-7]  trend one-hot: [bullish, bearish, range]
  [8-10] sentiment one-hot: [positive, negative, neutral]
  [11-13] volatility one-hot: [low, medium, high]
  [14-15] confirmation one-hot: [confirmed, not_confirmed]
  [16-18] zone_position one-hot: [inside, above, below]
  [19-21] position one-hot: [flat, long, short]
  [22]   unrealized_pnl (normalized)
  [23]   total_pnl (normalized)
  [24]   max_drawdown (0–1)
  [25]   steps_remaining (normalized 0–1)
  [26]   equity_ratio (current equity / initial equity)
  [27]   sl_hit_rate (recent sl hits / recent trades, from memory)
  [28]   tp_hit_rate (recent tp hits / recent trades, from memory)
  [29]   n_trades_this_ep (normalized)
  [30]   price_momentum (10-bar return, normalized)
  [31]   vol_regime (0=low,0.5=med,1=high mapped from label)
"""

import numpy as np
from typing import Any, Dict, List, Optional

STATE_DIM    = 32
PRICE_MIN    = 1000.0
PRICE_MAX    = 5000.0
ACCOUNT_SIZE = 10_000.0


def _norm_price(p: float) -> float:
    return max(0.0, min(1.0, (p - PRICE_MIN) / (PRICE_MAX - PRICE_MIN)))


def _one_hot(val: str, options: List[str]) -> List[float]:
    return [1.0 if val == o else 0.0 for o in options]


class StateEncoder:
    """
    Converts observation dicts to fixed-size float32 vectors.

    Maintains a short history window for computing momentum and hit rates.
    Reset between episodes.
    """

    def __init__(self, memory_len: int = 5):
        self.memory_len   = memory_len
        self._price_hist  : List[float] = []
        self._sl_hits     : int = 0
        self._tp_hits     : int = 0
        self._n_trades    : int = 0
        self._ep_trades   : int = 0

    def reset(self) -> None:
        self._price_hist = []
        self._sl_hits    = 0
        self._tp_hits    = 0
        self._n_trades   = 0
        self._ep_trades  = 0

    def update_from_info(self, info: Dict[str, Any]) -> None:
        """Call after each env.step() to update running stats."""
        if info.get("sl_hit"):
            self._sl_hits  += 1
            self._n_trades += 1
        if info.get("tp_hit"):
            self._tp_hits  += 1
            self._n_trades += 1
        if info.get("trade_opened"):
            self._ep_trades += 1
        if "price" in info:
            self._price_hist.append(info["price"])
            if len(self._price_hist) > 20:
                self._price_hist = self._price_hist[-20:]

    def encode(self, obs: Dict[str, Any]) -> np.ndarray:
        """Convert observation dict to STATE_DIM float32 vector."""
        vec = np.zeros(STATE_DIM, dtype=np.float32)
        price  = float(obs.get("price", 2300))
        f72    = float(obs.get("fib_72", 2290))
        f85    = float(obs.get("fib_85", 2310))
        spread = f85 - f72

        # [0-3] Price & fib
        vec[0] = _norm_price(price)
        vec[1] = _norm_price(f72)
        vec[2] = _norm_price(f85)
        vec[3] = max(0.0, min(1.0, spread / 100.0))   # normalize spread

        # [4] Position in zone: 0=at f72, 1=at f85, <0=below, >1=above
        if spread > 0:
            vec[4] = max(-0.5, min(1.5, (price - f72) / spread))
        else:
            vec[4] = 0.5

        # [5-7] trend
        trend = str(obs.get("trend", "range"))
        vec[5:8] = _one_hot(trend, ["bullish", "bearish", "range"])

        # [8-10] sentiment
        sent = str(obs.get("sentiment", "neutral"))
        vec[8:11] = _one_hot(sent, ["positive", "negative", "neutral"])

        # [11-13] volatility
        vol = str(obs.get("volatility", "medium"))
        vec[11:14] = _one_hot(vol, ["low", "medium", "high"])

        # [14-15] confirmation
        conf = str(obs.get("confirmation", "not_confirmed"))
        vec[14:16] = _one_hot(conf, ["confirmed", "not_confirmed"])

        # [16-18] zone position
        zone = str(obs.get("zone_position", "above_zone"))
        vec[16:19] = _one_hot(zone, ["inside_zone", "above_zone", "below_zone"])

        # [19-21] position state
        pos = str(obs.get("position", "flat"))
        vec[19:22] = _one_hot(pos, ["flat", "long", "short"])

        # [22] unrealized PnL (normalized to ±1 = ±$500)
        upnl = float(obs.get("unrealized_pnl", 0))
        vec[22] = max(-1.0, min(1.0, upnl / 500.0))

        # [23] total PnL (normalized to ±1 = ±$200)
        tpnl = float(obs.get("total_pnl", 0))
        vec[23] = max(-1.0, min(1.0, tpnl / 200.0))

        # [24] max drawdown
        vec[24] = max(0.0, min(1.0, float(obs.get("max_drawdown", 0)) / 100.0))

        # [25] steps remaining (normalized)
        steps_rem = float(obs.get("steps_remaining", 10))
        vec[25] = max(0.0, min(1.0, steps_rem / 20.0))

        # [26] equity ratio
        equity = float(obs.get("equity", ACCOUNT_SIZE))
        vec[26] = max(0.0, min(2.0, equity / ACCOUNT_SIZE))

        # [27-28] sl/tp hit rates from memory
        if self._n_trades > 0:
            vec[27] = self._sl_hits / self._n_trades
            vec[28] = self._tp_hits / self._n_trades
        else:
            vec[27] = 0.0
            vec[28] = 0.0

        # [29] episode trade count (normalized)
        vec[29] = min(1.0, self._ep_trades / 5.0)

        # [30] price momentum (10-bar return)
        ph = self._price_hist
        if len(ph) >= 10:
            mom = (ph[-1] - ph[-10]) / ph[-10]
            vec[30] = max(-0.1, min(0.1, mom)) / 0.1   # normalize ±10% to ±1
        else:
            vec[30] = 0.0

        # [31] vol regime (numeric)
        vec[31] = {"low": 0.0, "medium": 0.5, "high": 1.0}.get(vol, 0.5)

        return vec
