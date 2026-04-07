"""
env/position.py
===============
Real position tracking with PnL, slippage, fees, and trade lifecycle.

This is what was MISSING from v1:
  - position state: None | "long" | "short"
  - running unrealized PnL
  - realized PnL on close
  - slippage model
  - position sizing (risk % of account)
  - max drawdown tracking
"""

from dataclasses import dataclass, field
from typing import List, Optional
import math


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

ACCOUNT_SIZE     = 10_000.0   # USD
RISK_PER_TRADE   = 0.01       # 1% risk per trade
SLIPPAGE_PIPS    = 0.50       # execution slippage in price units
SPREAD           = 0.30       # bid-ask spread
COMMISSION       = 0.02       # per-trade commission as % of position value / 100


# ─────────────────────────────────────────────────────────────────────────────
# Trade record
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Trade:
    direction   : str       # "long" | "short"
    entry_price : float
    stop_loss   : float
    take_profit : float
    entry_bar   : int
    units       : float     # position size in ounces
    exit_price  : Optional[float] = None
    exit_bar    : Optional[int]   = None
    exit_reason : str = ""        # "sl_hit" | "tp_hit" | "manual" | "episode_end"
    realized_pnl: float = 0.0
    open        : bool = True

    @property
    def risk_per_unit(self) -> float:
        return abs(self.entry_price - self.stop_loss)

    @property
    def reward_per_unit(self) -> float:
        return abs(self.take_profit - self.entry_price)

    @property
    def risk_reward(self) -> float:
        if self.risk_per_unit == 0:
            return 0.0
        return self.reward_per_unit / self.risk_per_unit

    def unrealized_pnl(self, current_price: float) -> float:
        if self.direction == "long":
            return (current_price - self.entry_price) * self.units
        else:
            return (self.entry_price - current_price) * self.units

    def close(self, price: float, bar: int, reason: str) -> float:
        # Apply slippage on exit
        if self.direction == "long":
            exit_p = price - SLIPPAGE_PIPS
        else:
            exit_p = price + SLIPPAGE_PIPS

        self.exit_price  = exit_p
        self.exit_bar    = bar
        self.exit_reason = reason
        self.open        = False

        raw_pnl = (
            (exit_p - self.entry_price) * self.units
            if self.direction == "long"
            else (self.entry_price - exit_p) * self.units
        )
        commission = self.entry_price * self.units * COMMISSION / 100
        self.realized_pnl = raw_pnl - commission
        return self.realized_pnl


# ─────────────────────────────────────────────────────────────────────────────
# Position Manager
# ─────────────────────────────────────────────────────────────────────────────

class PositionManager:
    """
    Manages the full trade lifecycle within an episode.

    State:
        position        : None | "long" | "short"
        current_trade   : Trade | None
        equity          : running account value
        peak_equity     : for drawdown tracking
        trade_history   : all closed trades this episode
    """

    def __init__(self, account_size: float = ACCOUNT_SIZE):
        self.initial_equity  = account_size
        self.equity          = account_size
        self.peak_equity     = account_size
        self.position        : Optional[str]   = None
        self.current_trade   : Optional[Trade] = None
        self.trade_history   : List[Trade]     = []
        self._bar            = 0

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def is_flat(self) -> bool:
        return self.position is None

    @property
    def unrealized_pnl(self) -> float:
        if self.current_trade is None:
            return 0.0
        # Use last known price embedded in trade (updated each bar)
        return self.current_trade.unrealized_pnl(self._last_price)

    @property
    def total_pnl(self) -> float:
        return self.equity - self.initial_equity

    @property
    def max_drawdown(self) -> float:
        return (self.peak_equity - self.equity) / self.peak_equity

    @property
    def win_rate(self) -> float:
        closed = [t for t in self.trade_history if not t.open]
        if not closed:
            return 0.0
        wins = sum(1 for t in closed if t.realized_pnl > 0)
        return wins / len(closed)

    @property
    def avg_rr(self) -> float:
        closed = [t for t in self.trade_history if not t.open]
        if not closed:
            return 0.0
        return sum(t.risk_reward for t in closed) / len(closed)

    # ── Open a position ────────────────────────────────────────────────────────

    def open_position(
        self,
        direction: str,
        price: float,
        stop_loss: float,
        take_profit: float,
        bar: int,
    ) -> Optional[Trade]:
        """
        Open a new position. Returns None if:
          - already in a position
          - stop_loss == 0 (invalid)
          - risk per unit == 0
        """
        if not self.is_flat:
            return None
        if stop_loss <= 0 or take_profit <= 0:
            return None

        # Apply entry slippage
        entry = price + SLIPPAGE_PIPS if direction == "long" else price - SLIPPAGE_PIPS
        entry += SPREAD / 2

        risk_per_unit = abs(entry - stop_loss)
        if risk_per_unit < 0.01:
            return None

        # Position sizing: risk 1% of equity
        risk_dollars = self.equity * RISK_PER_TRADE
        units = round(risk_dollars / risk_per_unit, 4)
        units = max(units, 0.001)

        self._last_price = entry
        self.position    = direction
        trade = Trade(
            direction   = direction,
            entry_price = entry,
            stop_loss   = stop_loss,
            take_profit = take_profit,
            entry_bar   = bar,
            units       = units,
        )
        self.current_trade = trade
        return trade

    # ── Update each bar ────────────────────────────────────────────────────────

    def update(
        self,
        candle: dict,
        bar: int,
    ) -> dict:
        """
        Feed next candle. Check SL/TP hits. Returns update dict.
        """
        self._bar        = bar
        self._last_price = candle["close"]
        result = {
            "sl_hit"        : False,
            "tp_hit"        : False,
            "realized_pnl"  : 0.0,
            "unrealized_pnl": 0.0,
            "position"      : self.position,
        }

        if self.current_trade is None or not self.current_trade.open:
            return result

        t = self.current_trade
        h, l = candle["high"], candle["low"]

        # Check SL hit (worst case: wick touched SL)
        sl_hit = (t.direction == "long"  and l <= t.stop_loss) or \
                 (t.direction == "short" and h >= t.stop_loss)

        # Check TP hit
        tp_hit = (t.direction == "long"  and h >= t.take_profit) or \
                 (t.direction == "short" and l <= t.take_profit)

        # Both hit same candle: SL takes precedence (conservative)
        if sl_hit:
            pnl = t.close(t.stop_loss, bar, "sl_hit")
            self._finalize_trade(pnl)
            result.update({"sl_hit": True, "realized_pnl": pnl, "position": None})
        elif tp_hit:
            pnl = t.close(t.take_profit, bar, "tp_hit")
            self._finalize_trade(pnl)
            result.update({"tp_hit": True, "realized_pnl": pnl, "position": None})
        else:
            result["unrealized_pnl"] = t.unrealized_pnl(candle["close"])

        return result

    # ── Close manually ────────────────────────────────────────────────────────

    def close_position(self, price: float, bar: int, reason: str = "manual") -> float:
        if self.current_trade is None or not self.current_trade.open:
            return 0.0
        pnl = self.current_trade.close(price, bar, reason)
        self._finalize_trade(pnl)
        return pnl

    def _finalize_trade(self, pnl: float) -> None:
        self.equity        += pnl
        self.peak_equity    = max(self.peak_equity, self.equity)
        self.position       = None
        if self.current_trade:
            self.trade_history.append(self.current_trade)
        self.current_trade  = None

    # ── Episode-end summary ────────────────────────────────────────────────────

    def summary(self) -> dict:
        closed = [t for t in self.trade_history]
        n      = len(closed)
        return {
            "initial_equity" : self.initial_equity,
            "final_equity"   : round(self.equity, 2),
            "total_pnl"      : round(self.total_pnl, 2),
            "pnl_pct"        : round(self.total_pnl / self.initial_equity * 100, 3),
            "n_trades"       : n,
            "win_rate"       : round(self.win_rate * 100, 1),
            "max_drawdown"   : round(self.max_drawdown * 100, 3),
            "avg_rr"         : round(self.avg_rr, 2),
            "trades"         : [
                {
                    "direction"   : t.direction,
                    "entry"       : t.entry_price,
                    "exit"        : t.exit_price,
                    "sl"          : t.stop_loss,
                    "tp"          : t.take_profit,
                    "pnl"         : round(t.realized_pnl, 2),
                    "exit_reason" : t.exit_reason,
                    "rr"          : round(t.risk_reward, 2),
                }
                for t in closed
            ],
        }

    def reset(self) -> None:
        self.__init__(self.initial_equity)
