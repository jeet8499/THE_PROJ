"""agent/llm_agent.py — Oracle agent (importable by trainers)"""
from typing import Any, Dict

class OracleAgent:
    def act(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        trend  = obs.get("true_trend", obs.get("trend", "range"))
        zone   = obs.get("zone_position", "above_zone")
        conf   = obs.get("confirmation", "not_confirmed")
        pos    = obs.get("position", "flat")
        price  = obs.get("price", 2300)
        f72    = obs.get("fib_72", price - 20)
        f85    = obs.get("fib_85", price + 20)
        vol    = obs.get("volatility", "medium")
        buf    = {"low": 4, "medium": 8, "high": 15}.get(vol, 8)

        if pos != "flat":
            return {"decision": "hold", "stop_loss": 0.0, "take_profit": 0.0}
        if zone == "inside_zone" and conf == "confirmed":
            if trend == "bullish":
                sl = f72 - buf
                return {"decision": "buy",  "stop_loss": round(sl,2), "take_profit": round(price+2.2*(price-sl),2)}
            if trend == "bearish":
                sl = f85 + buf
                return {"decision": "sell", "stop_loss": round(sl,2), "take_profit": round(price-2.2*(sl-price),2)}
        return {"decision": "hold", "stop_loss": 0.0, "take_profit": 0.0}
