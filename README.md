# GoldTrading-XAU/USD-v4 — Hybrid RL + LLM

[![OpenEnv](https://img.shields.io/badge/spec-openenv--v1-gold)]() [![Gymnasium](https://img.shields.io/badge/API-gymnasium-blue)]()

## Three distinct tasks

| Task | Objective | Difficulty | Grader |
|------|-----------|-----------|--------|
| `task_easy` | Direction accuracy only | Easy | EasyGrader |
| `task_medium` | Direction + SL + TP hit | Medium | MediumGrader |
| `task_hard` | Multi-trade, DD<2%, profitable | Hard | HardGrader |

## Quick start
```bash
pip install -r requirements.txt
python -m openenv validate openenv.yaml
python inference.py --task all --episodes 5
```

## Docker
```bash
docker build -t gold-trading-v4 .
docker run -p 7860:7860 gold-trading-v4
curl http://localhost:7860/
```

## Environment variables
```bash
export API_BASE_URL=https://api.openai.com/v1
export MODEL_NAME=gpt-4o-mini
export HF_TOKEN=your-key
# Optional compatibility:
export OPENAI_API_KEY=$HF_TOKEN
python inference.py
```

## Observation space (gymnasium.spaces.Dict — all numeric)
| Field | Type | Description |
|-------|------|-------------|
| price, fib_72, fib_85, equity, unrealized_pnl, total_pnl, max_drawdown | Box(1,) | Numeric values |
| trend_enc | Discrete(3) | 0=bullish 1=bearish 2=range |
| sentiment_enc | Discrete(3) | 0=positive 1=negative 2=neutral |
| volatility_enc | Discrete(3) | 0=low 1=medium 2=high |
| confirmation_enc | Discrete(2) | 0=confirmed 1=not_confirmed |
| zone_enc | Discrete(3) | 0=inside 1=above 2=below |
| position_enc | Discrete(3) | 0=flat 1=long 2=short |
| steps_remaining | Discrete | Steps left in episode |

## Action space
| Field | Values |
|-------|--------|
| decision | Discrete(3): 0=hold 1=buy 2=sell |
| stop_loss | Box: [0, 10000] |
| take_profit | Box: [0, 10000] |

## Baseline scores (5 episodes, rule-based oracle)
| Task | Score |
|------|-------|
| task_easy | ~0.62 |
| task_medium | ~0.48 |
| task_hard | ~0.41 |
| Overall | ~0.50 |

Run `python inference.py --output logs/baseline.json` to reproduce.
The script emits structured stdout blocks using `[START]`, `[STEP]`, and `[END]`.

## Reward function (shaped, 7 components)
realized_pnl + rr_discipline + tp_bonus - sl_penalty - trend_penalty - drawdown_penalty + episode_bonus
