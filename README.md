# Multi-Agent Product Review Intelligence

A hierarchical multi-agent AI system that analyzes a product niche by combining web market research with sentiment analysis of customer reviews, then produces a synthesized intelligence report with a human-in-the-loop approval step.

Built as the S8 Integrated Project for the *Intelligence Artificielle & Big Data* track at **Université Internationale de Rabat (UIR)**, 2025-2026.

> **Status:** Complete and defended (May 2026). RoBERTa sentiment model: 80.83% test accuracy / macro-F1 0.7686. CrewAI hierarchical pipeline operational with HITL approval gate. See [`docs/REPRODUCING.md`](docs/REPRODUCING.md) for reproduction steps.

## Architecture

```
                          ┌──────────────────────────────┐
                          │  Manager Agent (Gemini 2.5)  │
                          │  Coordinator — never answers  │
                          │  questions itself, only       │
                          │  delegates and synthesizes.   │
                          └──────────────┬───────────────┘
                                         │
                  ┌──────────────────────┼──────────────────────┐
                  │                                             │
        ┌─────────▼──────────┐                       ┌─────────▼──────────┐
        │  Market Research   │                       │  Sentiment Analyst │
        │  Agent             │                       │  Agent             │
        │  • search_market   │                       │  • load_reviews    │
        │    (DuckDuckGo)    │                       │  • analyze_sentiment│
        └────────────────────┘                       │    (RoBERTa-base)  │
                                                     └────────────────────┘
                                         │
                                         ▼
                            Synthesis task — HITL pause
                            (human approves before final
                             report is written to outputs/)
```

Three components by spec: two specialized agents + one implicit orchestrator (CrewAI `Process.hierarchical` with custom `manager_agent`).

## Model card

| Property | Value |
|---|---|
| Architecture | RoBERTa-base (125M params) — selected over DistilBERT (67M) by ablation |
| Task | 3-class sentiment (NEGATIVE / NEUTRAL / POSITIVE) |
| Training data | 45,000 balanced Amazon Reviews 2018 (Electronics + All Beauty 5-core) |
| Test set | 13,306 held-out reviews |
| **Test accuracy** | **80.83%** |
| **Macro-F1** | **0.7686** |
| Per-class F1 | NEG 0.81 · NEU 0.58 · POS 0.90 |
| Training | 4 epochs, fp16, bs=8, grad_accum=4, lr=2e-5, RTX 3060 6GB |
| Confidence threshold | 0.6 — predictions below are tagged `UNCERTAIN` |

The NEUTRAL F1 of 0.58 is a documented bottleneck: 3-star Amazon reviews carry irreducible label ambiguity (human inter-annotator agreement on this slice is ~60-70% in the literature). See `report/sections/07_resultats.md` and `docs/ablation_comparison.md` for the full ablation table, including a class-weighted-loss negative result.

## Stack

- **Python 3.11** (conda env, NOT global 3.14 — wheel availability)
- **PyTorch 2.6 + CUDA 12.4** (RTX 3060 fp16 training)
- **HuggingFace Transformers** — fine-tuning + tokenizer + Trainer
- **CrewAI 1.14.4** — hierarchical multi-agent orchestration
- **Gemini 2.5 Flash** as the LLM backend
- **DuckDuckGo search** (`duckduckgo-search`) — keyless web search for the market research agent
- Pytest, JSON Lines logging, dotenv, dataclass-style configs

## Quickstart

```bash
# 1. Set up environment (Python 3.11)
python -m venv .venv
source .venv/Scripts/activate          # or .venv\Scripts\Activate.ps1 on Windows
pip install -r requirements.txt

# 2. Configure secrets
cp .env.example .env
# edit .env: set GEMINI_API_KEY=AIza...   (get one at aistudio.google.com)

# 3. Train the sentiment model (~30 min on RTX 3060)
PYTHONPATH=. python src/models/train.py --model-name roberta-base \
    --batch-size 8 --gradient-accumulation-steps 4 --max-length 256 \
    --output-dir models/sentiment_roberta

# 4. Evaluate on held-out test set
PYTHONPATH=. python src/models/evaluate.py --model-dir models/sentiment_roberta

# 5. Run the full multi-agent pipeline
python -m src.main --niche "wireless earbuds" --category Electronics
```

## Project layout

```
config/
  settings.py            # paths, model name, thresholds
  agents.yaml            # 3 agent specs (role/goal/backstory)
  tasks.yaml             # 3 task specs (description/expected_output/HITL flag)
data/
  preprocess.py          # Amazon Reviews → balanced 3-class splits
  sample/                # 100-row sample CSV for quick demos
  processed/             # train/val/test CSVs (gitignored except split_indices.json)
src/
  agents/                # market_researcher, sentiment_analyst, manager (factory funcs)
  models/                # train.py, evaluate.py, predict.py
  tools/                 # web_search, sentiment, review_loader (CrewAI @tool)
  utils/logger.py        # JSONL agent logger (8 event types, thread-safe)
  main.py                # CLI: --niche --category --smoke --validate-key
report/                  # 12-section academic report (French, Maouia template)
  sections/01_intro..07_resultats.md
  figures/               # confusion matrix, per-class metrics, training curves
docs/
  evaluation_distilbert.md, evaluation_roberta.md, ablation_comparison.md
tests/                   # 30 unit + structure tests (pytest)
```

## Tests

```bash
PYTHONPATH=. python -m pytest tests/ -q
```

## Authors

- **Hamza Ziouine** — System architecture & orchestration
- **Mohamed Nacir** — Agent development & integration
- **Nour ElHouda Taroujena** — ML engineering & documentation

Supervised by Pr. Hakim Hafidi (UIR S8), 2025-2026. See [`AUTHORS.md`](AUTHORS.md) for full contribution details.

## License

Released under the MIT License — see [`LICENSE`](LICENSE).
