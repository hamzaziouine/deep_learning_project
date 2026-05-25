# Reproducing the project from scratch

This document describes the exact commands required to reproduce the project starting from a fresh `git clone`. Three reproducibility paths are documented: full development setup (Python 3.11 venv + GPU training), inference-only setup (skip training, use a pre-trained checkpoint), and Docker-based setup (most portable).

## Path 1 — Full development setup (Linux / macOS / Windows + WSL or Git Bash)

```bash
# 1. Clone
git clone https://github.com/hamzaziouine/deep_learning_project.git
cd deep_learning_project

# 2. Python 3.11 — required for crewai + transformers wheel availability.
#    On Windows we used a conda env: C:/Users/<you>/.conda/envs/dlp/python.exe
#    On Linux: pyenv install 3.11.x && pyenv local 3.11.x
python3.11 -m venv .venv
source .venv/bin/activate          # or .venv\Scripts\Activate.ps1 on Windows

# 3. CPU-only torch is fine for inference. For GPU training, install the CUDA wheel:
pip install torch --index-url https://download.pytorch.org/whl/cu124

# Preferred: install pinned versions from the lock file (exact reproducibility):
pip install -r requirements.lock
# Fallback (looser pins) if requirements.lock is unavailable:
#     pip install -r requirements.txt

# 4. Configure the Gemini API key
cp .env.example .env
# Edit .env — set GEMINI_API_KEY=AIza...   (free-tier key from aistudio.google.com)

# 5. Sanity check
PYTHONPATH=. python -m pytest tests/ -q
# Expected: 28 passed, 1 deselected (smoke test gated by marker)

# 6. Validate the Gemini key
PYTHONPATH=. python -m src.main --validate-key
# Expected: Gemini key OK.

# 7a. (Training path) Download + preprocess Amazon Reviews 2018, then train RoBERTa
PYTHONPATH=. python data/download_data.py
PYTHONPATH=. python data/preprocess.py
PYTHONPATH=. python src/models/train.py \
    --model-name roberta-base \
    --batch-size 8 --gradient-accumulation-steps 4 \
    --max-length 256 \
    --output-dir models/sentiment_roberta
# ~30 min on RTX 3060 6GB

# 7b. (Inference-only path) Skip training and download a pre-trained checkpoint
#     (placeholder — operator to publish to HuggingFace Hub or Google Drive)

# 8. Evaluate
PYTHONPATH=. python src/models/evaluate.py --model-dir models/sentiment_roberta
# Writes docs/evaluation.md + report/figures/{confusion_matrix,per_class_metrics,training_curves}.png

# 9. Run the full multi-agent pipeline
PYTHONPATH=. python -m src.main --niche "wireless earbuds" --category Electronics
# Pauses at the HITL synthesis step — type "approve" or feedback then Enter.
# Writes outputs/<niche>_<timestamp>.md and logs/agent_<timestamp>.jsonl
```

### Common pitfalls

1. **Torch installs CPU-only by default**: without `--index-url https://download.pytorch.org/whl/cu124`, pip picks the CPU wheel even on a CUDA host. Training will run but be ~10× slower.
2. **Cloud-sync folders interfere with `.venv/`**: if the repo lives in a cloud-synced folder (OneDrive, Dropbox, Google Drive, etc.), the sync placeholder mechanism may temporarily make torch DLLs inaccessible. Pause sync during training, or move the repo to a non-synced location.
3. **HuggingFace symlink warning on Windows**: the cache emits a "symlinks not supported" warning. Harmless. To silence: enable Windows Developer Mode or run as administrator.
4. **DeBERTa-v3 sentencepiece error**: `microsoft/deberta-v3-base` requires `sentencepiece` package and a compatible tokenizer model. We attempted DeBERTa as an extra ablation and abandoned due to a tiktoken/spm download bug in our environment. RoBERTa-base is the supported architecture.
5. **Gemini free-tier quota**: 20 requests/day for `gemini-2.5-flash`. A single `--niche` run consumes ~10 requests. Plan accordingly. `LITELLM_CACHE=disk` (set in `src/main.py`) caches identical prompts to avoid burning quota.

## Path 2 — Docker-based setup

```bash
docker build -t dlp:latest .

# Run the validation entrypoint
docker run --rm -it \
    -v $(pwd)/.env:/app/.env \
    dlp:latest

# Run the full pipeline (mount models + .env)
docker run --rm -it \
    -v $(pwd)/models:/app/models \
    -v $(pwd)/.env:/app/.env \
    -v $(pwd)/outputs:/app/outputs \
    -v $(pwd)/logs:/app/logs \
    dlp:latest python -m src.main --niche "wireless earbuds" --category Electronics
```

Docker image is CPU-only inference. For training, run on host (Path 1) since GPU support requires `--gpus all` plus the matching CUDA base image, which is out of scope for this Dockerfile.

## Path 3 — Examiner shortcut (no training required)

If the examiner only wants to verify the orchestration without re-training:

```bash
# After Path 1 step 6 (sanity check + key validation):

# Fetch the trained RoBERTa checkpoint (~500 MB) from HuggingFace Hub.
python download_model.py

# Smoke-test the sentiment tool against the downloaded checkpoint.
PYTHONPATH=. python src/models/predict.py "This product is fantastic!"
```

`download_model.py` pulls `hamzaziouine/dlp-product-review-roberta` into
`models/sentiment_roberta/`. The download is idempotent: re-running the
script when the checkpoint is already present is a no-op.

If the download fails because the HF repo isn't published yet, the script
prints a clear error pointing the team to the upload step. The fallback
in that case is to re-train locally — see Path 1 step 7a.

### Replay mode (no LLM, no checkpoint, no network)

For a fully offline demo (defense-time fallback when Gemini quota is exhausted
or Wi-Fi fails):

```bash
PYTHONPATH=. python -m src.main --replay demo/wireless_earbuds_demo
```

This prints the saved canonical run (`demo/wireless_earbuds_demo.md` plus the
JSONL log) without making any LLM calls.

## Determinism notes

- The RoBERTa fine-tuning is fully deterministic at seed 42 (set in `src/models/train.py:39`). Re-training reproduces 80.83% test accuracy to ±0.1 pp. The DistilBERT ablation baseline (79.59%) reproduces with `--model-name distilbert-base-uncased`.
- The CrewAI orchestration via Gemini is **non-deterministic**: two runs on the same niche produce structurally identical reports (same section headers) but different exact wording. The "canonical run" referenced in the academic report (§7) corresponds to the run captured in `outputs/wireless_earbuds_<timestamp>.md` and is the reference for any quoted figures.

## Verifying the build is reproducible

After Path 1 step 9 (full pipeline run), the team should observe:
- `outputs/<niche>_<timestamp>.md` with five sections: Niche Summary, Top Competitors, Sentiment Breakdown, Key Pain Points, Opportunity Recommendation.
- `logs/agent_<timestamp>.jsonl` with at least: 1 `crew_start`, 2-4 `tool_called`, 1 `hitl_checkpoint`, 1 `report_generated`, 1 `crew_complete` events.
- `pytest tests/ -q` returns `28 passed, 1 deselected`.
- `pytest -m smoke tests/test_crew_smoke.py -v` returns `1 passed` (consumes 1 Gemini call).
