"""Download the trained RoBERTa sentiment checkpoint from HuggingFace Hub.

TODO (operator action — Hamza, before defense):
    Before any examiner can run this, you must publish the checkpoint:
        1. huggingface-cli login
        2. huggingface-cli upload hamzaziouine/dlp-product-review-roberta \
               models/sentiment_roberta
    The repo name (`hamzaziouine/dlp-product-review-roberta`) is referenced
    below as HF_REPO_ID. Update it if you publish under a different name and
    keep this script in sync.

Usage:
    python download_model.py
        # → downloads into models/sentiment_roberta/

The download is ~500 MB. Idempotent: if the directory already contains
model.safetensors, the script reports "already present" and exits 0.
"""
from __future__ import annotations

import sys
from pathlib import Path

HF_REPO_ID = "hamzaziouine/dlp-product-review-roberta"
LOCAL_DIR = Path(__file__).resolve().parent / "models" / "sentiment_roberta"
WEIGHTS_FILE = "model.safetensors"


def main() -> int:
    LOCAL_DIR.mkdir(parents=True, exist_ok=True)

    if (LOCAL_DIR / WEIGHTS_FILE).is_file():
        print(f"Checkpoint already present at {LOCAL_DIR} — skipping download.")
        return 0

    try:
        from huggingface_hub import snapshot_download  # type: ignore
    except ImportError:
        print(
            "ERROR: huggingface_hub is not installed.\n"
            "Install with:  pip install huggingface_hub\n"
            "Or install all project deps:  pip install -r requirements.lock",
            file=sys.stderr,
        )
        return 1

    print(f"Downloading {HF_REPO_ID} into {LOCAL_DIR} ...")
    try:
        snapshot_download(
            repo_id=HF_REPO_ID,
            local_dir=str(LOCAL_DIR),
        )
    except Exception as exc:  # pragma: no cover — network/auth path
        print(
            f"ERROR: download failed: {exc}\n\n"
            "Likely causes:\n"
            f"  1. The repo {HF_REPO_ID!r} does not yet exist on HuggingFace Hub.\n"
            "     the team must upload the checkpoint — see TODO at top of this file.\n"
            "  2. Network unreachable.\n"
            "  3. The repo is private and requires `huggingface-cli login` first.\n"
            "\nFallback: re-train locally — see docs/REPRODUCING.md Path 1 step 7a.",
            file=sys.stderr,
        )
        return 2

    print(f"Done. Checkpoint available at {LOCAL_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
