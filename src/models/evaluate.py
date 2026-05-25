"""Load a trained transformer sentiment checkpoint and evaluate on the held-out test set.

Defaults to the RoBERTa production model (`models/sentiment_roberta/`); pass
`--model-dir models/sentiment_distilbert` to re-run the ablation baseline.

Usage (operator runs after training completes):
    python src/models/evaluate.py [--model-dir PATH] [--batch-size N]

Outputs:
    docs/evaluation.md                  -- metrics + confusion matrix + examples
    report/figures/confusion_matrix.png -- seaborn heatmap
    report/figures/training_curves.png  -- train/val loss + val accuracy per epoch
    report/figures/per_class_metrics.png-- bar chart precision/recall/F1 per class
"""

import argparse
import json
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe for headless runs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import Dataset
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from config.settings import DATA_PROCESSED, SENTIMENT_MODEL_PATH

# ---------------------------------------------------------------------------
# Constants (mirror train.py)
# ---------------------------------------------------------------------------

ID2LABEL = {0: "NEGATIVE", 1: "NEUTRAL", 2: "POSITIVE"}
LABEL_NAMES = ["NEGATIVE", "NEUTRAL", "POSITIVE"]

FIGURES_DIR = Path(__file__).resolve().parents[2] / "report" / "figures"
DOCS_DIR = Path(__file__).resolve().parents[2] / "docs"


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------


def _tokenize_texts(texts: list[str], tokenizer, max_length: int = 256) -> dict:
    return tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        padding=True,
        return_tensors="pt",
    )


def run_inference(
    model,
    tokenizer,
    texts: list[str],
    labels: list[int],
    batch_size: int = 32,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """Run batched inference. Returns (predictions array, probabilities array)."""
    model.eval()
    all_preds = []
    all_probs = []

    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        enc = _tokenize_texts(batch_texts, tokenizer)
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        preds = np.argmax(probs, axis=-1)
        all_preds.append(preds)
        all_probs.append(probs)

    return np.concatenate(all_preds), np.concatenate(all_probs)


# ---------------------------------------------------------------------------
# Baseline classifiers
# ---------------------------------------------------------------------------


def baseline_random(labels: np.ndarray, n_classes: int = 3, seed: int = 42) -> float:
    rng = np.random.default_rng(seed)
    preds = rng.integers(0, n_classes, size=len(labels))
    return accuracy_score(labels, preds)


def baseline_majority(labels: np.ndarray) -> float:
    majority = np.bincount(labels).argmax()
    preds = np.full(len(labels), majority)
    return accuracy_score(labels, preds)


# ---------------------------------------------------------------------------
# Figure: confusion matrix
# ---------------------------------------------------------------------------


def plot_confusion_matrix(cm: np.ndarray, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=LABEL_NAMES,
        yticklabels=LABEL_NAMES,
        ax=ax,
    )
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion Matrix — Test Set (fine-tuned transformer)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure: training curves
# ---------------------------------------------------------------------------


def plot_training_curves(training_log: dict, out_path: Path) -> None:
    history = training_log.get("per_epoch_metrics", [])
    if not history:
        return

    epochs = [h["epoch"] for h in history]
    eval_losses = [h.get("eval_loss") for h in history]
    eval_accs = [h.get("eval_accuracy") for h in history]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(epochs, eval_losses, marker="o", color="steelblue", label="Val loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Validation Loss per Epoch")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, eval_accs, marker="o", color="seagreen", label="Val accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Validation Accuracy per Epoch")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("Sentiment Model Fine-tuning — Training Curves", fontsize=13)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure: per-class metrics bar chart
# ---------------------------------------------------------------------------


def plot_per_class_metrics(
    precisions: np.ndarray,
    recalls: np.ndarray,
    f1s: np.ndarray,
    out_path: Path,
) -> None:
    x = np.arange(len(LABEL_NAMES))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width, precisions, width, label="Precision", color="steelblue")
    ax.bar(x, recalls, width, label="Recall", color="seagreen")
    ax.bar(x + width, f1s, width, label="F1", color="coral")

    ax.set_xticks(x)
    ax.set_xticklabels(LABEL_NAMES)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Per-class Precision / Recall / F1 — Test Set")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Qualitative examples
# ---------------------------------------------------------------------------


def _pick_examples(
    texts: list[str],
    true_labels: np.ndarray,
    pred_labels: np.ndarray,
    probs: np.ndarray,
    n: int = 5,
    correct: bool = True,
) -> list[dict]:
    mask = (true_labels == pred_labels) if correct else (true_labels != pred_labels)
    indices = np.where(mask)[0]
    rng = np.random.default_rng(42)
    chosen = rng.choice(indices, size=min(n, len(indices)), replace=False)
    results = []
    for idx in sorted(chosen):
        results.append(
            {
                "text_excerpt": textwrap.shorten(texts[idx], width=200, placeholder="..."),
                "true": ID2LABEL[int(true_labels[idx])],
                "predicted": ID2LABEL[int(pred_labels[idx])],
                "confidence": round(float(probs[idx].max()), 4),
            }
        )
    return results


# ---------------------------------------------------------------------------
# Markdown report writer
# ---------------------------------------------------------------------------


def _fmt_table(headers: list[str], rows: list[list]) -> str:
    col_w = [max(len(str(h)), max((len(str(r)) for r in col), default=0))
             for h, col in zip(headers, zip(*rows, strict=False))] if rows else [len(h) for h in headers]
    sep = "| " + " | ".join("-" * w for w in col_w) + " |"
    header_line = "| " + " | ".join(str(h).ljust(w) for h, w in zip(headers, col_w)) + " |"
    body = "\n".join(
        "| " + " | ".join(str(cell).ljust(w) for cell, w in zip(row, col_w)) + " |"
        for row in rows
    )
    return "\n".join([header_line, sep, body])


def write_evaluation_md(
    overall_acc: float,
    macro_f1: float,
    precisions: np.ndarray,
    recalls: np.ndarray,
    f1s: np.ndarray,
    cm: np.ndarray,
    clf_report: str,
    baseline_random_acc: float,
    baseline_majority_acc: float,
    correct_examples: list[dict],
    misclassified_examples: list[dict],
    out_path: Path,
) -> None:
    lines = [
        "# Evaluation Report — Sentiment Classifier",
        "",
        "Generated by `src/models/evaluate.py`. Test set: held-out 13,306 reviews (never seen during training).",
        "",
        "## 1. Overall Metrics",
        "",
        _fmt_table(
            ["Metric", "Value"],
            [
                ["Accuracy", f"{overall_acc:.4f}"],
                ["Macro F1", f"{macro_f1:.4f}"],
            ],
        ),
        "",
        "## 2. Per-class Metrics",
        "",
        _fmt_table(
            ["Class", "Precision", "Recall", "F1"],
            [
                [
                    LABEL_NAMES[i],
                    f"{precisions[i]:.4f}",
                    f"{recalls[i]:.4f}",
                    f"{f1s[i]:.4f}",
                ]
                for i in range(3)
            ],
        ),
        "",
        "## 3. Confusion Matrix",
        "",
        "See `report/figures/confusion_matrix.png`.",
        "",
        "```",
        "Predicted →    NEGATIVE  NEUTRAL  POSITIVE",
    ]
    for i, row_name in enumerate(LABEL_NAMES):
        lines.append(f"True {row_name:<12} {cm[i, 0]:>8}  {cm[i, 1]:>7}  {cm[i, 2]:>8}")
    lines += [
        "```",
        "",
        "## 4. Baseline Comparisons",
        "",
        _fmt_table(
            ["Classifier", "Accuracy", "Notes"],
            [
                ["Random (uniform)", f"{baseline_random_acc:.4f}", "33% expected for 3 classes"],
                ["Majority class (POSITIVE)", f"{baseline_majority_acc:.4f}", "Always predicts the most frequent class"],
                ["Fine-tuned transformer (ours)", f"{overall_acc:.4f}", "RoBERTa-base or DistilBERT, 4 epochs, seed=42"],
            ],
        ),
        "",
        "## 5. Classification Report (sklearn)",
        "",
        "```",
        clf_report,
        "```",
        "",
        "## 6. Correct Predictions (5 examples)",
        "",
    ]
    for i, ex in enumerate(correct_examples, 1):
        lines += [
            f"### Example {i}",
            f"- **True / Predicted:** {ex['true']} / {ex['predicted']}",
            f"- **Confidence:** {ex['confidence']}",
            f"- **Text:** {ex['text_excerpt']}",
            "",
        ]
    lines += ["## 7. Misclassified Examples (5 examples)", ""]
    for i, ex in enumerate(misclassified_examples, 1):
        lines += [
            f"### Example {i}",
            f"- **True:** {ex['true']}  |  **Predicted:** {ex['predicted']}",
            f"- **Confidence:** {ex['confidence']}",
            f"- **Text:** {ex['text_excerpt']}",
            "",
        ]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main evaluation entry point
# ---------------------------------------------------------------------------


def evaluate_model(
    model_dir: Path = SENTIMENT_MODEL_PATH,
    batch_size: int = 64,
    max_length: int = 256,
) -> dict:
    """Load the saved model, evaluate on test set, write all artifacts.

    Returns a dict with key metrics (accuracy, macro_f1).
    """
    model_dir = Path(model_dir)
    if not model_dir.exists():
        raise FileNotFoundError(
            f"Model directory not found: {model_dir}\n"
            "Run `python src/models/train.py` first."
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load model + tokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    model.to(device)

    # Load training log for curves
    training_log_path = model_dir / "training_log.json"
    training_log = {}
    if training_log_path.exists():
        with open(training_log_path, encoding="utf-8") as fh:
            training_log = json.load(fh)

    # Load test set
    test_df = pd.read_csv(DATA_PROCESSED / "test.csv")
    texts = test_df["text"].tolist()
    true_labels = test_df["label"].to_numpy()

    print(f"Evaluating on {len(texts)} test samples...")
    pred_labels, probs = run_inference(
        model, tokenizer, texts, true_labels, batch_size=batch_size, device=device
    )

    # Metrics
    overall_acc = accuracy_score(true_labels, pred_labels)
    macro_f1 = f1_score(true_labels, pred_labels, average="macro", zero_division=0)
    precisions, recalls, f1s, _ = precision_recall_fscore_support(
        true_labels, pred_labels, labels=[0, 1, 2], zero_division=0
    )
    cm = confusion_matrix(true_labels, pred_labels, labels=[0, 1, 2])
    clf_report = classification_report(
        true_labels, pred_labels, target_names=LABEL_NAMES, zero_division=0
    )

    # Baselines
    rnd_acc = baseline_random(true_labels)
    maj_acc = baseline_majority(true_labels)

    # Qualitative examples
    correct_ex = _pick_examples(texts, true_labels, pred_labels, probs, correct=True)
    wrong_ex = _pick_examples(texts, true_labels, pred_labels, probs, correct=False)

    # Figures
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_confusion_matrix(cm, FIGURES_DIR / "confusion_matrix.png")
    plot_training_curves(training_log, FIGURES_DIR / "training_curves.png")
    plot_per_class_metrics(precisions, recalls, f1s, FIGURES_DIR / "per_class_metrics.png")

    # Markdown report
    write_evaluation_md(
        overall_acc=overall_acc,
        macro_f1=macro_f1,
        precisions=precisions,
        recalls=recalls,
        f1s=f1s,
        cm=cm,
        clf_report=clf_report,
        baseline_random_acc=rnd_acc,
        baseline_majority_acc=maj_acc,
        correct_examples=correct_ex,
        misclassified_examples=wrong_ex,
        out_path=DOCS_DIR / "evaluation.md",
    )

    print(f"Accuracy:  {overall_acc:.4f}")
    print(f"Macro F1:  {macro_f1:.4f}")
    print(f"Artifacts written to docs/evaluation.md + report/figures/")

    return {"accuracy": overall_acc, "macro_f1": macro_f1}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained transformer sentiment model (defaults to RoBERTa).")
    parser.add_argument(
        "--model-dir", type=Path, default=SENTIMENT_MODEL_PATH,
        help="Path to saved HuggingFace checkpoint directory."
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=256)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    evaluate_model(
        model_dir=args.model_dir,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
