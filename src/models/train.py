"""Fine-tune a transformer encoder (default: roberta-base) for 3-class sentiment classification.

The RoBERTa-base run is the production model (test acc 80.83% / macro-F1 0.7686).
DistilBERT-base-uncased is preserved as the ablation baseline (79.59% / 0.7617).
Pass --model-name to switch.

Usage (operator runs this — do not execute during script authoring):
    python src/models/train.py [--model-name NAME] [--epochs N] [--batch-size N] [--lr F]

Outputs:
    models/sentiment_roberta/   -- best checkpoint (HF format)
    models/sentiment_roberta/training_log.json  -- per-epoch metrics
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from torch.optim import AdamW
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    get_linear_schedule_with_warmup,
)
import sklearn.metrics as skm
import torch.nn as nn

from config.settings import DATA_PROCESSED, SENTIMENT_MODEL_PATH


class WeightedTrainer(Trainer):
    """Trainer subclass that injects class weights into the cross-entropy loss.

    Used to upweight the minority/hard NEUTRAL class. weights is a 1D tensor of
    length = num_labels (3 for this project).
    """

    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        weight = self._class_weights.to(logits.device) if self._class_weights is not None else None
        loss = nn.CrossEntropyLoss(weight=weight)(logits, labels)
        return (loss, outputs) if return_outputs else loss

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

SEED = 42


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Label mapping (must match evaluate.py and predict.py)
# ---------------------------------------------------------------------------

ID2LABEL = {0: "NEGATIVE", 1: "NEUTRAL", 2: "POSITIVE"}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load train/val/test CSVs from data/processed/."""
    train_df = pd.read_csv(DATA_PROCESSED / "train.csv")
    val_df = pd.read_csv(DATA_PROCESSED / "val.csv")
    test_df = pd.read_csv(DATA_PROCESSED / "test.csv")
    # Validate expected columns
    required = {"review_id", "text", "label"}
    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{name}.csv missing columns: {missing}")
    return train_df, val_df, test_df


def build_hf_dataset(df: pd.DataFrame, tokenizer, max_length: int) -> Dataset:
    """Convert a pandas DataFrame to a tokenized HuggingFace Dataset."""
    ds = Dataset.from_pandas(df[["text", "label"]].reset_index(drop=True))

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
            padding=False,  # DataCollatorWithPadding handles dynamic padding
        )

    ds = ds.map(tokenize, batched=True, remove_columns=["text"])
    ds = ds.rename_column("label", "labels")
    ds.set_format(type="torch")
    return ds


# ---------------------------------------------------------------------------
# Metrics (called by Trainer at each eval step)
# ---------------------------------------------------------------------------


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = skm.accuracy_score(labels, preds)
    macro_f1 = skm.f1_score(labels, preds, average="macro", zero_division=0)
    return {"accuracy": acc, "macro_f1": macro_f1}


# ---------------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------------


def train_model(
    model_name: str = "roberta-base",
    max_length: int = 256,
    batch_size: int = 16,
    gradient_accumulation_steps: int = 2,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01,
    num_epochs: int = 4,
    warmup_ratio: float = 0.1,
    early_stopping_patience: int = 2,
    fp16: bool = True,
    output_dir: Path = SENTIMENT_MODEL_PATH,
    class_weights: list[float] | None = None,
) -> dict:
    """Fine-tune the chosen transformer encoder and save the best checkpoint.

    Returns the final training log dict (also written to output_dir/training_log.json).

    Why fp16=True: RTX 3060 6GB VRAM; fp32 + batch_size=16 may OOM. fp16 halves
    memory footprint with negligible accuracy loss at this scale.

    Why gradient_accumulation_steps=2: effective batch_size = 16 * 2 = 32, matching
    the spec target without doubling VRAM usage.
    """
    set_seed(SEED)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Data
    train_df, val_df, _ = load_splits()
    train_ds = build_hf_dataset(train_df, tokenizer, max_length)
    val_ds = build_hf_dataset(val_df, tokenizer, max_length)

    # Model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    # Dynamic padding collator — pads each batch to its own max length
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        fp16=fp16 and torch.cuda.is_available(),
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=1,
        seed=SEED,
        data_seed=SEED,
        report_to="none",  # no wandb / tensorboard
        logging_steps=50,
        logging_dir=str(output_dir / "hf_logs"),
    )

    # Callbacks
    early_stopping = EarlyStoppingCallback(
        early_stopping_patience=early_stopping_patience
    )

    weights_tensor = (
        torch.tensor(class_weights, dtype=torch.float32) if class_weights is not None else None
    )
    trainer_cls = WeightedTrainer if weights_tensor is not None else Trainer
    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[early_stopping],
    )
    if weights_tensor is not None:
        trainer_kwargs["class_weights"] = weights_tensor
    trainer = trainer_cls(**trainer_kwargs)

    # Train
    train_start = time.time()
    train_result = trainer.train()
    train_elapsed = time.time() - train_start

    # Save best model + tokenizer
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    # Build training log
    history = []
    for log_entry in trainer.state.log_history:
        if "eval_loss" in log_entry:
            history.append(
                {
                    "epoch": log_entry.get("epoch"),
                    "eval_loss": log_entry.get("eval_loss"),
                    "eval_accuracy": log_entry.get("eval_accuracy"),
                    "eval_macro_f1": log_entry.get("eval_macro_f1"),
                }
            )

    training_log = {
        "model_name": model_name,
        "max_length": max_length,
        "batch_size": batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "effective_batch_size": batch_size * gradient_accumulation_steps,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "num_epochs_config": num_epochs,
        "warmup_ratio": warmup_ratio,
        "early_stopping_patience": early_stopping_patience,
        "fp16": fp16 and torch.cuda.is_available(),
        "seed": SEED,
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "train_runtime_seconds": round(train_elapsed, 1),
        "train_samples_per_second": round(
            train_result.metrics.get("train_samples_per_second", 0), 2
        ),
        "best_checkpoint": str(output_dir),
        "per_epoch_metrics": history,
    }

    log_path = output_dir / "training_log.json"
    with open(log_path, "w", encoding="utf-8") as fh:
        json.dump(training_log, fh, indent=2)

    print(f"Training complete. Best model saved to {output_dir}")
    print(f"Training log written to {log_path}")
    return training_log


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune a transformer encoder for sentiment (default: roberta-base).")
    parser.add_argument("--model-name", default="roberta-base")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--early-stopping-patience", type=int, default=2)
    parser.add_argument(
        "--no-fp16",
        action="store_true",
        help="Disable mixed-precision training (use if GPU < 6GB VRAM).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SENTIMENT_MODEL_PATH,
    )
    parser.add_argument(
        "--class-weights",
        type=str,
        default=None,
        help='Comma-separated weights "neg,neu,pos" e.g. "1.0,2.5,1.0" to upweight NEUTRAL.',
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    weights = [float(x) for x in args.class_weights.split(",")] if args.class_weights else None
    if weights is not None:
        print(f"Using class weights: NEG={weights[0]}, NEU={weights[1]}, POS={weights[2]}")
    train_model(
        model_name=args.model_name,
        max_length=args.max_length,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        num_epochs=args.epochs,
        warmup_ratio=args.warmup_ratio,
        early_stopping_patience=args.early_stopping_patience,
        fp16=not args.no_fp16,
        output_dir=args.output_dir,
        class_weights=weights,
    )
