"""
Fine-tune FinBERT on custom financial news data.

Dataset format (CSV):
    headline,label
    "Apple beats earnings expectations",positive
    "Market crash fears amid Fed rate hike",negative
    "Stock prices remain unchanged",neutral
"""

import argparse
import csv
from pathlib import Path

import torch
from datasets import Dataset
from loguru import logger
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

FINBERT_MODEL = "ProsusAI/finbert"
LABEL2ID = {"positive": 0, "negative": 1, "neutral": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


def load_csv_dataset(path: str) -> Dataset:
    """Load headline/label CSV into a HuggingFace Dataset."""
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = row["label"].strip().lower()
            if label not in LABEL2ID:
                continue
            rows.append({"text": row["headline"].strip(), "label": LABEL2ID[label]})
    logger.info(f"Loaded {len(rows)} examples from {path}")
    return Dataset.from_list(rows)


def tokenize(batch, tokenizer):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=128)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
    }


def finetune(train_path: str, val_path: str, output_dir: str = "checkpoints/finbert"):
    tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        FINBERT_MODEL,
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    )

    train_ds = load_csv_dataset(train_path).map(lambda b: tokenize(b, tokenizer), batched=True)
    val_ds = load_csv_dataset(val_path).map(lambda b: tokenize(b, tokenizer), batched=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        logging_dir="logs/finbert",
        report_to="mlflow",
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    logger.info(f"Fine-tuning FinBERT on {len(train_ds)} train / {len(val_ds)} val examples")
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info(f"Fine-tuned model saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True, help="Path to train CSV")
    parser.add_argument("--val", required=True, help="Path to validation CSV")
    parser.add_argument("--output", default="checkpoints/finbert")
    args = parser.parse_args()
    finetune(args.train, args.val, args.output)


if __name__ == "__main__":
    main()
