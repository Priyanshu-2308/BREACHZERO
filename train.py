import argparse
import os

import numpy as np
import pandas as pd
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

from config import BASE_MODEL, DATA_DIR, MAX_LEN, SEED
from metrics import compute_metrics


class TextDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.enc = tokenizer(texts, truncation=True, max_length=max_len)
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        item = {k: v[i] for k, v in self.enc.items()}
        item["labels"] = self.labels[i]
        return item


def hf_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    m = compute_metrics(labels, preds)
    m.pop("n")
    return m


def load_split(name, limit=0):
    path = os.path.join(DATA_DIR, f"{name}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {path}. Run data.py (Person 1) first.")
    df = pd.read_csv(path)
    if limit:
        df = df.sample(n=min(limit, len(df)), random_state=SEED).reset_index(drop=True)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=BASE_MODEL)
    ap.add_argument("--out", default="models/distilbert")
    ap.add_argument("--extra_train", default=None, help="extra CSV with text,label to append to training data")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--max_len", type=int, default=MAX_LEN)
    ap.add_argument("--limit", type=int, default=0, help="subsample train/val rows (0 = use all); for smoke tests")
    args = ap.parse_args()

    set_seed(SEED)

    train_df = load_split("train", args.limit)
    val_df = load_split("val", args.limit)

    n_extra = 0
    if args.extra_train:
        if not os.path.exists(args.extra_train):
            raise FileNotFoundError(f"Missing --extra_train file: {args.extra_train}")
        extra = pd.read_csv(args.extra_train)
        missing = {"text", "label"} - set(extra.columns)
        if missing:
            raise ValueError(f"{args.extra_train} is missing columns: {sorted(missing)}")
        if args.limit:
            extra = extra.sample(n=min(args.limit, len(extra)), random_state=SEED)
        n_extra = len(extra)
        train_df = pd.concat([train_df[["text", "label"]], extra[["text", "label"]]], ignore_index=True)
        train_df = train_df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)

    print(f"model={args.model}  train={len(train_df)} (extra={n_extra})  val={len(val_df)}  max_len={args.max_len}")
    print("train label counts:", train_df["label"].value_counts().to_dict())

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2)

    train_ds = TextDataset(train_df["text"].astype(str).tolist(), train_df["label"].astype(int).tolist(),
                           tokenizer, args.max_len)
    val_ds = TextDataset(val_df["text"].astype(str).tolist(), val_df["label"].astype(int).tolist(),
                         tokenizer, args.max_len)

    use_gpu = torch.cuda.is_available()
    targs = TrainingArguments(
        output_dir=os.path.join(args.out, "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        fp16=use_gpu,
        seed=SEED,
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=hf_metrics,
    )

    trainer.train()
    final = trainer.evaluate()

    os.makedirs(args.out, exist_ok=True)
    trainer.save_model(args.out)
    tokenizer.save_pretrained(args.out)

    print("\n" + "=" * 60)
    print(f"saved to {args.out}  (gpu={use_gpu})")
    print(f"best val f1={final['eval_f1']:.4f}  recall={final['eval_recall']:.4f}  "
          f"fpr={final['eval_false_positive_rate']:.4f}")
    print("=" * 60)
    print("train.py finished.")


if __name__ == "__main__":
    main()
