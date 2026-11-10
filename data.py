"""Downloads and curates datasets into train, val, test, and ood_jailbreak CSVs."""

import os

import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

from config import DATA_DIR, SEED

N_DOLLY_BENIGN = 3000


def load_prompt_injections():
    ds = load_dataset("deepset/prompt-injections")
    rows = []
    for split in ds:
        df = ds[split].to_pandas()[["text", "label"]].copy()
        df["source"] = "deepset_prompt_injections"
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


def load_gandalf_attacks():
    # Every row in this set is an attack prompt
    ds = load_dataset("Lakera/gandalf_ignore_instructions")
    rows = []
    for split in ds:
        df = ds[split].to_pandas()[["text"]].copy()
        df["label"] = 1
        df["source"] = "gandalf_ignore_instructions"
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


def load_dolly_benign(n, seed):
    ds = load_dataset("databricks/databricks-dolly-15k")
    df = ds["train"].to_pandas()[["instruction"]].rename(columns={"instruction": "text"})
    df = df.sample(n=min(n, len(df)), random_state=seed).reset_index(drop=True)
    df["label"] = 0
    df["source"] = "databricks_dolly_15k"
    return df


def load_ood_jailbreak():
    ds = load_dataset("jackhhao/jailbreak-classification")
    rows = []
    for split in ds:
        df = ds[split].to_pandas()[["prompt", "type"]].rename(columns={"prompt": "text"})
        rows.append(df)
    df = pd.concat(rows, ignore_index=True)
    df["label"] = (df["type"] == "jailbreak").astype(int)
    df["source"] = "jackhhao_jailbreak_classification"
    return df[["text", "label", "source"]]


def clean(df):
    df = df.copy()
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"] != ""]
    df = df.drop_duplicates(subset=["text"])
    return df.reset_index(drop=True)


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    injections = load_prompt_injections()
    attack_rows = injections[injections["label"] == 1][["text", "label", "source"]]
    benign_rows = injections[injections["label"] == 0][["text", "label", "source"]]

    gandalf = load_gandalf_attacks()[["text", "label", "source"]]
    dolly = load_dolly_benign(N_DOLLY_BENIGN, SEED)[["text", "label", "source"]]

    full = pd.concat([attack_rows, benign_rows, gandalf, dolly], ignore_index=True)
    full = clean(full)

    print("Class counts per source (pre-split):")
    print(full.groupby(["source", "label"]).size())

    # Stratified 70/15/15 split
    train_df, rest_df = train_test_split(
        full, test_size=0.30, random_state=SEED, stratify=full["label"]
    )
    val_df, test_df = train_test_split(
        rest_df, test_size=0.50, random_state=SEED, stratify=rest_df["label"]
    )

    train_df.to_csv(os.path.join(DATA_DIR, "train.csv"), index=False)
    val_df.to_csv(os.path.join(DATA_DIR, "val.csv"), index=False)
    test_df.to_csv(os.path.join(DATA_DIR, "test.csv"), index=False)

    ood_df = clean(load_ood_jailbreak())
    ood_df.to_csv(os.path.join(DATA_DIR, "ood_jailbreak.csv"), index=False)

    print("\nSplit sizes:")
    for name, df in [("train", train_df), ("val", val_df), ("test", test_df), ("ood_jailbreak", ood_df)]:
        print(f"  {name}: {len(df)} rows, label counts = {df['label'].value_counts().to_dict()}")

    print("\nClass counts per split and per source:")
    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        print(f"-- {name} --")
        print(df.groupby(["source", "label"]).size())

    print("\ndata.py finished: wrote train.csv, val.csv, test.csv, ood_jailbreak.csv to data/")


if __name__ == "__main__":
    main()
