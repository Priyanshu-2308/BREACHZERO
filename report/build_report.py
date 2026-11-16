# Reads the CSVs in results/ and data/ and writes tables.typ, then compiles the PDF.
# Every number in tables.typ comes from a file, nothing is typed by hand here.

import os
import subprocess
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS = os.path.join(ROOT, "results")
DATA = os.path.join(ROOT, "data")

DISGUISES = ["leetspeak", "cyrillic", "base64", "wrapped", "hindi"]
DISGUISE_LABEL = {
    "leetspeak": "Leetspeak",
    "cyrillic": "Cyrillic",
    "base64": "Base64",
    "wrapped": "Wrapped",
    "hindi": "Hindi-style",
}
MODEL_LABEL = {
    "baseline": "Baseline (TF-IDF + LR)",
    "distilbert": "English DistilBERT",
    "distilbert_multilingual": "Multilingual DistilBERT",
}
SET_LABEL = {
    "test": "In-distribution test",
    "ood_jailbreak": "OOD jailbreak",
    "leetspeak": "Attack: leetspeak",
    "cyrillic": "Attack: Cyrillic",
    "base64": "Attack: Base64",
    "wrapped": "Attack: wrapped",
    "hindi": "Attack: Hindi-style",
}


def f4(x):
    # Four decimals everywhere so the tables match the script printouts.
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "n/a"
    return f"{float(x):.4f}"


def pct(x):
    return f"{float(x) * 100:.2f}%"


def n_fmt(x):
    return f"{int(x):,}"


def s(text):
    # Emit a cell as a typst string so punctuation is never read as markup.
    t = str(text).replace("\\", "\\\\").replace('"', '\\"')
    return '[#"' + t + '"]'


def shorten(text, limit):
    t = " ".join(str(text).split())
    if len(t) <= limit:
        return t
    cut = t[: limit - 3]
    space = cut.rfind(" ")
    if space > limit * 0.6:
        cut = cut[:space]
    return cut.rstrip(" ,.;:") + "..."


def read(name):
    return pd.read_csv(os.path.join(RESULTS, name))


def table(columns, align, header, rows, extra=""):
    out = ["table("]
    out.append("  columns: (" + ", ".join(columns) + "),")
    out.append("  align: (" + ", ".join(align) + "),")
    if extra:
        out.append("  " + extra)
    out.append("  table.header(" + ", ".join(header) + "),")
    for r in rows:
        out.append("  " + ", ".join(r) + ",")
    out.append(")")
    return "\n".join(out)


def build_split_tables():
    tr = pd.read_csv(os.path.join(DATA, "train.csv"))
    va = pd.read_csv(os.path.join(DATA, "val.csv"))
    te = pd.read_csv(os.path.join(DATA, "test.csv"))
    ood = pd.read_csv(os.path.join(DATA, "ood_jailbreak.csv"))

    rows = []
    for name, df in [("train.csv", tr), ("val.csv", va), ("test.csv", te), ("ood_jailbreak.csv", ood)]:
        benign = int((df["label"] == 0).sum())
        attack = int((df["label"] == 1).sum())
        rows.append([
            s(name), f"[{n_fmt(len(df))}]", f"[{n_fmt(benign)}]", f"[{n_fmt(attack)}]",
            f"[{100 * attack / len(df):.2f}%]",
        ])
    t_splits = table(
        ["auto", "auto", "auto", "auto", "auto"],
        ["left", "right", "right", "right", "right"],
        [s("File"), s("Rows"), s("Benign (0)"), s("Attack (1)"), s("Attack share")],
        rows,
    )

    sources = ["databricks_dolly_15k", "gandalf_ignore_instructions", "deepset_prompt_injections"]
    rows = []
    for src in sources:
        cells = [s(src)]
        for df in (tr, va, te):
            sub = df[df["source"] == src]
            cells.append(f"[{n_fmt(int((sub['label'] == 0).sum()))}]")
            cells.append(f"[{n_fmt(int((sub['label'] == 1).sum()))}]")
        rows.append(cells)
    totals = [s("Total")]
    for df in (tr, va, te):
        totals.append(f"[{n_fmt(int((df['label'] == 0).sum()))}]")
        totals.append(f"[{n_fmt(int((df['label'] == 1).sum()))}]")
    rows.append(totals)

    t_sources = table(
        ["auto"] + ["auto"] * 6,
        ["left"] + ["right"] * 6,
        [s("Source dataset"), s("Train 0"), s("Train 1"), s("Val 0"), s("Val 1"), s("Test 0"), s("Test 1")],
        rows,
    )
    return t_splits, t_sources


def build_main_results():
    df = read("main_results.csv")
    rows = []
    for _, r in df.iterrows():
        rows.append([
            s(MODEL_LABEL[r["model"]]),
            s(SET_LABEL[r["test_set"]]),
            f"[{f4(r['recall'])}]",
            f"[{f4(r['false_positive_rate'])}]",
            f"[{f4(r['f1'])}]",
            f"[{n_fmt(r['n'])}]",
        ])
    return table(
        ["2.6fr", "2.5fr", "1.1fr", "1.1fr", "1.1fr", "1fr"],
        ["left", "left", "right", "right", "right", "right"],
        [s("Model"), s("Evaluation set"), s("Recall"), s("FPR"), s("F1"), s("N")],
        rows,
    )


def build_recall_vs_benign():
    main = read("main_results.csv")
    benign = read("benign_disguise_check.csv")
    rows = []
    for model in ["baseline", "distilbert", "distilbert_multilingual"]:
        for d in DISGUISES:
            rec = main[(main["model"] == model) & (main["test_set"] == d)]["recall"].iloc[0]
            fpr = benign[(benign["model"] == model) & (benign["disguise"] == d)]["false_positive_rate"].iloc[0]
            rows.append([
                s(MODEL_LABEL[model]),
                s(DISGUISE_LABEL[d]),
                f"[{f4(rec)}]",
                f"[{f4(fpr)}]",
                f"[{pct(fpr)}]",
            ])
    return table(
        ["2.8fr", "1.4fr", "1.4fr", "1.6fr", "1.2fr"],
        ["left", "left", "right", "right", "right"],
        [s("Model"), s("Disguise"), s("Attack recall"), s("Benign FPR"), s("Benign FPR %")],
        rows,
    )


def build_benign_disguise():
    df = read("benign_disguise_check.csv")
    rows = []
    for model in ["baseline", "distilbert", "distilbert_multilingual"]:
        cells = [s(MODEL_LABEL[model])]
        for d in DISGUISES:
            v = df[(df["model"] == model) & (df["disguise"] == d)]["false_positive_rate"].iloc[0]
            cells.append(f"[{f4(v)}]")
        rows.append(cells)
    return table(
        ["2.6fr"] + ["1fr"] * 5,
        ["left"] + ["right"] * 5,
        [s("Model")] + [s(DISGUISE_LABEL[d]) for d in DISGUISES],
        rows,
    )


def build_long_input():
    df = read("long_input_check.csv")
    rows = []
    for _, r in df.iterrows():
        rows.append([
            s(r["strategy"]),
            f"[{f4(r['recall'])}]",
            f"[{f4(r['false_positive_rate'])}]",
            f"[{f4(r['precision'])}]",
            f"[{f4(r['f1'])}]",
            f"[{n_fmt(r['n'])}]",
        ])
    return table(
        ["1.6fr", "1fr", "1fr", "1.2fr", "1fr", "1fr"],
        ["left", "right", "right", "right", "right", "right"],
        [s("Strategy"), s("Recall"), s("FPR"), s("Precision"), s("F1"), s("N")],
        rows,
    )


def build_ood_diagnosis():
    df = read("ood_fpr_diagnosis.csv")
    vals = dict(zip(df["metric"], df["value"]))

    rows = [
        [s("Benign prompts in the OOD set"), f"[{n_fmt(float(vals['total_benign']))}]"],
        [s("Flagged as attack (false positives)"), f"[{n_fmt(float(vals['flagged_false_positives']))}]"],
        [s("False positive rate"), f"[{f4(vals['false_positive_rate'])}]"],
        [s("Median characters, flagged"), f"[{float(vals['median_chars_flagged']):.1f}]"],
        [s("Median characters, not flagged"), f"[{float(vals['median_chars_unflagged']):.1f}]"],
        [s("Median subword tokens, flagged"), f"[{float(vals['median_tokens_flagged']):.1f}]"],
        [s("Median subword tokens, not flagged"), f"[{float(vals['median_tokens_unflagged']):.1f}]"],
    ]
    t_counts = table(
        ["auto", "auto"], ["left", "right"],
        [s("Quantity"), s("Value")], rows,
    )

    word_rows = []
    for i in range(1, 16):
        key = f"top_{i:02d}_word"
        if key not in vals:
            continue
        raw = str(vals[key])
        word = raw.split(" (")[0]
        inner = raw.split("(", 1)[1].rstrip(")")
        parts = dict(p.strip().split("=") for p in inner.split(","))
        word_rows.append([
            f"[{i}]", s(word), f"[{parts['ratio']}]", f"[{parts['FP']}]", f"[{parts['TN']}]",
        ])
    t_words = table(
        ["0.7fr", "2fr", "1fr", "1fr", "1fr"],
        ["right", "left", "right", "right", "right"],
        [s("Rank"), s("Word"), s("Ratio"), s("Count in FP"), s("Count in TN")],
        word_rows,
    )
    return t_counts, t_words


def build_ood_examples(limit=4, chars=86):
    df = read("ood_flagged_examples.csv").head(limit)
    rows = []
    for _, r in df.iterrows():
        rows.append([s(shorten(r["truncated_text"], chars)), f"[{f4(r['score'])}]"])
    return table(
        ["5.5fr", "1fr"], ["left", "right"],
        [s("Benign OOD prompt (truncated)"), s("Score")], rows,
    )


def build_shortcut():
    df = read("shortcut_check.csv")
    cols = ["baseline_prob", "distilbert_prob", "distilbert_multilingual_prob"]
    n = len(df)

    summary_rows = []
    for model, col in zip(["baseline", "distilbert", "distilbert_multilingual"], cols):
        flagged = int((df[col] >= 0.5).sum())
        summary_rows.append([
            s(MODEL_LABEL[model]), f"[{flagged} / {n}]", f"[{f4(flagged / n)}]",
        ])
    t_summary = table(
        ["2.8fr", "1.2fr", "1fr"], ["left", "right", "right"],
        [s("Model"), s("Flagged"), s("FPR")], summary_rows,
    )

    flagged_any = df[(df[cols] >= 0.5).any(axis=1)]
    rows = []
    for _, r in flagged_any.iterrows():
        rows.append([
            s(shorten(r["text"], 90)),
            f"[{f4(r['baseline_prob'])}]",
            f"[{f4(r['distilbert_prob'])}]",
            f"[{f4(r['distilbert_multilingual_prob'])}]",
        ])
    t_rows = table(
        ["4.6fr", "1fr", "1fr", "1.2fr"],
        ["left", "right", "right", "right"],
        [s("Benign prompt"), s("Baseline"), s("English"), s("Multilingual")],
        rows,
    )
    return t_summary, t_rows


def build_augment():
    df = read("augment_results.csv")
    models = ["distilbert_before", "distilbert_aug_hindi", "distilbert_aug_base64", "distilbert_aug_all"]
    order = ([f"attack_{d}" for d in DISGUISES]
             + [f"benign_{d}" for d in DISGUISES]
             + ["test_set", "ood_jailbreak"])
    label = {}
    for d in DISGUISES:
        label[f"attack_{d}"] = f"attack_{d}"
        label[f"benign_{d}"] = f"benign_{d}"
    label["test_set"] = "test_set"
    label["ood_jailbreak"] = "ood_jailbreak"

    rows = []
    for es in order:
        sub = df[df["eval_set"] == es]
        metric = sub["metric_type"].iloc[0]
        cells = [s(label[es]), s("Recall" if metric == "recall" else "FPR")]
        for m in models:
            v = sub[sub["model"] == m]["score"]
            cells.append(f"[{f4(v.iloc[0])}]" if len(v) else "[n/a]")
        rows.append(cells)

    header = [s("Evaluation set"), s("Metric"),
              s("before"), s("aug_hindi"), s("aug_base64"), s("aug_all")]
    return table(
        ["2.2fr", "1fr", "1.1fr", "1.3fr", "1.4fr", "1.1fr"],
        ["left", "left", "right", "right", "right", "right"],
        header, rows,
    )


def build_threshold():
    df = read("threshold_summary.csv")
    order = [("val", "fpr"), ("val", "recall"),
             ("test", "fpr"), ("test", "recall"), ("test", "f1")]
    order += [(f"attack_{d}", "recall") for d in DISGUISES]
    order += [("ood_jailbreak", "fpr")]
    order += [(f"benign_{d}", "fpr") for d in DISGUISES]

    rows = []
    for es, metric in order:
        v50 = df[(df["threshold"] == 0.5) & (df["eval_set"] == es) & (df["metric"] == metric)]["value"]
        v06 = df[(df["threshold"] == 0.06) & (df["eval_set"] == es) & (df["metric"] == metric)]["value"]
        rows.append([
            s(es),
            s({"fpr": "FPR", "recall": "Recall", "f1": "F1"}[metric]),
            f"[{f4(v50.iloc[0])}]",
            f"[{f4(v06.iloc[0])}]",
        ])
    return table(
        ["2.4fr", "1fr", "1.4fr", "1.6fr"],
        ["left", "left", "right", "right"],
        [s("Evaluation set"), s("Metric"), s("Threshold 0.50"), s("Threshold 0.06")],
        rows,
    )


def build_chat():
    df = read("chat_fpr_summary.csv")
    scores = read("chat_fpr_check.csv")
    rows = []
    for _, r in df.sort_values("threshold").iterrows():
        blocked = int((scores["score"] >= r["threshold"]).sum())
        rows.append([
            f"[{r['threshold']:.2f}]",
            f"[{blocked} / {int(r['n'])}]",
            f"[{f4(r['false_positive_rate'])}]",
            f"[{pct(r['false_positive_rate'])}]",
        ])
    return table(
        ["1.2fr", "1.2fr", "1.2fr", "1.2fr"],
        ["right", "right", "right", "right"],
        [s("Threshold"), s("Blocked"), s("FPR"), s("FPR %")],
        rows,
    )


def build_demo():
    df = read("demo_check.csv")
    names = {}
    for i, p in enumerate(df["prompt"].unique()):
        names[p] = f"Attack {i + 1}" if i < 3 else "Benign query"
    rows = []
    for _, r in df.iterrows():
        rows.append([
            s(names[r["prompt"]]),
            s(r["guard"]),
            f"[{f4(r['guard_score'])}]",
            s(str(r["blocked"])),
            s(str(r["leaked_password"])),
            s(shorten(r["reply_truncated"], 46)),
        ])
    return table(
        ["1.3fr", "0.8fr", "1fr", "1fr", "1.1fr", "3.4fr"],
        ["left", "center", "right", "center", "center", "left"],
        [s("Input"), s("Guard"), s("Score"), s("Blocked"), s("Leaked"), s("Reply (truncated)")],
        rows,
    )


def build_disguise_examples(limit=64):
    # One injection from data/attacks/*.csv shown through all five disguises.
    # The row is picked by a fixed rule so the table does not depend on hand-chosen text.
    base = pd.read_csv(os.path.join(ROOT, "data", "attacks", "leetspeak.csv"))
    lens = base["original_text"].astype(str).str.len()
    idx = int(lens[lens >= 40].idxmin())

    original = str(base.loc[idx, "original_text"])
    rows = [[s("Original"), s(shorten(original, 72))]]
    for d in DISGUISES:
        df = pd.read_csv(os.path.join(ROOT, "data", "attacks", f"{d}.csv"))
        assert str(df.loc[idx, "original_text"]) == original, f"row {idx} differs in {d}.csv"
        rows.append([s(DISGUISE_LABEL[d]), s(shorten(df.loc[idx, "text"], limit))])
    return table(
        ["1.2fr", "5fr"], ["left", "left"],
        [s("Variant"), s("Text")], rows,
    )


def build_output_check():
    """Input score, input-side decision and output-side decision for each demo prompt."""
    df = pd.read_csv(os.path.join(RESULTS, "output_check.csv"))
    prompts = list(dict.fromkeys(df["prompt"]))
    short = {}
    for i, p in enumerate(prompts):
        short[p] = f"Attack {i + 1}" if i < 3 else f"Benign {i - 2}"

    def mark(v):
        t = str(v).strip()
        if t in ("True", "False"):
            return t
        return "n/a"

    rows = []
    for p in prompts:
        g = df[df["prompt"] == p]
        score = g["guard_score"].iloc[0]
        for _, r in g.iterrows():
            thr = r["threshold"]
            thr_s = "n/a" if pd.isna(thr) or str(thr).strip() == "" else f"{float(thr):.2f}"
            rows.append([
                s(short[p]),
                f"[{f4(score)}]",
                s(r["guard"]),
                s(thr_s),
                s(mark(r["input_blocked"])),
                s(mark(r["reply_has_secret"])),
                s(mark(r["withheld_by_output_check"])),
            ])
    return table(
        ["1.3fr", "1.1fr", "0.9fr", "1fr", "1.2fr", "1.4fr", "1.2fr"],
        ["left", "right", "center", "right", "center", "center", "center"],
        [s("Input"), s("Score"), s("Guard"), s("Threshold"),
         s("Input blocked"), s("Secret in reply"), s("Withheld")],
        rows,
    )


def build_output_check_prompts():
    df = pd.read_csv(os.path.join(RESULTS, "output_check.csv"))
    prompts = list(dict.fromkeys(df["prompt"]))
    rows = []
    for i, p in enumerate(prompts):
        label = f"Attack {i + 1}" if i < 3 else f"Benign {i - 2}"
        score = df[df["prompt"] == p]["guard_score"].iloc[0]
        rows.append([s(label), s(shorten(p, 92)), f"[{f4(score)}]"])
    return table(
        ["1.2fr", "5.6fr", "1.1fr"],
        ["left", "left", "right"],
        [s("Label"), s("Prompt"), s("Guard score")],
        rows,
    )


def build_lime():
    # The nine LIME scores are parsed out of results/lime_examples.html.
    import re
    path = os.path.join(RESULTS, "lime_examples.html")
    with open(path, encoding="utf-8") as fh:
        html = fh.read()
    titles = re.findall(r"Example \d+: ([^<]+)", html)
    probs = re.findall(r"P\(Attack\)=([0-9.]+)", html)
    preds = re.findall(r'Model Prediction:</strong> <span style="color: #[a-f0-9]+;">(\w+)', html)
    rows = []
    for i, (t, p, pr) in enumerate(zip(titles, probs, preds), 1):
        rows.append([f"[{i}]", s(t.strip()), f"[{p}]", s(pr)])
    return table(
        ["0.6fr", "4fr", "1.2fr", "1.4fr"],
        ["right", "left", "right", "left"],
        [s("#"), s("Example category"), s("P(injection)"), s("Prediction")],
        rows,
    )


def main():
    t_splits, t_sources = build_split_tables()
    t_ood_counts, t_ood_words = build_ood_diagnosis()
    t_short_summary, t_short_rows = build_shortcut()

    blocks = {
        "tbl_splits": t_splits,
        "tbl_sources": t_sources,
        "tbl_disguise_examples": build_disguise_examples(),
        "tbl_main_results": build_main_results(),
        "tbl_recall_vs_benign": build_recall_vs_benign(),
        "tbl_benign_disguise": build_benign_disguise(),
        "tbl_long_input": build_long_input(),
        "tbl_ood_counts": t_ood_counts,
        "tbl_ood_words": t_ood_words,
        "tbl_ood_examples": build_ood_examples(),
        "tbl_shortcut_summary": t_short_summary,
        "tbl_shortcut_rows": t_short_rows,
        "tbl_augment": build_augment(),
        "tbl_threshold": build_threshold(),
        "tbl_chat": build_chat(),
        "tbl_demo": build_demo(),
        "tbl_output_prompts": build_output_check_prompts(),
        "tbl_output_check": build_output_check(),
        "tbl_lime": build_lime(),
    }

    out = ["// Generated by build_report.py from results/*.csv and data/*.csv.",
           "// Do not edit by hand; run `python report/build_report.py` instead.", ""]
    for name, body in blocks.items():
        out.append(f"#let {name} = " + body)
        out.append("")

    tables_path = os.path.join(HERE, "tables.typ")
    with open(tables_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(out))
    print(f"wrote {tables_path} ({len(blocks)} tables)")

    import typst
    src = os.path.join(HERE, "BREACHZERO_Report.typ")
    pdf = os.path.join(HERE, "BREACHZERO_Report.pdf")
    typst.compile(src, output=pdf, root=ROOT)
    print(f"wrote {pdf} ({os.path.getsize(pdf) / 1024:.0f} KB)")

    info = subprocess.run(["pdfinfo", pdf], capture_output=True, text=True)
    for line in info.stdout.splitlines():
        if line.startswith(("Pages", "Page size")):
            print(line)


if __name__ == "__main__":
    main()
