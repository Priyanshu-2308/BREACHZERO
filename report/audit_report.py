# Checks the built report against the files it claims to quote.
# Run after build_report.py. Exits non-zero if anything fails.

import os
import re
import subprocess
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS = os.path.join(ROOT, "results")
DATA = os.path.join(ROOT, "data")
PDF = os.path.join(HERE, "BREACHZERO_Report.pdf")

failures = []
notes = []


def fail(msg):
    failures.append(msg)
    print("FAIL " + msg)


def ok(msg):
    print("ok   " + msg)


# ---------------------------------------------------------------- pdf text

def pdf_text():
    out = subprocess.run(["pdftotext", "-layout", PDF, "-"],
                         capture_output=True, text=True)
    if out.returncode != 0:
        print("pdftotext failed, falling back to pypdf")
        from pypdf import PdfReader
        return "\n".join(p.extract_text() or "" for p in PdfReader(PDF).pages)
    return out.stdout


def page_count():
    from pypdf import PdfReader
    return len(PdfReader(PDF).pages)


# ---------------------------------------------------------------- 1. numbers

def collect_known_numbers():
    """Every value that may legitimately appear in the report, as a 4dp string."""
    known = set()

    def add(v, digits=4):
        try:
            x = float(v)
        except (TypeError, ValueError):
            return
        if pd.isna(x):
            return
        known.add(f"{x:.{digits}f}")
        known.add(f"{x * 100:.2f}")
        known.add(f"{x:.2f}")
        known.add(f"{x:.1f}")
        if float(x).is_integer():
            known.add(f"{int(x):,}")
            known.add(str(int(x)))

    for name in sorted(os.listdir(RESULTS)):
        if not name.endswith(".csv"):
            continue
        df = pd.read_csv(os.path.join(RESULTS, name))
        for col in df.columns:
            for v in df[col]:
                add(v)
                # "word (ratio=40.83, FP=18, TN=0)" style cells
                if isinstance(v, str):
                    for m in re.findall(r"=([0-9.]+)", v):
                        add(m)
                    # digits inside quoted prompt text, e.g. "Windows 11"
                    for m in re.findall(r"\b(\d+(?:\.\d+)?)\b", v):
                        add(m)
        # rank column of the top-word table
        if name == "ood_fpr_diagnosis.csv":
            for i in range(1, 16):
                add(i)

    with open(os.path.join(RESULTS, "chosen_threshold.txt")) as fh:
        add(fh.read().strip())

    for name in ["train.csv", "val.csv", "test.csv", "ood_jailbreak.csv"]:
        df = pd.read_csv(os.path.join(DATA, name))
        add(len(df))
        for lbl in (0, 1):
            add(int((df["label"] == lbl).sum()))
        add((df["label"] == 1).mean())
        add((df["label"] == 0).mean())
        if "source" in df.columns:
            for src in df["source"].unique():
                sub = df[df["source"] == src]
                add(len(sub))
                for lbl in (0, 1):
                    add(int((sub["label"] == lbl).sum()))

    for sub in ["attacks", "benign_disguised"]:
        d = os.path.join(DATA, sub)
        for name in sorted(os.listdir(d)):
            if name.endswith(".csv"):
                df = pd.read_csv(os.path.join(d, name))
                add(len(df))
                if "method" in df.columns:
                    for v in df["method"].value_counts():
                        add(int(v))
    add(len(pd.read_csv(os.path.join(DATA, "augmented_train_extra.csv"))))

    # LIME weights and probabilities quoted from the saved explanations
    with open(os.path.join(RESULTS, "lime_examples.html"), encoding="utf-8") as fh:
        html = fh.read()
    for m in re.findall(r"P\(Attack\)=([0-9.]+)", html):
        add(m)
    for m in re.findall(r'\["[^"]{1,80}", \d+, (-?[0-9.]+)\]', html):
        add(abs(float(m)))
    return known


# Values that come from source code or are plain arithmetic on reported numbers,
# each listed with where it is defined or how it is derived.
CODE_CONSTANTS = {
    "42": "config.py SEED",
    "256": "config.py MAX_LEN",
    "3,000": "data.py N_DOLLY_BENIGN",
    "3000": "data.py N_DOLLY_BENIGN",
    "0.7": "attacks.py leetspeak substitution probability",
    "10.0": "baseline.py chosen LogisticRegression C",
    "0.5": "baseline.py candidate C grid / default decision threshold",
    "1.0": "baseline.py candidate C grid",
    "2.0": "baseline.py candidate C grid / threshold.py FPR bound in percent",
    "5.0": "baseline.py candidate C grid",
    "1000": "baseline.py max_iter",
    "1,000": "augment_experiment.py 1000 attack and 1000 benign augmented rows",
    "2": "baseline.py word min_df / number of TF-IDF vectorisers",
    "3": "baseline.py char min_df / train.py epochs / number of sources",
    "5": "attacks.py number of disguises and templates",
    "6": "train.py DistilBERT layer count / explain.py num_features",
    "1": "attacks.py leet map and label 1",
    "4": "attacks.py leet map",
    "0": "label 0",
    "7": "attacks.py leet map",
    "25": "attacks.py Cyrillic homoglyph map size",
    "27": "attacks.py HINDI_PHRASES rule count",
    "32": "train.py batch size",
    "0.01": "train.py weight decay",
    "0.1": "train.py warmup ratio",
    "350": "explain.py num_samples",
    "91": "threshold.py np.linspace(0.05, 0.95, 91)",
    "0.05": "threshold.py sweep lower bound",
    "0.95": "threshold.py sweep upper bound",
    "0.06": "results/chosen_threshold.txt",
    "0.02": "threshold.py FPR bound",
    "200": "long_input_check.py chunk stride / augment_experiment.py sample size",
    "254": "long_input_check.py content tokens per window",
    "40": "shortcut_check.py prompt count",
    "60": "chat_fpr_check.py prompt count",
    "15": "chat_fpr_check.py prompts per category / top-15 word list",
    "9": "explain.py example count",
    "64": "guard.py and long_input_check.py batch size",
    "4": "number of research questions",
    "70": "data.py 70/15/15 split",
    "150": "app.py max_new_tokens",
    "2.5": "Qwen2.5-0.5B-Instruct model name",
    "0.5737": "results/augment_results.csv",
    "1910.01108": "reference [1] arXiv id",
    "1602.04938": "reference [2] arXiv id",
    "2019": "reference [1] year",
    "2016": "reference [2] year",
    "1135": "reference [2] pages",
    "1144": "reference [2] pages",
    "22": "reference [2] conference number",
    "2026": "report date",
    "16": "report date",
    "504": "course code DSL504",
    "66": "DistilBERT parameter scale mention",
    "134": "multilingual DistilBERT parameter scale mention",
    "0430": "attacks.py Cyrillic codepoint U+0430",
    "043": "attacks.py Cyrillic codepoint U+043E",
    "0.53": "100 / 190, one prompt as a share of a 190-row disguise set",
    "3.53": "0.9631 - 0.9278 in percentage points",
    "2.35": "0.3997 - 0.3762 in percentage points",
    "1.00": "0.8180 - 0.8080 in percentage points",
    "0.98": "clean test F1 stated as 'above 0.98'",
    "15k": "dataset name databricks-dolly-15k",
    "2": "number of leading causes discussed",
}


def strip_navigation(text):
    """Remove section, table, figure, page and citation numbers: they are navigation, not data."""
    t = text
    t = re.sub(r"\bSections?\s+\d+(\.\d+)?(\s*(,|and|or|to)\s*\d+(\.\d+)?)*", " ", t)
    t = re.sub(r"\b(Table|Figure)s?\s+\d+(\s*(,|and|to)\s*\d+)*", " ", t)
    t = re.sub(r"(?m)^\s*\d+(\.\d+)*\.\s", " ", t)
    t = re.sub(r"(?m)^\s*\d{1,3}\s*$", " ", t)
    t = re.sub(r"\[\d+\]", " ", t)
    t = re.sub(r"arXiv[:\s]*\d+\.\d+", " ", t)
    t = re.sub(r"U\+[0-9A-F]{4}", " ", t)
    return t


def check_numbers(text, known):
    """Every numeric token in the report must be traceable to a file or to code."""
    body = strip_navigation(text)
    bad = {}
    for raw in re.findall(r"(?<![A-Za-z0-9_./-])(\d+(?:,\d{3})*(?:\.\d+)?)(?![\d.]*[A-Za-z])", body):
        tok = raw.strip()
        if tok in CODE_CONSTANTS:
            continue
        plain = tok.replace(",", "")
        if plain in known or tok in known:
            continue
        # also accept the plain integer / comma form of a known value
        try:
            as_f = float(plain)
        except ValueError:
            continue
        if f"{as_f:.4f}" in known or f"{as_f:.2f}" in known or f"{as_f:.1f}" in known:
            continue
        if as_f.is_integer() and (str(int(as_f)) in known or f"{int(as_f):,}" in known):
            continue
        bad[tok] = bad.get(tok, 0) + 1
    if bad:
        for tok, count in sorted(bad.items(), key=lambda kv: -kv[1]):
            fail(f"number not traceable to a file or to code: {tok!r} (x{count})")
    else:
        ok(f"every numeric token traces to results/, data/ or a documented code constant")


def check_tables_against_csv(text):
    """Spot-check headline values straight out of the CSVs."""
    checks = []

    main = pd.read_csv(os.path.join(RESULTS, "main_results.csv"))
    for _, r in main.iterrows():
        for col in ["recall", "false_positive_rate", "f1"]:
            if not pd.isna(r[col]):
                checks.append((f"main_results {r['model']}/{r['test_set']}/{col}",
                               f"{r[col]:.4f}"))

    ben = pd.read_csv(os.path.join(RESULTS, "benign_disguise_check.csv"))
    for _, r in ben.iterrows():
        checks.append((f"benign_disguise {r['model']}/{r['disguise']}",
                       f"{r['false_positive_rate']:.4f}"))

    lng = pd.read_csv(os.path.join(RESULTS, "long_input_check.csv"))
    for _, r in lng.iterrows():
        for col in ["recall", "false_positive_rate", "precision", "f1"]:
            checks.append((f"long_input {r['strategy']}/{col}", f"{r[col]:.4f}"))

    aug = pd.read_csv(os.path.join(RESULTS, "augment_results.csv"))
    for _, r in aug.iterrows():
        checks.append((f"augment {r['model']}/{r['eval_set']}", f"{r['score']:.4f}"))

    thr = pd.read_csv(os.path.join(RESULTS, "threshold_summary.csv"))
    for _, r in thr.iterrows():
        checks.append((f"threshold {r['threshold']}/{r['eval_set']}/{r['metric']}",
                       f"{r['value']:.4f}"))

    demo = pd.read_csv(os.path.join(RESULTS, "demo_check.csv"))
    for _, r in demo.iterrows():
        checks.append((f"demo {r['guard']}", f"{r['guard_score']:.4f}"))

    out = pd.read_csv(os.path.join(RESULTS, "output_check.csv"))
    for _, r in out.iterrows():
        checks.append((f"output_check score {r['prompt'][:24]}", f"{r['guard_score']:.4f}"))

    missing = [name for name, val in checks if val not in text]
    if missing:
        for name in missing[:25]:
            fail(f"CSV value missing from the report: {name}")
        if len(missing) > 25:
            fail(f"... and {len(missing) - 25} more missing CSV values")
    else:
        ok(f"all {len(checks)} CSV values from the six result tables appear in the report")


# ---------------------------------------------------------------- 2. lexicon

BANNED_WORDS = [
    "catastrophic", "outstanding", "massive", "stark", "rigorous", "robust",
    "comprehensive", "seamless", "production-ready", "dependable",
    "groundbreaking", "flawless",
]
BANNED_PHRASES = [
    "it is worth noting", "leverage", "delve", "in conclusion", "this ensures",
]
TOOL_WORDS = [
    "artificial intelligence", "\\bai\\b", "ai-", "assistant tool", "chatgpt",
    "copilot", "claude", "gpt-", "llm-generated", "auto-generated",
    r"generated by (an? )?(ai|assistant|model|tool|bot)\b",
    "language model assistant", "prompt engineer",
    "figma", "canva",
]


def check_lexicon(text, source_files):
    low = text.lower()

    hits = [w for w in BANNED_WORDS if re.search(r"\b" + re.escape(w) + r"\b", low)]
    hits += [p for p in BANNED_PHRASES if p in low]
    if hits:
        for h in hits:
            fail(f"banned word or filler phrase in the report text: {h!r}")
    else:
        ok("no banned hype words or filler phrases in the report text")

    if "—" in text:
        n = text.count("—")
        fail(f"em dash found in the report text ({n} occurrences)")
    else:
        ok("no em dashes in the report text")

    emoji = re.findall(
        "[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF⬀-⯿]",
        text)
    if emoji:
        fail(f"emoji found in the report text: {set(emoji)}")
    else:
        ok("no emoji in the report text")

    # "assistant" is legitimate project vocabulary; only flag tooling claims
    for pat in TOOL_WORDS:
        for m in re.finditer(pat, low):
            ctx = low[max(0, m.start() - 45):m.end() + 45].replace("\n", " ")
            if pat == "\\bai\\b" and ("ai " not in ctx and " ai" not in ctx):
                continue
            fail(f"possible tool or assistant mention: ...{ctx}...")
    else:
        pass
    if not any(re.search(p, low) for p in TOOL_WORDS):
        ok("no mention of assistants, generation tools or design tools in the report text")

    # the same sweep over every source file in report/
    for path in source_files:
        with open(path, encoding="utf-8") as fh:
            src = fh.read()
        slow = src.lower()
        if "—" in src:
            fail(f"em dash in {os.path.basename(path)}")
        for pat in TOOL_WORDS:
            if re.search(pat, slow):
                m = re.search(pat, slow)
                ctx = slow[max(0, m.start() - 40):m.end() + 40].replace("\n", " ")
                fail(f"tool or assistant mention in {os.path.basename(path)}: ...{ctx}...")
        for w in BANNED_WORDS:
            # skip comments quoting nothing; a plain word hit is still a hit
            if re.search(r"\b" + re.escape(w) + r"\b", slow):
                fail(f"banned word {w!r} in {os.path.basename(path)}")
    ok(f"swept {len(source_files)} source files in report/ for the same terms")


def check_no_inline_bold(typ_path):
    """Bold is allowed for headings, table headers and the contribution labels only."""
    with open(typ_path, encoding="utf-8") as fh:
        lines = fh.readlines()
    allowed = re.compile(r"^\s*\*(Samarth|Prashant|Sarthak|Shivam|Priyanshu)\.\*")
    bad = []
    for i, line in enumerate(lines, 1):
        if allowed.match(line):
            continue
        for m in re.finditer(r"(?<!\\)\*([^*\n]{1,120})\*", line):
            bad.append((i, m.group(1)))
    if bad:
        for i, t in bad:
            fail(f"bold text inside a paragraph at {os.path.basename(typ_path)}:{i}: {t!r}")
    else:
        ok("no bold runs inside paragraphs (headings and contribution labels excepted)")


# ---------------------------------------------------------------- 3. structure

def check_structure(text, pages):
    if 12 <= pages <= 18:
        ok(f"page count {pages} is within the 12 to 18 range")
    else:
        fail(f"page count {pages} is outside the 12 to 18 range")

    if "Registration Number" in text and "Full Name" in text:
        ok("member table present on the title page")
    else:
        fail("member table missing from the title page")

    if "PENDING" in text:
        fail("member table still contains PENDING placeholders")
    else:
        ok("member table has no placeholder rows")

    required = ["Abstract", "Introduction and Problem Statement",
                "Dataset Design and Confounding", "Methodology",
                "Experimental Setup", "Results", "Analysis and Discussion",
                "Interactive Demo Walkthrough", "Limitations", "Conclusion",
                "Team Contributions", "References"]
    missing = [s for s in required if s not in text]
    if missing:
        for s in missing:
            fail(f"required section missing: {s}")
    else:
        ok(f"all {len(required)} required sections present")

    # every numbered table and figure must be referred to in the body text
    tables = sorted({int(m) for m in re.findall(r"Table (\d+):", text)})
    figures = sorted({int(m) for m in re.findall(r"Figure (\d+):", text)})
    for n in tables:
        refs = len(re.findall(rf"Table {n}\b", text))
        if refs < 2:
            fail(f"Table {n} has a caption but is never referred to in the body text")
    for n in figures:
        refs = len(re.findall(rf"Figure {n}\b", text))
        if refs < 2:
            fail(f"Figure {n} has a caption but is never referred to in the body text")
    uncited = [n for n in tables if len(re.findall(rf"Table {n}\b", text)) < 2]
    uncited += [n for n in figures if len(re.findall(rf"Figure {n}\b", text)) < 2]
    if tables and figures and not uncited:
        ok(f"{len(tables)} tables and {len(figures)} figures, each cited in the body text")

    for img in ["augment_results.png", "threshold_curve.png",
                "lime_examples.png", "demo_screenshot.png", "demo_output_check.png"]:
        if img in text:
            ok(f"figure embedded and named: {img}")
        else:
            fail(f"required figure not embedded: {img}")


def check_no_overflow(text):
    """Flag any line that runs past the typeset measure, a sign of a wide table."""
    long_lines = [l for l in text.splitlines() if len(l.rstrip()) > 200]
    if long_lines:
        for l in long_lines[:5]:
            fail(f"line longer than the page measure, possible overflow: {l[:90]}...")
    else:
        ok("no over-wide lines in the extracted text")


def main():
    if not os.path.exists(PDF):
        print("PDF not found, run build_report.py first")
        sys.exit(1)

    text = pdf_text()
    pages = page_count()

    print("\n-- 1. numerical accuracy --")
    known = collect_known_numbers()
    check_tables_against_csv(text)
    check_numbers(text, known)

    print("\n-- 2. lexicon audit --")
    # audit_report.py necessarily contains the banned list itself, so it is not swept
    sources = [os.path.join(HERE, f) for f in sorted(os.listdir(HERE))
               if f.endswith((".typ", ".py")) and f != "audit_report.py"]
    check_lexicon(text, sources)
    check_no_inline_bold(os.path.join(HERE, "BREACHZERO_Report.typ"))

    print("\n-- 3. structural verification --")
    check_structure(text, pages)
    check_no_overflow(text)

    print("\n" + "-" * 60)
    if failures:
        print(f"{len(failures)} check(s) failed:")
        for f in failures:
            print("  - " + f)
        sys.exit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
