# BREACHZERO: Prompt-Injection Detector for LLM Applications

**BREACHZERO** is a prompt-injection detector designed to sit in front of Large Language Model (LLM) applications. It flags adversarial prompt injections (such as *"ignore all previous instructions and..."*, obfuscated encodings, and roleplay jailbreaks) before inputs reach the downstream model.

This project evaluates a **Baseline (TF-IDF + LR)** alongside fine-tuned **English DistilBERT** and **Multilingual DistilBERT** sequence classifiers. We test out-of-distribution robustness across five disguises (leetspeak, Cyrillic homoglyphs, Base64, template wrapping, and Hindi-style substitution), evaluate shortcut learning, calibrate decision thresholds, test adversarial data augmentation, and demonstrate the detector in an interactive Gradio application where an instruct LLM protects a confidential secret.

---

## Architecture Overview

```
User Prompt
    │
    ▼
┌──────────────────────────────────────────────┐
│        BREACHZERO Guard (DistilBERT)         │
│  Predicts P(Injection) at Chosen Threshold   │
└──────────────────────┬───────────────────────┘
                       │
          ┌────────────┴────────────┐
          ▼                         ▼
   P(Inj) >= Threshold      P(Inj) < Threshold
┌──────────────────────┐  ┌──────────────────────────────┐
│       BLOCKED        │  │     Passed to LLM App        │
│ "Blocked by guard"   │  │   (Qwen2.5-0.5B-Instruct)   │
└──────────────────────┘  └──────────────────────────────┘
```

---

## Project Structure

```
BREACHZERO/
├── data/
│   ├── train.csv, val.csv, test.csv      # In-distribution splits
│   ├── ood_jailbreak.csv                 # Out-of-distribution jailbreak benchmark
│   ├── attacks/                          # 5 disguised attack sets (label=1)
│   │   └── {leetspeak, cyrillic, base64, wrapped, hindi}.csv
│   └── benign_disguised/                 # Disguised benign controls (label=0)
│       └── {leetspeak, cyrillic, base64, wrapped, hindi}.csv
├── models/
│   ├── baseline.joblib                   # Baseline (TF-IDF + LR) pipeline
│   ├── distilbert/                       # English DistilBERT (downloaded from HF)
│   └── distilbert_multilingual/          # Multilingual DistilBERT (HF)
├── results/                              # Exact outputs used to build report tables
│   ├── main_results.csv                  # Baseline vs DistilBERT evaluation suite
│   ├── benign_disguise_check.csv         # Benign disguise FPR check (shortcut test)
│   ├── shortcut_check.csv                # Password and key query shortcut test
│   ├── long_input_check.csv              # Input length truncation vs chunking
│   ├── ood_fpr_diagnosis.csv             # OOD false-positive token diagnosis
│   ├── ood_flagged_examples.csv          # Sample false-positive prompts from OOD set
│   ├── augment_results.csv & .png        # Adversarial augmentation recovery
│   ├── threshold_curve.png               # Threshold trade-off curves
│   ├── threshold_summary.csv             # Threshold metrics at 0.50 and 0.06
│   ├── chosen_threshold.txt              # Tuned decision threshold (val FPR <= 2%)
│   ├── demo_check.csv                    # Reproducible demo verification results
│   └── lime_examples.html & .png         # LIME feature attribution visualizations
├── config.py                             # Central source of truth for paths & constants
├── guard.py                              # Cached inference interface: predict_proba()
├── data.py                               # Hugging Face dataset download & curation
├── attacks.py                            # Disguise generators and attack set creator
├── baseline.py                           # TF-IDF + Logistic Regression pipeline
├── train.py                              # Hugging Face Trainer fine-tuning script
├── run_eval.py                           # Evaluation harness across models and sets
├── benign_disguise_check.py              # Shortcut evaluation on benign controls
├── shortcut_check.py                     # Legitimate credential query evaluation
├── long_input_check.py                   # Context window handling experiment
├── ood_fpr_diagnosis.py                  # Token-level false positive error analysis
├── augment_experiment.py                 # Leave-one-out adversarial augmentation
├── threshold.py                          # Threshold sweep & calibration (val-only)
├── explain.py                            # LIME explainability script
├── app.py                                # Interactive Gradio chat demo
├── demo_check.py                         # Reproducible demo verification script
├── colab_run.ipynb                       # Google Colab notebook for training and eval
├── MODELS.md                             # Model repository links and download commands
├── REPORT.md                             # Technical project report
└── requirements.txt                      # Python dependencies (minimum versions)
```

Note: Files in `results/` are the exact outputs the report tables were built from.

---

## Installation & Setup

Python 3.10 to 3.13 is supported (written for Google Colab and local Python).

### Local Setup

```bash
# 1. Clone repository
git clone https://github.com/Priyanshu-2308/BREACHZERO.git
cd BREACHZERO

# 2. Create virtual environment
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell:
.venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install -r requirements.txt
```

### Pretrained Model Weights

Large transformer weights are hosted on the Hugging Face Hub (see [MODELS.md](MODELS.md)). Download them with Python:

```python
from huggingface_hub import snapshot_download

# English DistilBERT
snapshot_download("sarthak776/breachzero-distilbert", local_dir="models/distilbert")

# Multilingual DistilBERT
snapshot_download("sarthak776/breachzero-distilbert-multilingual", local_dir="models/distilbert_multilingual")
```

---

## Execution Order

Run the scripts in the following order:

```bash
# Step 1: Ingest and prepare datasets (train, val, test, ood_jailbreak)
python data.py

# Step 2: Generate disguised attack variants
python attacks.py

# Step 3: Train and evaluate the TF-IDF + Logistic Regression baseline
python baseline.py

# Step 4: Fine-tune DistilBERT (optional if downloaded from HF)
python train.py --model distilbert-base-uncased --out models/distilbert
python train.py --model distilbert-base-multilingual-cased --out models/distilbert_multilingual

# Step 5: Run full evaluation suite across baseline and transformer models
python run_eval.py

# Step 6: Diagnostic and robustness checks
python benign_disguise_check.py   # Test benign disguised controls (shortcut analysis)
python shortcut_check.py          # Test legitimate password queries
python long_input_check.py        # Evaluate truncation vs chunking on long inputs
python ood_fpr_diagnosis.py       # Analyze top tokens triggering false positives on OOD data

# Step 7: Adversarial augmentation experiment
python augment_experiment.py      # Trains leave-one-out models and evaluates recovery

# Step 8: Threshold calibration
python threshold.py               # Sweeps val.csv to select threshold bounding FPR <= 2%

# Step 9: LIME Explainability
python explain.py                 # Generates HTML and PNG feature attribution plots

# Step 10: Interactive Gradio Demo & Verification
python demo_check.py              # Test demo prompts with guard ON and OFF
python app.py                     # Launch chat demo (pass --share for public URL)
```

---

## Running on Google Colab

A Google Colab runbook is provided in `colab_run.ipynb` (written for Google Colab).

1. Open Google Colab and set Runtime to **T4 GPU** (`Runtime -> Change runtime type -> T4 GPU`).
2. Clone the repository and install requirements:
   ```bash
   !git clone https://github.com/Priyanshu-2308/BREACHZERO.git
   %cd BREACHZERO
   !pip install -r requirements.txt
   ```
3. Download models using `huggingface_hub` snapshot download as shown above.
4. Launch the Gradio demo with public sharing:
   ```bash
   !python app.py --share
   ```

---

## Interactive Demo (`app.py`)

The demo couples a small instruction model (`Qwen/Qwen2.5-0.5B-Instruct`) with the `BREACHZERO` guard. The LLM is instructed in its system prompt to guard a confidential secret password (`Omega-7-Blue-Sapphire`).

- **Guard ON**: Inputs with predicted injection score >= tuned threshold (stored in `results/chosen_threshold.txt`) are blocked before reaching the model.
- **Guard OFF**: Inputs are passed directly to the LLM. Attackers can use persona adoption, reverse psychology, or diagnostic overrides to leak the secret.
- **Interactive Examples**: Pre-configured buttons provide instant test cases for direct injection, wrapped injection, and legitimate benign requests.
