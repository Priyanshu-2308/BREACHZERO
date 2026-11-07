# BREACHZERO: Prompt-Injection Detector for LLM Apps

A guard classifier that sits in front of an LLM app and flags prompt-injection inputs
("ignore previous instructions and...") before they reach the model. We compare a
TF-IDF + Logistic Regression baseline against a fine-tuned DistilBERT, test robustness on
disguised attacks and unseen attack styles, apply an adversarial-augmentation fix, and demo
the guard in a Gradio chat where a small LLM protects a secret password.

## Setup

```bash
pip install -r requirements.txt
```

## Run order

```bash
python data.py        # build train/val/test/ood_jailbreak CSVs from Hugging Face datasets
python attacks.py      # generate disguised-attack variants (leetspeak, cyrillic, base64, wrapped, hindi)
python baseline.py     # TF-IDF + Logistic Regression baseline
python train.py        # fine-tune DistilBERT
python run_eval.py     # evaluate baseline vs DistilBERT, incl. OOD/disguised-attack robustness
python app.py          # Gradio demo: LLM guarding a secret password, with a "Guard on" checkbox
```
