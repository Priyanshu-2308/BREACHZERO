# BREACHZERO: Prompt-Injection Detector for LLM Apps (NLP course project)

**Repo:** https://github.com/Priyanshu-2308/BREACHZERO (clone it; you have write access)

## Goal
Build a guard classifier that sits in front of an LLM app and flags prompt-injection inputs ("ignore previous instructions and...") before they reach the model. We compare a TF-IDF + Logistic Regression baseline against a fine-tuned DistilBERT, then test robustness on disguised attacks and on unseen attack styles (the main finding), apply a fix (adversarial augmentation), and demo it in a Gradio chat where a small LLM guards a secret password with a "Guard on" checkbox.

## Team workflow (5 people, strictly sequential hand-offs, deadline ~10 days)
Order: **P1 (data) -> P2 (attacks + baseline) -> P3 (DistilBERT + eval) -> P4 (experiments) -> P5 (explainability + demo + report)**.

You may only start after the previous person has pushed their files. Before starting, run `git pull --rebase origin main` and CHECK that the files you depend on exist. If something you depend on is missing or malformed, STOP and tell the user instead of inventing it.

## Git rules (do not deviate)
- Work on `main`. Only create/edit the files listed under YOUR section. Never rewrite or reformat other people's files. If you need a change in someone else's file, tell the user.
- Commit small and often. Before every push: `git pull --rebase origin main && git push origin main`.
- Never commit model weights, checkpoints, or files over 20 MB. `models/distilbert*/`, `*.safetensors`, `*.bin`, `*.pt` are gitignored. Share big models via the Hugging Face Hub or Google Drive and record the link in `MODELS.md`.
- Small CSVs (< 20 MB) in `data/` and `results/` ARE committed so the next person can use them.

## Shared conventions (the contract between teammates)
- Python 3.10+, runs on Google Colab free tier (T4 GPU). Pin versions in `requirements.txt`.
- `config.py` (created by P1) is the single source of paths/constants. Import from it, never hardcode: `SEED=42`, `MAX_LEN=256`, `BASE_MODEL="distilbert-base-uncased"`, `MULTI_MODEL="distilbert-base-multilingual-cased"`, `DATA_DIR="data"`, `ATTACK_DIR="data/attacks"`, `MODEL_DIR="models"`, `RESULTS_DIR="results"`.
- Label convention everywhere: **label = 1 means injection/attack, label = 0 means benign.**
- All datasets are CSVs with at least columns `text`, `label`. Extra columns allowed.
- File layout:
  - `data/train.csv`, `data/val.csv`, `data/test.csv` (columns: text,label,source)
  - `data/ood_jailbreak.csv` (columns: text,label,source; never used for training)
  - `data/attacks/{leetspeak,cyrillic,base64,wrapped,hindi}.csv` (columns: text,label,original_text,attack_type; label always 1)
  - `models/baseline.joblib` (committed, small)
  - `models/distilbert/` and `models/distilbert_multilingual/` (NOT committed)
  - `results/*.csv`, `results/*.png`
- Do NOT assume Hugging Face dataset column names. Load each dataset, print its columns and a few rows, and adapt. If a dataset can't be downloaded, say so; do not fabricate data.
- Report real numbers only. Never hardcode or invent metrics.
- Keep code simple, commented, and runnable top-to-bottom as scripts: `python <script>.py`. Each script must print a short summary when it finishes.

---

# YOUR ROLE: Person 1 of 5 (Repo foundation + dataset)

You build the foundation everyone else depends on. Be fast and push within ~1 day, since the next person is blocked until you do.

## Files you own
`requirements.txt`, `config.py`, `.gitignore`, `README.md` (skeleton only), `MODELS.md` (empty stub), `data.py`, and the generated files `data/train.csv`, `data/val.csv`, `data/test.csv`, `data/ood_jailbreak.csv`. Create folders `data/`, `data/attacks/`, `models/`, `results/` with a `.gitkeep` in each.

## Tasks
1. First check whether the repo already contains any of these files (a scaffold may exist). If so, reuse and fix them instead of rewriting. If not, create them.
2. `requirements.txt`: datasets, transformers, torch, scikit-learn, pandas, numpy, joblib, gradio, lime, matplotlib, accelerate. Pin versions that work on Colab.
3. `config.py` with exactly the constants listed in the Shared conventions.
4. `.gitignore`: `models/distilbert*/`, `*.safetensors`, `*.bin`, `*.pt`, `__pycache__/`, `.ipynb_checkpoints/`.
5. `data.py` builds the training data from Hugging Face:
   - **Injections (label 1):** `deepset/prompt-injections` (keep its label-1 rows) and `Lakera/gandalf_ignore_instructions` (all rows are attacks).
   - **Benign (label 0):** `deepset/prompt-injections` label-0 rows plus 3000 random prompts from `databricks/databricks-dolly-15k` (use the instruction field), seeded with SEED.
   - **Out-of-distribution set:** `jackhhao/jailbreak-classification` -> `data/ood_jailbreak.csv` (map type "jailbreak" -> 1, "benign" -> 0). These rows must NEVER appear in train/val/test.
   - Inspect actual column names first and adapt. Drop empty rows and exact duplicates, strip whitespace.
   - Stratified split 70/15/15 into train/val/test using SEED. Add a `source` column.
   - Print class counts per split and per source.
6. Run it (Colab is fine), then commit the generated CSVs.
7. `README.md` skeleton: project title, one-paragraph description, setup instructions (`pip install -r requirements.txt`) and the run order: data.py, attacks.py, baseline.py, train.py, run_eval.py, app.py.

## Hand-off (confirm before finishing)
All four CSVs pushed with columns `text,label,source`; both labels present in every split; `config.py` importable. Print the split sizes in your final message so the next person can see them.
