# TASK: Flatten the repo layout

**Repo:** https://github.com/Priyanshu-2308/BREACHZERO (clone it; you have write access)

## Context
All project files currently sit inside `PS_1/` and `PS_2/` subfolders. `PS_2/` is a superset of `PS_1/`: it contains identical copies of `config.py`, `data.py`, `requirements.txt`, `README.md`, `MODELS.md` and `data/*.csv`, plus `attacks.py`, `baseline.py`, `metrics.py`, `data/attacks/*.csv`, `models/baseline.joblib` and `results/baseline_metrics.csv`.

The project contract requires everything at the **repo root**: `config.py`, `data.py`, `attacks.py`, `baseline.py`, `metrics.py`, `requirements.txt`, `README.md`, `MODELS.md`, `data/`, `models/`, `results/`. Other teammates' scripts and instructions assume these root-level paths.

## Rules
- Work on `main`. Do exactly the steps below, no extra refactoring, no content changes to any file.
- Do not delete anything until step 2 confirms the copies are identical.
- Before pushing: `git pull --rebase origin main && git push origin main`.

## Steps
1. `git pull --rebase origin main`
2. Verify the PS_1 and PS_2 copies are identical before deleting anything: run `diff -r PS_1/data PS_2/data` (only `data/attacks` should differ), and diff every `.py`, `.txt` and `.md` file that exists in both folders. If ANY shared file differs, STOP and report the difference to the user.
3. `mkdir docs && git mv PS_1/0_ProblemStatement.md docs/PS1_ProblemStatement.md`
4. Move the PS_2 contents to the repo root with `git mv`: `attacks.py baseline.py metrics.py config.py data.py requirements.txt README.md MODELS.md`, and the folders `data/`, `models/`, `results/`.
5. `git rm -r PS_1`, and make sure `PS_2/` no longer exists.
6. Do not edit `.gitignore`. Its patterns (`models/distilbert*/`, `*.safetensors`, `*.bin`, `*.pt`, `__pycache__/`, `.ipynb_checkpoints/`) are correct once the files are at root.
7. Verify from the repo root: `python -c "import config, metrics"` works, and `python baseline.py` runs and reproduces `results/baseline_metrics.csv` unchanged (check with `git diff`; ignore line-ending differences).
8. `git commit -m "Flatten repo to root; move problem statements to docs/"`, then `git pull --rebase origin main && git push origin main`.
9. Report the final `ls -R` of the repo root (excluding `.git`) in your final message.

## Hand-off
Repo root contains all files listed in the Context section, and `PS_1/` and `PS_2/` are gone. Tell the user as soon as it's pushed, since the next teammates are waiting on this.
