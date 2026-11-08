# Model links

Large model weights/checkpoints are gitignored (see `.gitignore`). Record download links here as they're produced.

| Model | Location | Notes |
|---|---|---|
| baseline (TF-IDF + LogReg) | `models/baseline.joblib` | committed to the repo, no download needed |
| DistilBERT (English) | https://huggingface.co/sarthak776/breachzero-distilbert | fine-tuned `distilbert-base-uncased`, goes in `models/distilbert` |
| DistilBERT (multilingual) | https://huggingface.co/sarthak776/breachzero-distilbert-multilingual | fine-tuned `distilbert-base-multilingual-cased`, goes in `models/distilbert_multilingual` |

Download each into the right folder:

```python
from huggingface_hub import snapshot_download

snapshot_download("sarthak776/breachzero-distilbert", local_dir="models/distilbert")
snapshot_download("sarthak776/breachzero-distilbert-multilingual", local_dir="models/distilbert_multilingual")
```

Then `python run_eval.py` picks both models up automatically.
