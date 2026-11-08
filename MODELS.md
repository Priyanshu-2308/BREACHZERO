# Model links

Large model weights/checkpoints are gitignored (see `.gitignore`). Record download links here as they're produced.

| Model | Location | Notes |
|---|---|---|
| baseline (TF-IDF + LogReg) | `models/baseline.joblib` | committed to the repo, no download needed |
| DistilBERT (English) | https://huggingface.co/<HF_USERNAME>/breachzero-distilbert | fine-tuned `distilbert-base-uncased`, goes in `models/distilbert` |
| DistilBERT (multilingual) | https://huggingface.co/<HF_USERNAME>/breachzero-distilbert-multilingual | fine-tuned `distilbert-base-multilingual-cased`, goes in `models/distilbert_multilingual` |

Download each into the right folder:

```bash
pip install -U huggingface_hub
huggingface-cli download <HF_USERNAME>/breachzero-distilbert --local-dir models/distilbert
huggingface-cli download <HF_USERNAME>/breachzero-distilbert-multilingual --local-dir models/distilbert_multilingual
```

Then `python run_eval.py` picks both models up automatically.
