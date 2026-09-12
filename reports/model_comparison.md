# Model Comparison

| System | Accuracy | Macro F1 | Notes |
|---|---:|---:|---|
| Trivial baseline | 0.1000 | 0.0182 | Always predicts ACCOUNT_SECURITY |
| TF-IDF + Logistic Regression | 0.1100 | 0.0874 | Weakly-labelled training |
| Initial Groq classifier | 0.0900 | 0.0881 | Zero-shot structured classification |
| Hybrid classifier | 0.1150 | 0.0914 | Rules + historical retrieval |

## Interpretation

The hybrid classifier provides only a modest improvement over the
classical baselines. The relatively low absolute scores reflect the
noisy Twitter language and overlap between the manually-defined intents.

The hybrid system is therefore used as a routing component rather than
treated as a perfect classifier. Low-confidence or unresolved cases are
eligible for human escalation.