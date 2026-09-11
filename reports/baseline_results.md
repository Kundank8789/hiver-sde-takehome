# Baseline Results

## Evaluation setup

All models are evaluated on the frozen 200-example human-labelled
golden set.

The golden set is never used for model training.

---

## Intent classification

| System | Accuracy | Macro-F1 |
|---|---:|---:|
| Trivial majority baseline | 0.1000 | 0.0182 |
| TF-IDF + Logistic Regression | 0.1100 | 0.0874 |

### TF-IDF training

The TF-IDF classifier was trained on a separate weakly-labelled
historical interaction corpus.

The weak labels were produced using high-precision lexical rules.
Ambiguous examples were rejected.

92 golden-set interaction IDs were found in the weak training corpus
and explicitly removed before training.

The training set was capped at 2,500 examples per class.

---

## Interpretation

The TF-IDF baseline provides only a small improvement over the trivial
baseline on the frozen benchmark.

This suggests that the weak lexical supervision does not transfer
well to the human-labelled evaluation set, especially for ambiguous
and semantically overlapping support issues.

The strongest baseline performance was observed for:
- PERFORMANCE_STABILITY
- APPS_MEDIA
- FEATURE_HOW_TO
- IOS_UPDATE

The baseline failed to reliably identify:
- ACCOUNT_SECURITY
- DEVICE_HARDWARE
- OTHER_UNCLEAR