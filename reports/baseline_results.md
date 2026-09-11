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

## Retrieval diagnostic

| Retrieval system | Recall@1 | Recall@3 | Recall@5 |
|---|---:|---:|---:|
| TF-IDF lexical retrieval | 0.0800 | 0.1350 | 0.1450 |
| Intent-aware reranking | 0.0900 | 0.1350 | 0.1500 |

The intent-aware reranker produced only a modest improvement over lexical
TF-IDF retrieval.

These values are diagnostic rather than definitive retrieval-ground-truth
metrics because historical retrieved interactions do not have human intent
labels. Historical intents were inferred using high-precision weak
supervision, while the golden query intent comes from the independently
human-labelled evaluation set.

The result suggests that lexical overlap alone is insufficient for robust
support-case retrieval, motivating semantic intent classification and
stronger retrieval/reranking in the main system.