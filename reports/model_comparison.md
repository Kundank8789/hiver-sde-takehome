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


## Reply-quality judge validation

A 30-example sample was independently rated by a human reviewer on:

- overall quality
- groundedness
- relevance
- helpfulness

The same examples were scored by the LLM judge.

Results:

| Metric | Result |
|---|---:|
| Exact agreement | 6.67% |
| Within ±1 point | 10.00% |
| Quadratic weighted kappa | -0.0584 |
| Spearman correlation | -0.1989 |
| Unsupported-claim rate | 0.00% |
| Judge runtime errors | 0 |

The low agreement indicates that the judge should not be treated as a
high-confidence substitute for human evaluation. It is retained as a
diagnostic tool rather than a ground-truth quality metric.