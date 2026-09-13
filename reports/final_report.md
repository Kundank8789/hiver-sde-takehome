# Hiver SDE Take-Home — AI Support Agent

## 1. Executive Summary

This project builds an AI-assisted AppleSupport customer-support agent
that classifies the customer's primary support intent, retrieves
historically similar support interactions, decides whether the case
should be automated or escalated, and drafts a historically grounded
response.

The system deliberately separates:
- intent classification
- deterministic escalation
- historical retrieval
- response generation

The frozen evaluation set contains 200 human-labelled examples across
10 support intents.

## 2. Dataset and Golden Evaluation Set

The source dataset contained approximately 2.8 million tweets.

After identifying direct AppleSupport interactions, the pipeline produced
106,648 usable support interactions.

A balanced 200-example golden evaluation set was created with 20 examples
per intent and frozen before final evaluation.

The ten intents are:

- IOS_UPDATE
- BATTERY_POWER
- PERFORMANCE_STABILITY
- CONNECTIVITY
- APPS_MEDIA
- DEVICE_HARDWARE
- FEATURE_HOW_TO
- ACCOUNT_SECURITY
- PURCHASE_REPAIR_WARRANTY
- OTHER_UNCLEAR

The golden set was kept separate from weakly-labelled training data and
historical retrieval.

## 3. Baselines and Classifier Experiments

| System | Accuracy | Macro F1 |
|---|---:|---:|
| Trivial baseline | 10.00% | 1.82% |
| TF-IDF + Logistic Regression | 11.00% | 8.74% |
| Initial Groq classifier | 9.00% | 8.81% |
| Hybrid classifier | 11.50% | 9.14% |

The hybrid system combines high-precision deterministic rules with
historical retrieval and optional LLM fallback.

The hybrid classifier provides a modest improvement over the classical
baselines, but absolute performance remains low.

A major reason is noisy Twitter language and substantial overlap between
the manually defined intent categories.

## 4. Retrieval

Leakage-safe TF-IDF retrieval was evaluated on the frozen 200-example
golden set.

| Retrieval method | Recall@1 | Recall@3 | Recall@5 |
|---|---:|---:|---:|
| Lexical | 8.0% | 13.5% | 14.5% |
| Intent-aware reranking | 9.0% | 13.5% | 15.0% |

The retrieval system excludes all frozen golden-set interaction IDs.

Historical retrieval is primarily used as grounding evidence rather than
as a fully reliable semantic search system.

## 5. Escalation

The escalation layer is deterministic and independent of the LLM.

Escalation occurs for:
- account-specific security/access issues
- purchase, repair, warranty, refund, and replacement workflows
- potentially unsafe battery/hardware situations
- low-confidence classification
- insufficiently informative requests

The initial escalation evaluation produced:

- Accuracy: 54.50%
- Precision: 59.48%
- Recall: 75.83%
- F1: 66.67%

These results depend on the upstream intent prediction and therefore are
reported separately from final reply quality.

## 6. Full Agent Evaluation

The complete pipeline was evaluated on the frozen 200-example set.

Results:

- Intent accuracy: 10.00%
- Intent Macro F1: 5.03%
- Escalation accuracy: 54.50%
- Escalation precision: 59.48%
- Escalation recall: 75.83%
- Escalation F1: 66.67%
- Self-reported grounded-output rate: 97.00%
- Average latency: 10.16 seconds
- Runtime errors: 3

The 97% grounded figure is a system-generated indicator rather than an
independent factual validation metric.

## 7. Reply Quality and Judge Validation

A 30-example subset of generated replies was reviewed by a human and
independently scored by an LLM judge.

The judge/human agreement was weak:

- Exact agreement: 6.67%
- Within ±1 point: 10.00%
- Quadratic weighted kappa: -0.0584
- Spearman correlation: -0.1989
- Judge unsupported-claim rate: 0.00%
- Judge runtime errors: 0

Therefore the LLM judge is treated as a diagnostic tool and not as
ground truth for reply quality.

## 8. Failure Analysis

The most important observed failure modes were:

### A. Ambiguous intent boundaries

Examples frequently move between:
- IOS_UPDATE and symptom-specific intents
- FEATURE_HOW_TO and APPS_MEDIA
- CONNECTIVITY and APPS_MEDIA
- DEVICE_HARDWARE and symptom-based categories

### B. Noisy and underspecified Twitter messages

Short messages such as "fix this" or messages dominated by product names,
mentions, or emotional language provide insufficient evidence for
reliable classification.

### C. Multi-symptom conversations

A single tweet can contain battery, connectivity, performance, update,
and hardware symptoms simultaneously.

The system therefore uses explicit primary-intent precedence rules.

### D. Historical retrieval limitations

TF-IDF retrieval is effective for lexical overlap but misses semantic
similarity when users describe the same problem using different wording.

### E. Benchmark label ambiguity

Manual inspection showed several cases where the gold label appears
in tension with the primary support need expressed in the customer
message. These cases are reported as benchmark/annotation conflicts,
not automatically treated as model failures.

## 9. What Is Misleading About the Headline Number?

A single intent-accuracy number is incomplete.

The dataset is:
- small for a 10-class problem
- intentionally balanced across intents
- derived from noisy public Twitter conversations
- subject to ambiguous primary-intent boundaries
- evaluated against human labels that may themselves contain ambiguity

Therefore the benchmark should be interpreted together with:
- per-class metrics
- confusion patterns
- retrieval quality
- escalation metrics
- grounding checks
- human/LLM judge agreement

## 10. Production Safety

The system does not autonomously:
- modify accounts
- issue refunds
- make purchases
- perform private account actions
- guarantee outcomes

Sensitive and potentially unsafe cases are routed to human support.

The escalation policy is intentionally deterministic so safety handling is
not entirely dependent on LLM behavior.

## 11. One More Week

The highest-value improvements would be:

1. Replace lexical TF-IDF retrieval with semantic embeddings and reranking.
2. Increase the human evaluation set and refine ambiguous taxonomy
   boundaries.
3. Calibrate classifier confidence on a dedicated validation set.
4. Improve multi-intent primary-issue resolution.
5. Add stronger response-grounding verification before customer delivery.
6. Reduce end-to-end latency and LLM token usage with caching and
   selective generation.

## 12. Conclusion

The project demonstrates a complete support-agent architecture rather
than optimizing a single model metric.

The strongest engineering choices are:
- frozen holdout evaluation
- leakage-safe retrieval
- deterministic escalation
- explicit safety handling
- historical grounding
- measured baselines
- failure analysis
- explicit validation of the LLM judge itself

The current system is a working research prototype, with clear areas for
improvement before production deployment.