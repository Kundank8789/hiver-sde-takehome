# Decision Log

## Decision 1 — Target brand

**Decision:** AppleSupport

**Why:**
- AppleSupport provided a large volume of recurring technical-support interactions.
- The interactions contain useful troubleshooting and escalation patterns.
- The domain provides diverse but coherent support intents.
- The dataset is suitable for retrieval, classification, escalation, and response-generation experiments.

**Alternatives considered:**
- AmazonHelp
- Uber_Support

**Why AppleSupport was selected:**
AppleSupport provides a strong balance between interaction volume, recurring technical issues, troubleshooting responses, and opportunities to evaluate both automated handling and escalation.

---

## Decision 2 — Define the evaluation unit as a support interaction

**Decision:** Do not treat the entire Twitter reply tree as one conversation.

**Why:**
The raw Twitter reply graph can connect unrelated customers and support issues under the same root tweet.

**New definition:**
A support interaction is an AppleSupport response paired with its direct parent customer message, optionally augmented with preceding parent-chain context.

**Benefit:**
This prevents unrelated messages from being treated as conversation context and provides a clean unit for classification, retrieval, reply generation, and evaluation.

---

## Decision 3 — Use a 10-intent taxonomy

**Decision:** Use 10 broad support intents.

**Why:**
The sampled interactions showed recurring clusters around:
- software updates
- battery/power
- performance/stability
- connectivity
- apps/media
- device hardware
- feature/how-to
- account/security
- purchase/repair/warranty
- unclear requests

**Why not more classes:**
Individual products and individual bugs would create an overly fragmented taxonomy and make reliable evaluation harder.

**Why include OTHER_UNCLEAR:**
Some messages are too vague or incomplete to classify reliably. The system should be able to represent uncertainty rather than force a misleading label.

---

## Decision 4 — Use a balanced 200-example golden set

**Decision:** Build a 200-example human-labelled evaluation set with 20 examples per intent.

**Why:**
A balanced set provides coverage of all 10 intents and prevents high-volume classes from dominating the evaluation.

**Caveat:**
This distribution is intentionally constructed for coverage and is not representative of production traffic frequency.

---

## Decision 5 — Freeze the golden evaluation set

**Decision:** Freeze `evaluation/golden_set.csv` before final evaluation.

**Why:**
The benchmark should not change as models are developed.

**Evaluation constraint:**
The frozen set is excluded from training and few-shot examples.

**Benefit:**
This prevents benchmark contamination and makes comparisons between experiments meaningful.

---

## Decision 6 — Prevent golden-set leakage from retrieval

**Decision:** Exclude all golden-set interaction IDs from the retrieval index.

**Why:**
If the same customer message appears in retrieval during evaluation, the model could retrieve the answer or near-duplicate associated with the evaluation example.

**Benefit:**
Retrieval results represent historical evidence outside the evaluation benchmark.

---

## Decision 7 — Build weakly-labelled training data

**Decision:** Use high-precision heuristic rules to create weak training labels from historical support interactions.

**Why:**
Manual labelling of the entire 100K+ interaction dataset is impractical.

**Trade-off:**
Weak labels provide scale but introduce noise, so they are used for training/supporting experiments rather than as ground truth.

---

## Decision 8 — Use TF-IDF as the first retrieval baseline

**Decision:** Start with TF-IDF lexical retrieval.

**Why:**
TF-IDF is transparent, deterministic, inexpensive, and provides a reproducible retrieval baseline.

**Limitation:**
It relies strongly on lexical overlap and can miss semantically similar messages with different wording.

---

## Decision 9 — Treat lexical retrieval as a baseline, not the final retriever

**Decision:** Do not rely on TF-IDF retrieval alone for production-quality semantic retrieval.

**Evidence:**
Lexical Recall@5 was 0.145, while intent-aware reranking reached 0.150.

**Conclusion:**
The retrieval layer is useful as historical evidence but is not sufficiently reliable to determine relevance from lexical similarity alone.

**Next step:**
Use semantic retrieval and reranking while retaining lexical retrieval as a transparent baseline.

---

## Decision 10 — Separate classification from escalation

**Decision:** Do not allow the LLM to make the final escalation decision by itself.

**Why:**
Classification and escalation are different decisions. A message can be correctly classified but still require human handling.

**Implementation:**
A deterministic escalation policy handles:
- account-specific security cases
- purchase/repair/warranty workflows
- low-confidence cases
- insufficient information
- potentially unsafe battery/hardware conditions

**Benefit:**
Safety-sensitive behavior remains predictable and auditable.

---

## Decision 11 — Use historical responses as grounding evidence

**Decision:** Ground response generation in retrieved AppleSupport interactions.

**Why:**
The objective is not merely to generate plausible support text. Replies should reflect patterns found in historical support responses.

**Constraint:**
The model is instructed not to invent troubleshooting instructions, policies, guarantees, refunds, or unsupported actions.

---

## Decision 12 — Do not expose raw Twitter artifacts

**Decision:** Clean retrieved historical responses before using them as customer-facing evidence.

**Why:**
Historical responses contain Twitter usernames, URLs, DM language, formatting noise, and platform-specific artifacts.

**Benefit:**
This prevents internal/public-platform artifacts from leaking into customer-facing replies.

---

## Decision 13 — Audit the golden set before optimizing the model

**Decision:** Treat high-confidence model-vs-gold disagreements as possible benchmark/annotation issues rather than automatically treating them as model failures.

**Why:**
Several inspected examples contained ambiguous or conflicting primary-intent interpretations.

**Benefit:**
This avoids optimizing the model against potentially noisy labels and makes failure analysis more trustworthy.

**Important limitation:**
The original frozen golden set remains the official benchmark; the audit is treated as a diagnostic exercise.

---

## Decision 14 — Validate the LLM judge against human ratings

**Decision:** Do not assume LLM-as-judge scores are ground truth.

**Why:**
Automated judges can disagree with human reviewers.

**Method:**
A 30-example subset was independently reviewed by a human and scored by the LLM judge.

**Result:**
Judge-human agreement was weak, so judge scores are treated as diagnostic evidence rather than ground truth.

---

## Decision 15 — Preserve transparent baselines and negative results

**Decision:** Keep the trivial baseline, TF-IDF baseline, initial LLM result, hybrid result, and failure analysis rather than reporting only the best result.

**Why:**
The purpose of the take-home is to demonstrate engineering judgment and empirical reasoning, not benchmark optimization through selective reporting.

**Benefit:**
The final report shows what worked, what failed, and why.