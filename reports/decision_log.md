# Decision Log

## Decision 1 — Target brand

**Decision:** AppleSupport

**Why:**
- 106,623 customer messages were directly replied to by AppleSupport.
- The dataset contains substantial recurring technical-support issues.
- Conversations contain useful troubleshooting and escalation patterns.
- The domain provides diverse but coherent support intents.
- The brand provides enough historical interactions for retrieval and evaluation.

**Alternatives considered:**
- AmazonHelp
- Uber_Support

**Why AppleSupport was selected:**
AppleSupport provides a strong balance between interaction volume, recurring technical issues, troubleshooting responses, and opportunities to evaluate both automated handling and escalation.

## Decision 2 — Define the evaluation unit as a support interaction

**Decision:** Do not treat the entire Twitter reply tree as one conversation.

**Why:**
The raw Twitter reply graph can connect many unrelated customers and support issues under the same root tweet. Large connected components contained unrelated topics and multiple customers.

**New definition:**
A support interaction is an AppleSupport response paired with its direct parent customer message, optionally augmented with the preceding parent-chain context.

**Benefit:**
This prevents unrelated messages from being treated as conversation context and gives us a clean unit for intent classification, retrieval, reply generation, and evaluation.

## Decision 3 — Use a 10-intent taxonomy

**Decision:** Use 10 broad support intents for AppleSupport.

**Why:**
The sampled interactions showed recurring clusters around software updates, battery/power, performance, connectivity, apps/media, hardware, feature/how-to questions, account/security, purchases/repairs/warranty, and ambiguous requests.

**Why not more classes:**
Individual products and individual bugs would create an overly fragmented taxonomy and make reliable evaluation harder.

**Why include OTHER_UNCLEAR:**
Some customer messages are too vague or contain insufficient information. The agent should be able to explicitly acknowledge uncertainty instead of forcing an incorrect intent.

## Decision 5 — Freeze the golden evaluation set

**Decision:** Freeze 200 human-labelled examples as `evaluation/golden_set.csv`.

**Why:**
The set was independently labelled after blind candidate sampling and passed structural validation with no missing annotations, invalid labels, or duplicate interaction IDs.

**Composition:**
- 200 examples
- 10 intent classes
- 20 examples per intent
- 120 marked for escalation
- 80 marked for non-escalation

**Evaluation constraint:**
The frozen set will not be used for model training or prompt-example retrieval during evaluation.

**Caveat:**
The class distribution was intentionally constructed for coverage and is therefore not representative of production traffic frequency.