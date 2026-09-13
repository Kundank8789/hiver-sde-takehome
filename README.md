# AppleSupport AI Support Agent

An AI-assisted customer-support agent built from historical
AppleSupport conversations.

## Architecture

```text
Customer message
      |
      v
Hybrid intent classifier
      |
      v
Deterministic escalation policy
      |
      v
Historical retrieval
      |
      v
Grounded response generation
      |
      v
Structured support response

Dataset
Raw tweets: ~2.8M
AppleSupport interactions: 106,648
Frozen golden evaluation set: 200
Intents: 10
Setup
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt

Create .env:

GROQ_API_KEY=your_key_here
Reproduce
Build support interactions
python src/build_support_interactions.py
Build golden candidates
python src/create_golden_candidates.py
Build retrieval index
python src/build_retrieval_index.py
Evaluate retrieval
python src/evaluate_retrieval.py
Evaluate hybrid classifier
python src/evaluate_hybrid_classifier.py
Offline pipeline test
python src/test_offline_agent.py
python src/test_pipeline.py
Full agent evaluation

Requires a configured Groq API key:

python src/evaluate_full_agent.py
Results

See:

reports/final_report.md
reports/failure_analysis.md
reports/model_comparison.md
Design Principles
Frozen evaluation set is never used as training data.
Retrieval excludes frozen golden-set examples.
Safety escalation is deterministic.
Account and case-specific workflows are escalated.
LLM judge results are treated as diagnostic evidence, not ground truth.

### 3. Then check your repository

Run:

```powershell
git status