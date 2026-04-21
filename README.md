# completion-filter

A confidence estimation and safety routing pipeline for LLM outputs. Estimates how confident a model is in its response using three complementary methods, and pre-filters harmful queries before they reach the main model.

## Methods

| Method | How it works |
|---|---|
| **Logprob** | Computes `exp(mean token log-probability)` from a single pass with `logprobs=true` |
| **Semantic Entropy** | Samples 5 responses at temperature 0.8, embeds them, and measures agreement via mean pairwise cosine similarity |
| **Verbalized** | Asks the model to self-assess its confidence on a 0–100 scale |

A **harm classification pre-filter** runs first to route queries between ethical refusal and normal confidence estimation.

## Setup

```bash
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -r requirements.txt
export OPENAI_API_KEY=your-key
```

## Usage

```bash
# Run demo with default test prompts
python demo.py

# Single prompt through full pipeline (harm filter + all methods)
python demo.py "What is the capital of France?"

# Single method only
python demo.py --method logprob "What is 2+2?"
python demo.py --method semantic_entropy "When was the moon landing?"
python demo.py --method verbalized "What causes rain?"

# All methods with custom aggregation
python demo.py --all --aggregation weighted "Will it rain tomorrow?"
```

## Project Structure

- `shared/models.py` — Shared dataclasses (`ConfidenceResult`, `IntentResult`, `OutputRequest`, `FinalOutput`)
- `confidence/methods/` — Individual estimation methods (logprob, semantic entropy, verbalized)
- `confidence/harm_filter.py` — Prompt-based harm classifier
- `confidence/estimator.py` — Orchestrator that runs methods concurrently and aggregates scores
