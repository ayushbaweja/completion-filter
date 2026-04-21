# completion-filter

Confidence estimation pipeline for LLM outputs. Estimates how confident a model is in its response using three complementary methods.

## Methods

| Method | How it works |
|---|---|
| **Logprob** | Computes `exp(mean token log-probability)` from a single pass with `logprobs=true` |
| **Semantic Entropy** | Samples 5 responses at temperature 0.8, embeds them, and measures agreement via mean pairwise cosine similarity |
| **Verbalized** | Asks the model to self-assess its confidence on a 0-100 scale |

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

# Single prompt, all methods
python demo.py "What is the capital of France?"

# Single method only
python demo.py --method logprob "What is 2+2?"
python demo.py --method semantic_entropy "When was the moon landing?"
python demo.py --method verbalized "What causes rain?"

# Custom aggregation strategy
python demo.py --aggregation weighted "Will it rain tomorrow?"
```

## Project Structure

- `shared/models.py` - Shared dataclasses (`ConfidenceResult`, `IntentResult`, `OutputRequest`, `FinalOutput`)
- `confidence/methods/` - Individual estimation methods (logprob, semantic entropy, verbalized)
- `confidence/estimator.py` - Orchestrator that runs methods concurrently and aggregates scores
