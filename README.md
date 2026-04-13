# AI Integrity Eval Lab

A fine-tuned binary toxicity classifier built for content-moderation research and portfolio demonstration.
The project fine-tunes **DistilBERT** on the **lmsys/toxic-chat** dataset, evaluates it with a comprehensive
metric suite, and exposes a lightweight **FastAPI** inference endpoint — all structured for reproducibility
and easy extension.

---

## Motivation

Automated content moderation is a critical layer in any large-scale platform's trust and safety stack.
Detecting toxic, harmful, or policy-violating content before it reaches users — or before it is injected
into an LLM's context — reduces real-world harm and lowers the cost of human review.

This project demonstrates:
- How to frame toxicity detection as a supervised binary classification task.
- How to fine-tune a pre-trained transformer efficiently with the Hugging Face `Trainer` API.
- How to evaluate beyond accuracy: precision, recall, F1, ROC-AUC, confusion matrices, and PR curves.
- How to productionise a model behind a typed REST API with structured error handling.

---

## Dataset

**[lmsys/toxic-chat](https://huggingface.co/datasets/lmsys/toxic-chat)** (config: `toxicchat0124`)

A dataset of real user prompts collected from the Vicuna online demo, annotated by human raters for toxicity.

| Field | Description |
|---|---|
| `user_prompt` | The raw text input from the user (our model input) |
| `toxicity` | Binary label: `1` = toxic, `0` = non-toxic |
| `jailbreaking` | Whether the prompt attempts a jailbreak (not used in training) |
| `model_output` | The model's response (not used in training) |

### Label Mapping

| Raw value | Mapped integer | String label |
|---|---|---|
| `0` / `False` | `0` | `non-toxic` |
| `1` / `True` | `1` | `toxic` |

The dataset is **imbalanced** (toxic examples are a minority). We track F1 and ROC-AUC as primary metrics
rather than accuracy, and optionally use weighted metrics. See the exploratory notebook for the exact ratio.

### Splits

The dataset ships with only a `train` split. We deterministically carve out validation (10%) and test (10%)
subsets so results are reproducible:

```
train ≈ 80%  |  validation ≈ 10%  |  test ≈ 10%
```

---

## Project Structure

```
ai-integrity-eval-lab/
├── configs/
│   └── base.yaml            # All hyperparameters and paths
├── notebooks/
│   └── exploratory_analysis.ipynb
├── outputs/                 # Generated at runtime (gitignored)
│   ├── checkpoints/         # Trainer checkpoints
│   ├── model/               # Final saved model + tokenizer
│   ├── metrics/             # JSON metrics, misclassified.csv
│   └── figures/             # ROC curve, PR curve, confusion matrix
├── scripts/
│   ├── run_train.sh
│   ├── run_eval.sh
│   └── run_api.sh
├── src/
│   ├── config.py            # Typed config dataclasses + YAML loader
│   ├── data.py              # Dataset loading and splitting
│   ├── preprocess.py        # Cleaning, label mapping, tokenization
│   ├── train.py             # Fine-tuning with HF Trainer
│   ├── evaluate_model.py    # Metrics, plots, misclassified examples
│   ├── predict.py           # CLI + programmatic predictor
│   ├── api.py               # FastAPI /health + /predict
│   └── utils.py             # Logging, seeding, directory helpers
├── tests/
│   ├── test_preprocess.py
│   └── test_predict.py
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## Setup

### Prerequisites

- Python 3.10+
- A GPU is recommended for training but not required (CPU works with smaller batch sizes)

### Install

```bash
# Clone the repo
git clone https://github.com/your-username/ai-integrity-eval-lab.git
cd ai-integrity-eval-lab

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install the package in editable mode (makes src importable)
pip install -e .
```

---

## Configuration

All tuneable parameters live in [`configs/base.yaml`](configs/base.yaml).
Edit that file or pass a different config path to any script.

Key parameters:

| Section | Key | Default | Description |
|---|---|---|---|
| `model` | `name` | `distilbert-base-uncased` | HF model identifier |
| `model` | `max_length` | `128` | Tokenizer truncation length |
| `training` | `num_train_epochs` | `3` | Training epochs |
| `training` | `learning_rate` | `2e-5` | AdamW learning rate |
| `training` | `per_device_train_batch_size` | `16` | Training batch size |
| `data` | `max_train_samples` | `null` | Cap training set size (null = all) |

---

## Training

```bash
# Default config
bash scripts/run_train.sh

# Custom config
bash scripts/run_train.sh configs/base.yaml
```

Or directly:

```bash
python -m src.train --config configs/base.yaml
```

Training artifacts are written to `outputs/`:
- `outputs/checkpoints/` — per-epoch Trainer checkpoints
- `outputs/model/` — final model and tokenizer
- `outputs/metrics/train_metrics.json` — final validation metrics

---

## Evaluation

```bash
# Evaluate on the test split (default)
bash scripts/run_eval.sh

# Evaluate on the validation split
bash scripts/run_eval.sh configs/base.yaml validation
```

Or directly:

```bash
python -m src.evaluate_model --config configs/base.yaml --split test
```

Outputs:
- `outputs/metrics/eval_test_metrics.json` — accuracy, precision, recall, F1, ROC-AUC
- `outputs/figures/confusion_matrix_test.png`
- `outputs/figures/roc_curve_test.png`
- `outputs/figures/pr_curve_test.png`
- `outputs/metrics/misclassified.csv` — top false positives and false negatives

---

## CLI Prediction

```bash
# Single text
python -m src.predict "You are an absolute idiot."

# Custom model dir
python -m src.predict --model-dir outputs/model "Hello, how are you?"

# Interactive REPL (no text argument)
python -m src.predict
```

Example output:

```
Text    : You are an absolute idiot.
Label   : toxic
Confidence: 0.9821
Toxic prob: 0.9821
```

---

## API

### Start the server

```bash
bash scripts/run_api.sh          # port 8000 (default)
bash scripts/run_api.sh 9000     # custom port
```

Interactive docs: [http://localhost:8000/docs](http://localhost:8000/docs)

### Endpoints

#### `GET /health`

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "ok",
  "model_loaded": true,
  "uptime_seconds": 12.5
}
```

#### `POST /predict`

```bash
curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"text": "You are going to regret this."}'
```

```json
{
  "label": "toxic",
  "label_id": 1,
  "confidence": 0.9734,
  "toxic_prob": 0.9734
}
```

```bash
curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"text": "Could you help me write a cover letter?"}'
```

```json
{
  "label": "non-toxic",
  "label_id": 0,
  "confidence": 0.9912,
  "toxic_prob": 0.0088
}
```

---

## Running Tests

```bash
pytest
```

The test suite covers:
- Label mapping correctness (integers, strings, booleans)
- Null/empty row cleaning
- Tokenization column layout and padding
- Predictor loading, output schema, edge cases (empty text, long text)
- Batch prediction

Tests use a tiny randomly-initialized DistilBERT model and do **not** require a trained checkpoint
or internet access after the tokenizer is cached.

---

## Limitations and Bias Considerations

1. **Dataset scope** — lmsys/toxic-chat is derived from prompts sent to Vicuna (an open-source chat model).
   The distribution may not generalise to other platforms, languages, or user demographics.

2. **Class imbalance** — Toxic examples are a minority. The model is optimised for F1, but at a fixed
   decision threshold of 0.5 it may still exhibit higher false-negative rates on rare toxic sub-types
   (e.g. subtle hate speech, coded language).

3. **English-only** — The model and dataset are English-only. Do not apply this classifier to multilingual
   content without additional fine-tuning or translation.

4. **Annotation disagreement** — Binary toxicity labels collapse a continuous, context-dependent judgment
   into a single bit. Borderline cases annotated differently by raters introduce irreducible label noise.

5. **No subgroup analysis** — Performance may vary significantly across demographic subgroups, topics,
   and writing styles. Fairness-aware evaluation (e.g. disaggregated metrics by topic or identity term)
   is an important next step before any production deployment.

6. **Threshold sensitivity** — All metrics reported at threshold 0.5. Operational use typically requires
   threshold tuning to balance precision and recall for the specific moderation policy.

---

## Future Improvements

- **Compare against `google/civil_comments`** — a larger, more diverse toxicity dataset from comment
  sections; useful for domain transfer benchmarking.
- **Threshold tuning** — sweep decision thresholds and choose the operating point that best matches
  the target precision/recall trade-off for the deployment context.
- **PEFT / LoRA experiment** — apply parameter-efficient fine-tuning (via `peft`) to reduce GPU memory
  requirements and enable fine-tuning of larger models (e.g. `roberta-large`, `deberta-v3-base`).
- **Calibration analysis** — measure expected calibration error (ECE) and apply temperature scaling
  so that `toxic_prob` is a reliable probability estimate, not just a relative score.
- **Subgroup and intersectional analysis** — disaggregate metrics by topic, demographic mention,
  and jailbreak type to identify fairness gaps.
- **Ensemble / multi-label extension** — extend to multi-label classification (hate speech, harassment,
  self-harm, etc.) to better reflect real-world content policy taxonomies.

---

## License

MIT
