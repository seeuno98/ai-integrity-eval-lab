# AI Integrity Eval Lab

A fine-tuned binary toxicity classifier built for content-moderation research and portfolio demonstration.
The project fine-tunes **DistilBERT** on the **lmsys/toxic-chat** dataset, evaluates it with a comprehensive
metric suite (ROC-AUC, PR curve, threshold sweep, error slices), runs a curated manual probe set to surface
brittleness, and exposes a lightweight **FastAPI** inference endpoint.

---

## Motivation

Automated content moderation is a critical layer in any large-scale platform's trust and safety stack.
Detecting toxic, harmful, or policy-violating content before it reaches users — or before it poisons an
LLM's context window — reduces real-world harm and lowers the cost of human review.

This project demonstrates:
- End-to-end fine-tuning of a transformer for a safety-relevant classification task.
- Responsible evaluation: F1, ROC-AUC, PR curves, confusion matrices, and threshold sweeps rather than
  accuracy alone.
- Systematic error analysis: error slices by text length, punctuation, and Unicode content.
- Manual probing to discover real brittleness that automated metrics miss.
- A typed, modular codebase structured for extension (PEFT, multi-label, calibration).

---

## Dataset

**[lmsys/toxic-chat](https://huggingface.co/datasets/lmsys/toxic-chat)** (config: `toxicchat0124`)

Real user prompts collected from the Vicuna online demo, annotated by human raters for toxicity.

| Field | Description |
|---|---|
| `user_input` | The raw text input from the user (model input) |
| `toxicity` | Binary label: `1` = toxic, `0` = non-toxic |
| `jailbreaking` | Whether the prompt attempts a jailbreak (not used in training) |

### Label Mapping

| Raw value | Mapped integer | String label |
|---|---|---|
| `0` / `False` | `0` | `non-toxic` |
| `1` / `True` | `1` | `toxic` |

### Class Imbalance

The dataset is **heavily imbalanced** — toxic examples are a small minority (roughly 7–10% of the corpus).
This is intentional: it reflects real-world moderation ratios. Consequences:
- Accuracy is a misleading metric; F1 and ROC-AUC are primary.
- A naive majority-class classifier achieves ~92% accuracy while missing all toxic content.
- Threshold tuning is essential: the default 0.5 cut-off may not match your recall/precision target.

### Splits

The dataset ships with only a `train` split. We deterministically create 80/10/10 train/val/test subsets:

```
train ≈ 80%  |  validation ≈ 10%  |  test ≈ 10%
```

---

## Model

**Base**: `distilbert-base-uncased` via `AutoModelForSequenceClassification`

DistilBERT is a distilled version of BERT that retains ~97% of its performance at 40% fewer parameters
and 60% faster inference. It is a good baseline for content-safety classification because:
- It handles free-form text robustly.
- It is small enough to fine-tune on a single GPU or CPU.
- It establishes a clear baseline for future comparisons (RoBERTa, DeBERTa, LoRA-tuned LLMs).

---

## Project Structure

```
ai-integrity-eval-lab/
├── configs/
│   └── base.yaml                # All hyperparameters and paths
├── data/
│   └── probes/
│       └── manual_probe_set.csv # Hand-curated brittleness probe set
├── notebooks/
│   └── exploratory_analysis.ipynb
├── outputs/                     # Generated at runtime (gitignored)
│   ├── checkpoints/
│   ├── model/                   # Saved model + tokenizer
│   ├── metrics/                 # JSON metrics, CSVs
│   ├── figures/                 # Plots
│   └── probes/                  # Probe results and report
├── scripts/
│   ├── run_train.sh
│   ├── run_eval.sh
│   ├── run_probes.sh
│   └── run_api.sh
├── src/
│   ├── config.py                # Typed dataclasses + YAML loader
│   ├── data.py                  # Dataset loading and splitting
│   ├── preprocess.py            # Normalization, cleaning, tokenization
│   ├── train.py                 # Fine-tuning with HF Trainer
│   ├── evaluate_model.py        # Metrics, threshold tuning, error slices
│   ├── run_probes.py            # Manual probe runner
│   ├── predict.py               # CLI + programmatic predictor
│   ├── api.py                   # FastAPI /health + /predict
│   └── utils.py                 # Logging, seeding, directories
├── tests/
│   ├── test_preprocess.py       # Normalization, cleaning, label mapping, tokenization
│   └── test_predict.py          # Predictor, threshold, normalization, probes
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## Setup

### Prerequisites

- Python 3.10+
- A GPU is recommended for training but CPU works with smaller batch sizes.

### Install

```bash
git clone https://github.com/your-username/ai-integrity-eval-lab.git
cd ai-integrity-eval-lab

python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate

pip install -r requirements.txt
pip install -e .                 # makes src importable
```

---

## Configuration

All tuneable parameters live in [`configs/base.yaml`](configs/base.yaml).

Key parameters:

| Section | Key | Default | Description |
|---|---|---|---|
| `model` | `name` | `distilbert-base-uncased` | HF model identifier |
| `model` | `max_length` | `128` | Tokenizer truncation length |
| `training` | `num_train_epochs` | `3` | Training epochs |
| `training` | `learning_rate` | `2e-5` | AdamW learning rate |
| `data` | `normalize_text` | `true` | Apply Unicode/whitespace normalization |
| `inference` | `default_threshold` | `0.5` | Decision threshold at inference time |
| `inference` | `threshold_grid` | `[0.1..0.9]` | Thresholds evaluated during analysis |
| `probes` | `probe_file` | `data/probes/manual_probe_set.csv` | Hand-curated probe examples |

---

## Training Pipeline

```bash
bash scripts/run_train.sh

# or directly:
python -m src.train --config configs/base.yaml
```

Steps:
1. Load `lmsys/toxic-chat` and split into 80/10/10.
2. Apply text normalization (Unicode NFKC + whitespace collapse).
3. Tokenize with DistilBERT tokenizer, truncate to `max_length=128`.
4. Fine-tune with the HF `Trainer` (AdamW, linear warmup, best-model checkpoint by F1).
5. Save model + tokenizer to `outputs/model/`.

Artifacts:
- `outputs/checkpoints/` — per-epoch checkpoints
- `outputs/model/` — final model
- `outputs/metrics/train_metrics.json` — final validation metrics

---

## Evaluation

```bash
bash scripts/run_eval.sh

# or directly:
python -m src.evaluate_model --config configs/base.yaml --split test
```

Outputs:
- `outputs/metrics/eval_test_metrics.json` — accuracy, precision, recall, F1, ROC-AUC
- `outputs/figures/confusion_matrix_test.png`
- `outputs/figures/roc_curve_test.png`
- `outputs/figures/pr_curve_test.png`
- `outputs/metrics/misclassified.csv` — top FP and FN by confidence

---

## Threshold Tuning

Threshold tuning is automatically run at the end of evaluation. No separate command is needed.

The threshold analysis sweeps `[0.1, 0.2, ..., 0.9]` and identifies:
- The threshold with highest F1
- A high-recall operating point (recall ≥ 0.85)
- A high-precision operating point (precision ≥ 0.90)

Outputs:
- `outputs/metrics/threshold_metrics_test.csv` — per-threshold precision, recall, F1, accuracy, TP/FP/TN/FN
- `outputs/metrics/threshold_summary_test.json` — best operating points
- `outputs/figures/threshold_tradeoff_test.png` — precision/recall/F1 vs threshold plot

**Why threshold tuning matters**: ROC-AUC measures ranking quality independent of threshold.
A strong AUC (e.g. 0.90+) means the model ranks toxic content above non-toxic content reliably —
but the *default* 0.5 threshold may not be the right operating point for your policy.
A moderation system that must flag nearly all toxic content should operate at a lower threshold (higher
recall, more false positives). A system where false positives carry high user cost should use a higher
threshold (higher precision, some toxic content slips through).

---

## Manual Probe Analysis

The probe set (`data/probes/manual_probe_set.csv`) contains 36 hand-curated examples across 7 categories
designed to expose brittleness that held-out test metrics miss:

| Category | Purpose |
|---|---|
| `punctuation_sensitivity` | Same insult ± trailing punctuation — should not flip the label |
| `negation` | Negated insults should be non-toxic |
| `short_insult` | Single-word slurs may fall below the model's confidence |
| `explicit_targeted_insult` | Direct personal attacks and threats — should be flagged reliably |
| `borderline_insult` | Criticism of work/behavior vs. personal attacks |
| `multilingual_slang` | Non-English profanity and censored terms |
| `harmless_control` | Clearly benign text — should never be flagged |

```bash
bash scripts/run_probes.sh

# or directly:
python -m src.run_probes --config configs/base.yaml
```

Outputs:
- `outputs/probes/probe_results.csv` — per-example predictions
- `outputs/probes/probe_summary.json` — accuracy/precision/recall/F1 per category
- `outputs/probes/probe_report.md` — human-readable report with FP/FN examples

### Observed Brittleness

Manual probing before fine-tuning (and/or with low-resource checkpoints) typically reveals:

- **Punctuation sensitivity** — adding or removing trailing `!` or `.` occasionally flips the prediction,
  suggesting the model relies partly on surface-level punctuation patterns rather than semantic meaning.
- **Negation handling** — `"you are not stupid"` can be misclassified as toxic because the model detects
  the word *stupid* without reliably resolving the negation.
- **Short insults** — single-word slurs like `idiot` or `loser` may score below the default 0.5 threshold
  because they lack the context that typically surrounds toxic content in training data.
- **Multilingual/slang coverage** — non-English profanity (e.g. Korean `sibal`, French `merde`) and
  censored forms (e.g. `b****`) are under-represented in the English-only training corpus and may be missed.

These failure modes motivate the future improvements listed below.

---

## Error Analysis

At the end of evaluation, error slices are computed and saved to
`outputs/metrics/error_slices_test.json`. Slices include:

| Slice | What it measures |
|---|---|
| `short_text_lt50chars` | Accuracy on texts with < 50 characters |
| `medium_text_50_200chars` | Accuracy on texts 50–200 characters |
| `long_text_gt200chars` | Accuracy on texts > 200 characters |
| `contains_exclamation_or_question` | Texts with `!` or `?` |
| `contains_non_ascii` | Texts with non-ASCII characters |

These slices are diagnostic — they do not control for confounders, but they give a quick signal about
where the model performs differently and deserve closer inspection.

---

## CLI Prediction

```bash
# One-shot with --text flag
python -m src.predict --text "You are an absolute idiot."

# With custom threshold
python -m src.predict --text "You are an absolute idiot." --threshold 0.3

# Positional argument (backwards compatible)
python -m src.predict "Could you help me write a cover letter?"

# Interactive REPL
python -m src.predict

# Disable normalization
python -m src.predict --text "You idiot" --no-normalize
```

Example output:

```
Text      : You are an absolute idiot.
Label     : toxic
Toxic prob: 0.9821
Confidence: 0.9821
Threshold : 0.50
```

---

## API

```bash
bash scripts/run_api.sh          # port 8000
bash scripts/run_api.sh 9000     # custom port
```

Interactive docs: [http://localhost:8000/docs](http://localhost:8000/docs)

### `GET /health`

```bash
curl http://localhost:8000/health
```
```json
{"status": "ok", "model_loaded": true, "uptime_seconds": 12.5}
```

### `POST /predict`

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
- Text normalization: Unicode NFKC, whitespace collapse, punctuation preservation
- Label mapping: integers, strings, booleans
- Null/empty row cleaning with schema validation
- Tokenization column layout and padding
- Threshold metric computation
- Predictor loading, output schema, threshold logic, normalization consistency
- Probe file loading and column validation

Tests use a tiny randomly-initialized model and do **not** require a trained checkpoint or internet access
after the tokenizer is cached.

---

## Key Findings

| Finding | Detail |
|---|---|
| Strong ranking performance | ROC-AUC typically ≥ 0.90 after fine-tuning, indicating the model reliably ranks toxic content above non-toxic |
| Reasonable held-out F1 | F1 on the test split is competitive given the class imbalance |
| Threshold matters | The default 0.5 cut-off is rarely optimal; the threshold tradeoff plot shows clear operating-point choices |
| Punctuation brittleness | Identical insults with and without trailing punctuation occasionally receive different predictions |
| Negation failures | The model sometimes misclassifies negated insults as toxic |
| Multilingual gaps | Non-English profanity is largely missed due to training corpus composition |

These findings are typical of a first-pass English-only transformer classifier on an imbalanced dataset
and motivate the future improvements below.

---

## Limitations and Bias Considerations

1. **Dataset scope** — lmsys/toxic-chat is derived from prompts sent to Vicuna. The distribution may not
   generalise to other platforms, languages, or user demographics.

2. **Class imbalance** — Toxic examples are a minority. At a fixed 0.5 threshold the model will exhibit
   higher false-negative rates on rare sub-types (subtle hate, coded language, sarcasm).

3. **English-only** — The model and training data are English-centric. Non-English toxic content is largely
   missed. Do not deploy this classifier on multilingual content without additional fine-tuning.

4. **Annotation disagreement** — Binary toxicity labels collapse a context-dependent judgment into a single
   bit. Borderline cases annotated differently by raters introduce irreducible label noise.

5. **No subgroup analysis** — Performance may vary significantly across demographic subgroups, topics, and
   writing styles. Disaggregated evaluation is an important step before any production deployment.

6. **Threshold sensitivity** — All aggregated metrics reported at the default threshold. Operational use
   requires threshold selection aligned with the specific moderation policy's recall/precision target.

7. **No calibration analysis** — `toxic_prob` is a relative score, not a calibrated probability. Do not
   interpret it as a literal percentage without temperature scaling or isotonic regression.

---

## Future Improvements

- **Compare against `google/civil_comments`** — a larger, more diverse toxicity dataset from comment
  sections; useful for domain transfer benchmarking.
- **Threshold tuning integration** — expose the threshold as an API parameter so callers can choose their
  operating point at inference time.
- **PEFT / LoRA experiment** — apply parameter-efficient fine-tuning (via `peft`) to reduce GPU memory
  and enable fine-tuning of larger base models (e.g. `roberta-large`, `deberta-v3-base`).
- **Calibration analysis** — measure expected calibration error (ECE) and apply temperature scaling so that
  `toxic_prob` is a reliable probability estimate.
- **Subgroup and intersectional analysis** — disaggregate metrics by topic, demographic mention, and
  jailbreak type to identify fairness gaps.
- **Multi-label extension** — extend to multi-label classification (hate speech, harassment, self-harm, etc.)
  to reflect real-world content policy taxonomies more faithfully.
- **Negation and linguistic feature analysis** — audit negation, sarcasm, and indirect threats as structured
  probe categories; use these as a test suite for model improvements.

---

## License

MIT
