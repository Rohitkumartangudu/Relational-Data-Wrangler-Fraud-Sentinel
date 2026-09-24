# Relational Data Wrangler & Fraud Sentinel

An end-to-end fraud-risk pipeline for messy relational banking data. The project combines data cleaning, relational joins, adversarial prompt-injection sanitization, weak supervision, fine-tuning of a small language model, hybrid risk classification, grounded explanations, and strict JSON validation.

## Problem

The hackathon provides three deliberately messy datasets:

- `transactions.csv`
- `accounts.csv`
- `customers.csv`

The pipeline must clean and merge the data, neutralize prompt-injection attempts in untrusted transaction text, use an open-weight language model below 3B parameters, fine-tune the small model, and produce one strict JSON risk profile per transaction.

## Data Disclosure

All data in this repository (`transactions.csv`, `accounts.csv`, `customers.csv`) is synthetic, generated for this hackathon. No real customer, account, or transaction data is used or represented.

## Final Pipeline

```text
transactions.csv
accounts.csv
customers.csv
        |
        v
Data cleaning & normalization
        |
        v
Relational merge + integrity flags
        |
        v
Fraud feature engineering
        |
        v
Prompt-injection sanitization
        |
        v
Weak supervision / SFT dataset creation
        |
        v
QLoRA fine-tuning of Qwen2.5-1.5B-Instruct
        |
        v
Hybrid inference
  |                 |
  | strong/low     | ambiguous
  v                 v
Rules              Fine-tuned SLM
  |                 |
  +--------+--------+
           |
           v
Evidence-grounded justification
           |
           v
Strict schema validation
           |
           v
fraud_predictions.json
```

## Model

The notebook uses:

`Qwen/Qwen2.5-1.5B-Instruct`

The model is fine-tuned with QLoRA/LoRA using the cleaned transaction/account/customer features.

The final inference pipeline uses the fine-tuned SLM as a secondary classifier for ambiguous transactions. Strong and low-evidence cases are handled by deterministic behavioral risk rules. This prevents unsupported model reasoning from controlling the final output.

## Results

Evaluated on a 40-example held-out split (20 fraud / 20 non-fraud, stratified, never seen during fine-tuning). See "Evaluation Methodology" below for what
this measures and does not measure.

### Initial attempt (full-sequence loss)

| Model             | Valid JSON | Agreement with weak label | Behavior                        |
|--------------------|-----------|----------------------------|----------------------------------|
| Base (no adapter)  | 38/40     | 19/38 (50.0%)               | Predicted `is_fraud: true` for every valid output |
| Fine-tuned (LoRA)  | 36/40     | 18/36 (50.0%)               | Predicted `is_fraud: true` for every valid output |

Both models collapsed to always predicting fraud, regardless of the input features — the 50% agreement is exactly what a coin flip would score against
a balanced set, not evidence of reasoning. Confirmed by inspecting raw generations directly (justifications cited features inconsistently, e.g.
citing *low* credit utilization, normally a low-risk signal, as evidence of fraud) and by checking the fine-tuned model against its own **training**
data, where it also scored 50% (9/18) — meaning it had not learned to discriminate the task at all.

**Root cause:** the SFT training cell built each example as a single flat chat-template string and trained with the default full-sequence loss, which
scores every token — the long, largely-boilerplate prompt included. The assistant turn (the part that actually encodes the label) was a small
fraction of total tokens, so its gradient contribution was diluted by the much longer, easier-to-predict prompt text.

### Fix: assistant-only loss + adjusted learning rate/epochs

Two changes were required together — neither alone was sufficient:

1. `assistant_only_loss=True` (TRL 1.13's built-in completion-only masking, applied to a `messages`-formatted dataset) — restricts the loss to the
   assistant turn's tokens only.
2. `learning_rate` raised from `2e-5` to `2e-4`, and `num_train_epochs` raised from `2` to `5` — masking alone only moved held-out agreement to
   51.3%, still degenerate; the higher learning rate and additional epochs were needed to actually shift the adapter weights enough to learn the
   discrimination.

| Model                        | Valid JSON | Agreement with weak label | Confusion matrix (TP/FN/FP/TN) |
|-------------------------------|-----------|-----------------------------|----------------------------------|
| Base (no adapter)             | 38/40     | 19/38 (50.0%)                | 19/0/19/0 — always predicts fraud |
| Fine-tuned (LoRA, fixed)      | 40/40     | 34/40 (85.0%)                | 17/3/3/17 — balanced errors both directions |
| Fine-tuned, on own training data | 20/20 | 19/20 (95.0%)                | 9/0/1/10 |

The fine-tuned model now shows genuine discrimination (errors on both sides of the confusion matrix, not a one-sided collapse), with a modest and
expected generalization gap between training (95%) and held-out (85%) performance — consistent with a small model fine-tuned on 160 examples,
not memorization (which would show near-100% on training and near-chance on held-out).

### Evaluation Methodology

This evaluates agreement between each model's `is_fraud` output and the weak-label rule, on examples held out from fine-tuning but drawn from the
same confident-bucket distribution (score ≥4 or ≤1) used to build the training set. It does not measure performance on the ambiguous middle band
(score 1.0–4.0) that the model actually classifies in production, since no ground truth exists there to check against. The 85% figure should be read
as "the model learned to reproduce the rule on unseen confident-bucket examples," not as a measure of real-world fraud-detection accuracy, and it
does not by itself confirm how the model performs on the harder, ambiguous cases it is actually deployed against.

## Data Processing

The preprocessing pipeline:

1. Loads the three CSV files.
2. Normalizes string and categorical fields.
3. Converts transaction amounts and other numeric columns safely.
4. Parses transaction timestamps with mixed-format handling.
5. Removes exact duplicates and duplicate transaction IDs.
6. Merges transactions with accounts and customers.
7. Tracks missing account/customer relationships.
8. Builds behavioral fraud-risk features.
9. Sanitizes untrusted text against common prompt-injection patterns.

## Fraud-Risk Features

The pipeline considers signals including:

- new device usage
- foreign transactions
- distance from home
- transaction amount
- amount-to-account-average ratio
- transaction velocity
- failed/reversed transactions
- negative post-transaction balance
- account/customer relationship integrity

The supplied files do not contain a verified `is_fraud` ground-truth label. Therefore, the training labels are weak/pseudo-labels derived from observable behavioral risk signals. The resulting `is_fraud` values should be interpreted as risk classifications, not measured ground-truth fraud predictions.

## Adversarial Prompt Injection

Transaction text is treated as untrusted data. Known prompt-injection patterns such as attempts to ignore previous instructions, override system/developer messages, or force a safe classification are redacted before model use.

## Output

The final output is:

`fraud_predictions.json`

Each record contains exactly:

```json
{
  "transaction_id": "TXN_00001",
  "is_fraud": true,
  "confidence": 0.92,
  "justification": "One-sentence plain-text justification based on transaction behavior."
}
```

The final generated artifact contains 977 predictions with unique transaction IDs and schema-valid records.

## Repository Structure

```text
.
├── relational_data_wrangler_fraud_sentinel.ipynb
├── Data/
│   ├── transactions.csv
│   ├── accounts.csv
│   └── customers.csv
├── fraud_predictions.json
├── requirements.txt
├── .gitignore
└── README.md
```

Only include the supplied CSV files in the repository if redistribution is permitted by the hackathon rules.

## Setup

The notebook was developed for a GPU-enabled environment such as Google Colab with a T4 GPU.

Install dependencies:

```bash
pip install -r requirements.txt
```

Place the three CSV files in the same directory as the notebook:

```text
transactions.csv
accounts.csv
customers.csv
```

Then open:

```text
relational_data_wrangler_fraud_sentinel.ipynb
```

and execute the notebook from top to bottom.

The notebook downloads the Qwen model from Hugging Face, prepares it for 4-bit loading, fine-tunes a LoRA adapter, runs hybrid inference, validates the generated schema, and writes `fraud_predictions.json`.

## Testing

A standalone test suite validates the prompt-injection sanitizer against both known attack patterns and deliberately-varied phrasings, independent of the notebook (the provided dataset has no text field to exercise it against directly — see "Limitations").

```bash
pip install -r requirements-dev.txt
python -m pytest tests/test_sanitizer.py -v
```

23 tests, covering: known injection patterns (redacted), benign merchant names (untouched), missing values (handled safely), and known bypass cases (confirmed to currently evade detection — see "Limitations").

Note: `tests/test_sanitizer.py` duplicates the sanitizer logic from the notebook's Cell 6, since that logic isn't yet extracted into an importable module. See "Open Items."

## Reproducibility

The notebook includes deterministic random seeds, but exact model-training results can still vary with GPU/runtime/library versions. The training configuration is intentionally kept compatible with the environment used during development.

## Limitations

The supplied dataset does not include verified fraud labels, so conventional supervised metrics such as accuracy, precision, recall, and F1 cannot be calculated against ground truth.

The SLM is therefore used within a hybrid architecture rather than being treated as an independently validated fraud classifier.

The provided dataset does not include a text field (notes, description, etc.) on transactions, so the prompt-injection sanitizer 
(see "Adversarial Prompt Injection") is implemented but not exercised by the supplied data. A synthetic test suite (`tests/test_sanitizer.py`, 23 cases) 
validates the sanitizer separately: it correctly redacts all 11 known injection patterns and leaves benign merchant text untouched, but 5 deliberately-varied 
phrasings (paraphrases, a non-English phrase, spacing evasion, restructured sentences) confirm it as a brittle, pattern-matching defense — novel phrasing 
bypasses it. See "Testing" below to run it yourself.

## Open Items

- ~~Re-run fine-tuning with completion-only loss masking~~ — done; see "Results" above. Required combining `assistant_only_loss=True` with a higher learning rate and more epochs.
- No evaluation exists for the ambiguous middle band (score 1.0–4.0) that the model actually classifies in production, since there is no ground truth to check 
  against there. The 85% held-out figure only covers the same confident-bucket distribution used for training.
- LoRA config (`r=8`, attention-only target modules) and the exact learning-rate/epoch values were not swept — the current settings were found by manual 
  adjustment to fix the degenerate collapse, not tuned for best performance.
- Extract the sanitizer logic (currently only in notebook Cell 6) into a shared, importable module so `tests/test_sanitizer.py` doesn't need to duplicate it.

## Submission

The hackathon accepts either a completed Jupyter Notebook or a Python script. This repository uses the completed Jupyter Notebook because it captures the complete executed pipeline, preprocessing, fine-tuning, inference, validation, and output-generation workflow.
