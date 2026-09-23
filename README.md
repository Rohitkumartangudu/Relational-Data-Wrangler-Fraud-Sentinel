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

Evaluated on a 40-example held-out split (20 fraud / 20 non-fraud, stratified,
never seen during fine-tuning). See "Evaluation Methodology" below for what
this measures and does not measure.

| Model             | Valid JSON | Agreement with weak label | Behavior                        |
|--------------------|-----------|----------------------------|----------------------------------|
| Base (no adapter)  | 38/40     | 19/38 (50.0%)               | Predicted `is_fraud: true` for every valid output |
| Fine-tuned (LoRA)  | 36/40     | 18/36 (50.0%)               | Predicted `is_fraud: true` for every valid output |

Both models collapsed to always predicting fraud, regardless of the input features — the 50% agreement is exactly what a coin flip would score against
a balanced set, not evidence of reasoning. This was confirmed by inspecting raw generations directly: justifications cited features inconsistently
(e.g. citing *low* credit utilization, normally a low-risk signal, as evidence of fraud), and a follow-up check showed the fine-tuned model also
scored 50% (9/18) on a sample of its own **training** data — meaning it had not learned to discriminate the task at all, not merely failed to generalize.

### Root cause

The SFT training cell builds each example as a single flat chat-template string (system + user + assistant turns) and hands it to `SFTTrainer` with
no completion-only loss masking. By default, this computes training loss across every token in the sequence, including the long, largely-boilerplate
user turn (transaction JSON + instructions). The assistant turn — the one part that actually encodes the fraud/non-fraud label — is a small fraction
of the total tokens, so its contribution to the gradient was diluted by the much longer, more repetitive, and easier-to-predict prompt text. Loss
dropped during training, but that mostly reflects the model getting better at predicting prompt boilerplate, not at classifying transactions.

A fix (completion-only loss masking) is scoped as follow-up work; see [open items] below.

### Evaluation Methodology

This evaluates agreement between each model's `is_fraud` output and the weak-label rule, on examples held out from fine-tuning but drawn from the
same confident-bucket distribution (score ≥4 or ≤1) used to build the training set. It does not measure performance on the ambiguous middle band
(score 1.0–4.0) that the model actually classifies in production, since no ground truth exists there to check against. Given the result above, it also
currently cannot demonstrate any benefit from fine-tuning, since neither model learned the task.

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

## Reproducibility

The notebook includes deterministic random seeds, but exact model-training results can still vary with GPU/runtime/library versions. The training configuration is intentionally kept compatible with the environment used during development.

## Limitations

The supplied dataset does not include verified fraud labels, so conventional supervised metrics such as accuracy, precision, recall, and F1 cannot be calculated against ground truth.

The SLM is therefore used within a hybrid architecture rather than being treated as an independently validated fraud classifier.

The provided dataset does not include a text field (notes, description, etc.) on transactions, so the prompt-injection sanitizer (see "Adversarial Prompt Injection") is implemented but not exercised by the supplied data. A synthetic test suite with injected adversarial notes is used separately to validate the sanitizer (see `tests/` — added in a later update).

## Open Items

Re-run fine-tuning with completion-only loss masking (mask the loss to the assistant turn only) and re-evaluate against the same held-out split, to test whether
the model can actually learn to discriminate fraud vs. non-fraud once the training signal isn't diluted by prompt tokens.

## Submission

The hackathon accepts either a completed Jupyter Notebook or a Python script. This repository uses the completed Jupyter Notebook because it captures the complete executed pipeline, preprocessing, fine-tuning, inference, validation, and output-generation workflow.
