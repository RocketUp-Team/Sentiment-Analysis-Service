---
title: Fine-tuning ABSA & Sarcasm Detection with MLflow Tracking
date: 2026-04-17
author: AI Assistant
status: Draft
---

# Fine-tuning ABSA & Sarcasm Detection with MLflow Tracking

## 1. Overview
The goal is to design and implement a fine-tuning pipeline for the existing Sentiment Analysis Service, ensuring high performance on Aspect-Based Sentiment Analysis (ABSA), adding Sarcasm Detection capabilities, and supporting **Multi-Language Inference**, all while tracking experiments with MLflow.

## 2. Goals & Metrics
- **Performance Metrics**: 
  - Sentiment F1-score (Macro) >= 0.80
  - Sarcasm Detection F1-score >= 0.80
  - ABSA Aspect F1 >= 0.70
  - **Inference Latency**: Strict P95 latency <= 200ms per request.
- **Multi-language Support**: Utilize cross-lingual transfer learning to support languages beyond English. Evaluate on at least one non-English holdout set.
- **Efficiency**: Utilize PEFT/LoRA ($r=8$) to fine-tune the existing heavy baseline models with a low hardware resource footprint (target GPU memory < 16GB).
- **Observability**: Complete MLflow integration to log hyperparameters, tracked metrics over epochs, and saved model weights.

## 3. Data Strategy (Hybrid Approach)
We primarily use English data relying on the model's innate Cross-Lingual Transfer capabilities.
- **Core ABSA & Sentiment Data**: Yelp Reviews Dataset / SemEval-2014 mapping out predefined categories (`food`, `service`, `ambiance`, `price`, `location`, `general`).
  - *Annotation Note*: Yelp data lacks native aspect labels. We will use the existing zero-shot baseline (DeBERTa) to generate "silver" aspect labels on a Yelp subset, unifying them with the SemEval-2014 schema.
- **Sarcasm Addition**: Explicitly labeled English sarcasm/irony dataset from HuggingFace (e.g., `tweet_eval` sub-task `irony`). The binary label (`ironic`) maps directly to `sarcasm_flag=True`.
  - *Ingestion Update*: `src/data/downloader.py` and the `download` DVC stage must be updated to pull both Yelp and `tweet_eval` alongside the existing SemEval dataset.
- **Data Mixing & Imbalance**: Yelp/SemEval data is much larger than `tweet_eval/irony`. We will use a 3:1 (Sentiment:Sarcasm) sampling ratio during batching, with oversampling for sarcasm minority classes.
- **Cross-Lingual Transfer**: By fine-tuning a multilingual base architecture on high-quality English data, the model generalizes the "concept" of Sarcasm and ABSA to other pre-trained languages.

## 4. Model Architecture & LoRA Strategy
To satisfy the multi-language challenge while handling multiple tasks efficiently:
- **Base Model Swap**: We will switch the current English-only baselines to their Multilingual equivalents:
  - Sequence Classification (Sentiment + Sarcasm): Migrate from RoBERTa to **XLM-RoBERTa** (`cardiffnlp/twitter-xlm-roberta-base-sentiment`). Instead of breaking pipeline compatibility with a custom multi-task head, we will use a single **Multi-Label Classification** head (`num_labels=4` mapping to `negative`, `neutral`, `positive`, and `sarcastic`) with `BCEWithLogitsLoss`.
  - ABSA/NLI: Migrate from DeBERTa to **mDeBERTa** (`MoritzLaurer/mDeBERTa-v3-base-mnli-xnli`).
- **PEFT/LoRA**: 
  - Rank $r=8$, Alpha $\alpha=16$, targeting `query` and `value` attention projections.
  - **Modules to Save**: `modules_to_save=["classifier"]` is strictly required. Changing to `num_labels=4` forces the classification head to be re-initialized, so it must be trained alongside the adapters.
  - Dropout $p=0.05$.
  - At inference time, LoRA adapters will be **merged** into the base model weights to eliminate adapter routing latency.

## 5. Hyperparameters
- **Optimizer**: AdamW (8-bit if memory constrained, otherwise standard 32-bit).
- **Learning Rate**: 2e-4 with linear warmup and cosine decay.
- **Batch Size**: 16 or 32 per device (with gradient accumulation of 2 if needed).
- **Epochs**: 3-5 epochs.

## 6. Evaluation & Validation
- **Datasets**: 
  - Evaluate ABSA using the **SemEval-2014 test gold** split.
  - Evaluate Sarcasm using the test split of **tweet_eval**.
  - Cross-lingual evaluation: Utilize the **Multilingual Amazon Reviews Corpus (MARC)** to benchmark transfer performance natively without manual curation effort.
- **Metrics definition**: Macro F1 calculated independently for Sentiment (3-class), Sarcasm (binary), and Aspect extraction.

## 7. Migration & Backward Compatibility
- **API Contract**: The `/predict` API schema remains unchanged. `sarcasm_flag` (currently hardcoded to `False`) will be actively populated by the multi-task model output.
- **Config Flags**: Create a `ModelConfig_v2` in `config.py` temporarily or use a feature flag `USE_FINETUNED_MODELS=True` to gate the new model usage. Label maps (varying between XLM-RoBERTa and RoBERTa) will be handled independently.
- **Rollback**: If the fine-tuned model underperforms, the system can smoothly fall back to `baseline.py` configurations safely.

## 8. MLflow Integration
A new entrypoint script `src/scripts/run_finetuning.py` abstracting the fine-tuning logic:
- **Parameters**: `mlflow.log_params()` captures learning rates, epochs, LoRA combinations.
- **Metrics**: End-of-epoch triggers will measure macro F1, and log them back to MLflow.
- **Artifacts**: Store merged LoRA checkpoints via `mlflow.pytorch.log_model()`.

## 9. Dependency Updates
Update `requirements.txt` to include:
- `peft`
- `datasets`
- `sentencepiece`
- `accelerate`

## 10. Directory Structure & DVC Additions
Fine-tuning must be integrated as a DVC stage rather than an untracked script.
```text
Sentiment-Analysis-Service/
├── src/
│   ├── training/
│   │   ├── __init__.py 
│   │   ├── trainer.py           # Wrapped HF Trainer with MLflow
│   │   ├── lora_config.py       # Configuration specifics for PEFT
│   │   └── data_mixer.py        # Logic to merge Yelp + Sarcasm HF datasets
│   └── scripts/
│       └── run_finetuning.py    # DVC-invoked main script
├── dvc.yaml                     # Updated with `train` and `eval` stages
├── mlruns/                      # MLflow local tracking storage (auto-generated)
```

## 11. Timeline & Phasing
- **Phase 1**: Sarcasm fine-tuning + Multi-task head setup. Integration into inference flow.
- **Phase 2**: ABSA fine-tuning with mDeBERTa and silver Yelp labels.
- **Phase 3**: Evaluation, LoRA merging, Multi-language testing, and full release.

## 12. Hardware Requirements & Risks
- **Hardware**: Targeting modern consumer GPUs (e.g., RTX 3090 / 4090 with 24GB VRAM) or cloud T4/L4 instances. Expect memory usage ~12-14 GB with $r=8$ LoRA on base models.
- **Risks**: 
  - **Silver-Labeling**: Silver-labeling Yelp with baseline DeBERTa might reinforce baseline errors. *Mitigation*: rigorously evaluate against the gold SemEval dataset.
  - **Explainability (SHAP) Regression**: If model architectures deviate from standard pipelines, `shap.Explainer` will fail. *Mitigation*: The proposed Multi-Label Architecture (num_labels=4) strictly maintains `AutoModelForSequenceClassification` compatibility, ensuring SHAP values can still be extracted without writing custom PyTorch forward wrappers.
