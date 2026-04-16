# Fine-tuning ABSA & Sarcasm Detection with MLflow Tracking

## 1. Overview
The goal is to design and implement a fine-tuning pipeline for the existing Sentiment Analysis Service, ensuring high performance on Aspect-Based Sentiment Analysis (ABSA), adding Sarcasm Detection capabilities, and supporting **Multi-Language Inference**, all while tracking experiments with MLflow.

## 2. Goals & Metrics
- **Performance Metrics**: 
  - F1-score (Macro) >= 0.80
  - ABSA Aspect F1 >= 0.70
- **Multi-language Support**: Utilize cross-lingual transfer learning to support languages beyond English.
- **Efficiency**: Utilize PEFT/LoRA to fine-tune the existing heavy baseline models with a low hardware resource footprint.
- **Observability**: Complete MLflow integration to log hyperparameters, tracked metrics over epochs, and saved model weights.

## 3. Data Strategy (Hybrid Approach)
We primarily use English data relying on the model's innate Cross-Lingual Transfer capabilities:
- **Core ABSA & Sentiment Data**: Yelp Reviews Dataset / SemEval-2014 mapping out predefined categories (`food`, `service`, `ambiance`, `price`, `general`).
- **Sarcasm Addition**: Explicitly labeled English sarcasm/irony dataset from HuggingFace (e.g., `tweet_eval` sub-task `irony`).
- **Cross-Lingual Transfer**: By fine-tuning a multilingual base architecture on high-quality English data, the model generalizes the "concept" of Sarcasm and ABSA to other pre-trained languages (Zero-Shot Cross-Lingual Inference). No need to rebuild a massive multilingual dataset.

## 4. Model Architecture & LoRA Strategy
To satisfy the multi-language challenge while handling multiple tasks efficiently:
- **Base Model Swap**: We will switch the current English-only baselines to their Multilingual equivalents:
  - Sequence Classification: Migrate from RoBERTa to **XLM-RoBERTa** (`cardiffnlp/twitter-xlm-roberta-base-sentiment`).
  - ABSA/NLI: Migrate from DeBERTa to **mDeBERTa** (`MoritzLaurer/mDeBERTa-v3-base-mnli-xnli`).
- **PEFT/LoRA**: We will mount LoRA adapters (Rank $r=16$) on the Attention projections of these models. This freezes the massive multilingual embedding layers and only updates a tiny subset of parameters.

## 5. MLFlow Integration
A new entrypoint `train.py` abstracting the fine-tuning logic:
- **Parameters**: `mlflow.log_params()` captures learning rates, epochs, LoRA combinations.
- **Metrics**: End-of-epoch triggers will measure macro F1, and log them back to MLflow.
- **Artifacts**: Store LoRA checkpoints via `mlflow.pytorch.log_model()`.

## 6. Directory Structure Additions
```text
Sentiment-Analysis-Service/
├── src/
│   ├── training/
│   │   ├── __init__.py 
│   │   ├── trainer.py           # Wrapped HF Trainer with MLFlow
│   │   ├── lora_config.py       # Configuration specifics for PEFT
|   |   └── data_mixer.py        # Logic to merge Yelp + Sarcasm HF datasets
│   └── scripts/
│       └── run_finetuning.py    # Main script to execute
├── mlruns/                      # MLflow local tracking storage (auto-generated)
```
