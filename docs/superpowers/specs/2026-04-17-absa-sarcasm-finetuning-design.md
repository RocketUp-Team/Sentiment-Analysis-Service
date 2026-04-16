# Fine-tuning ABSA & Sarcasm Detection with MLflow Tracking

## 1. Overview
The goal is to design and implement a fine-tuning pipeline for the existing Sentiment Analysis Service, ensuring high performance on Aspect-Based Sentiment Analysis (ABSA) and adding Sarcasm Detection capabilities, all while tracking experiments with MLflow.

## 2. Goals & Metrics
- **Performance Metrics**: 
  - F1-score (Macro) >= 0.80
  - ABSA Aspect F1 >= 0.70
- **Efficiency**: Utilize PEFT/LoRA to fine-tune the existing heavy baseline models with low hardware resource footprint.
- **Observability**: Complete MLflow integration to log hyperparameters, tracked metrics over epochs, and saved model weights.

## 3. Data Strategy (Hybrid Approach)
Instead of relying solely on the constraints of a specific single dataset, the system will construct a hybrid training set:
- **Core ABSA & Sentiment Data**: Yelp Reviews Dataset / SemEval-2014 mapping out predefined categories (`food`, `service`, `ambiance`, `price`, `general`).
- **Sarcasm Addition**: Pull an explicitly labeled sarcasm/irony dataset from HuggingFace (e.g., `tweet_eval` sub-task `irony` or `iSarcasm`) to teach the model implicit semantic flips.
- **Preprocessing Flow**: Unified pipeline outputting a standardized format `(text, aspect, sentiment, is_sarcasm)` bridging both datasets.

## 4. Model Architecture & LoRA Strategy
To manage multiple tasks efficiently and satisfy the **Multi-language Support** requirement:
- **Base Model Swap**: Before/during fine-tuning, `config.py` will point `model_name` to `cardiffnlp/twitter-xlm-roberta-base-sentiment` (or similar XLM-R variant) and `absa_model_name` to `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli`. This guarantees zero-shot cross-lingual capabilities.
- **LoRA Adapters**: We will mount LoRA adapters (using `peft` from HuggingFace) on the existing baseline architectures.
- **Rank setting**: $r=8$ or $r=16$ targeting only Q (Query) and V (Value) attention projections to substantially reduce memory utilization and freeze the core base model embeddings.

## 5. MLFlow Integration
A new entrypoint `train.py` (or pipeline extension `src/training/`) will abstract the fine-tuning logic:
- **Parameters**: `mlflow.log_params()` will capture learning rates, epochs, LoRA combinations.
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
