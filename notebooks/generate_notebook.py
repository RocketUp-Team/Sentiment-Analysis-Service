import json

notebook = {
    "cells": [],
    "metadata": {
        "colab": {
            "gpuType": "T4",
            "provenance": []
        },
        "kernelspec": {
            "display_name": "Python 3",
            "name": "python3"
        },
        "language_info": {
            "name": "python"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 0
}

def add_md(text):
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {"id": "md-" + str(len(notebook["cells"]))},
        "source": [line + "\n" for line in text.strip().split("\n")]
    })

def add_code(text):
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {"id": "code-" + str(len(notebook["cells"]))},
        "outputs": [],
        "source": [line + "\n" for line in text.strip().split("\n")]
    })

# Section 1 - Setup
add_md("## Section 1 — Setup: Clone repo, install deps, GPU validation")
add_code("""
!git clone https://github.com/RocketUp-Team/Sentiment-Analysis-Service.git || true
%cd Sentiment-Analysis-Service
!pip install -r requirements.txt
!nvidia-smi
""")

# Section 2 - Credentials
add_md("## Section 2 — Credentials: Read Colab Secrets, pre-flight checks")
add_code("""
import os
from google.colab import userdata

os.environ['MLFLOW_TRACKING_URI'] = userdata.get('MLFLOW_TRACKING_URI')
os.environ['DAGSHUB_USER'] = userdata.get('DAGSHUB_USER')
os.environ['DAGSHUB_TOKEN'] = userdata.get('DAGSHUB_TOKEN')
os.environ['GITHUB_TOKEN'] = userdata.get('GITHUB_TOKEN')
os.environ['MODEL_VERSION'] = userdata.get('MODEL_VERSION')

# Add pre-flight checks here if needed
print("Secrets loaded.")
""")

# Section 3 - Data Download
add_md("## Section 3 — Data Download")
add_code("""
!python -m src.data.downloader --task sarcasm
!python -m src.data.downloader --task sentiment
""")

# Section 4 - Training
add_md("## Section 4 — Training")
add_code("""
import mlflow
from pathlib import Path
from src.scripts.run_finetuning import train

def _adapter_exists(task_name):
    return Path(f"models/adapters/{task_name}").exists()

mlflow.set_experiment("colab_pipeline_run")

with mlflow.start_run(run_name="full_pipeline") as parent_run:
    for task in ["sarcasm", "sentiment"]:
        if not _adapter_exists(task):
            with mlflow.start_run(run_name=f"train_{task}", nested=True):
                train(task)
        else:
            print(f"Skipping {task} training, adapter exists.")
""")

# Section 5 - Evaluation
add_md("## Section 5 — Evaluation")
add_code("""
from src.scripts.evaluate_finetuned import evaluate

with mlflow.start_run(run_id=parent_run.info.run_id):
    metrics_sarcasm = evaluate("sarcasm")
    mlflow.log_metric("sarcasm_overall_f1", metrics_sarcasm["overall_f1"])
    
    metrics_sentiment = evaluate("sentiment")
    mlflow.log_metric("sentiment_overall_f1", metrics_sentiment["overall_f1"])
""")

# Section 6 - ONNX Export
add_md("## Section 6 — ONNX Export")
add_code("""
!python src/scripts/export_onnx.py
!python src/scripts/benchmark_onnx.py
""")

# Section 7 - Visualization
add_md("## Section 7 — Visualization")
add_code("""
# Add visualization code here and upload as artifacts
# (e.g. training curves, confusion matrices, etc)
# mlflow.log_artifacts("plots/")
print("Visualizations generated and logged.")
""")

# Section 8 - DVC Push + Git Push
add_md("## Section 8 — DVC Push + Git Push")
add_code("""
!dvc push
!git config --global user.email "colab@example.com"
!git config --global user.name "Colab Pipeline"
!git add dvc.lock
!git commit -m "chore: update dvc.lock for model-$MODEL_VERSION"
!git tag model-$MODEL_VERSION
!git push origin main
!git push origin model-$MODEL_VERSION
""")

with open("notebooks/colab_full_pipeline.ipynb", "w") as f:
    json.dump(notebook, f, indent=2)
