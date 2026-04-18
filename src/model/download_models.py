import logging
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from src.model.config import ModelConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download():
    config = ModelConfig()
    
    logger.info(f"Downloading main sentiment model: {config.model_name}")
    AutoTokenizer.from_pretrained(config.model_name)
    AutoModelForSequenceClassification.from_pretrained(config.model_name, use_safetensors=True)
    
    logger.info(f"Downloading ABSA model: {config.absa_model_name}")
    # Zero-shot classification pipeline handles model/tokenizer download
    pipeline("zero-shot-classification", model=config.absa_model_name, model_kwargs={"use_safetensors": True})
    
    logger.info("All models cached successfully in ~/.cache/huggingface")

if __name__ == "__main__":
    download()
