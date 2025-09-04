"""
Model configuration for the filtering pipeline.
Contains model names, parameters, and device settings.
"""

from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class ModelConfig:
    """Configuration for individual models."""
    name: str
    model_kwargs: Dict[str, Any] = None
    tokenizer_kwargs: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.model_kwargs is None:
            self.model_kwargs = {}
        if self.tokenizer_kwargs is None:
            self.tokenizer_kwargs = {}


class ModelsConfig:
    """Central configuration for all models used in the pipeline."""
    
    # Sentence Transformer Models
    SENTENCE_TRANSFORMER = ModelConfig(
        name="Qwen/Qwen3-Embedding-0.6B",
        model_kwargs={
            "device_map": "auto"
        },
        tokenizer_kwargs={
            "padding_side": "left"
        }
    )

    # Image Classification Models
    IMAGE_PROCESSOR = ModelConfig(
        name="google/mobilenet_v2_1.0_224"
    )

    IMAGE_CLASSIFICATION = ModelConfig(
        name="google/mobilenet_v2_1.0_224"
    )

    # Spam Detection Models
    SPAM_DETECTION = ModelConfig(
        name="mshenoda/roberta-spam",
        model_kwargs={
            "device_map": "auto"
        },
        tokenizer_kwargs={
            "padding": True,
            "truncation": True,
            "return_tensors": "pt"
        }
    )

    # Sentiment Analysis Models
    SENTIMENT_ANALYSIS = ModelConfig(
        name="cardiffnlp/twitter-roberta-base-sentiment-latest",
        model_kwargs={
            "device_map": "auto"
        },
        tokenizer_kwargs={
            "truncation": True,
            "max_length": 512,
            "return_tensors": "pt"
        }
    )
    


    @classmethod
    def get_sentence_transformer_config(cls) -> ModelConfig:
        return cls.SENTENCE_TRANSFORMER

    @classmethod
    def get_image_processor_config(cls) -> ModelConfig:
        return cls.IMAGE_PROCESSOR

    @classmethod
    def get_image_classifier_config(cls) -> ModelConfig:
        return cls.IMAGE_CLASSIFICATION

    @classmethod
    def get_spam_detection_config(cls) -> ModelConfig:
        return cls.SPAM_DETECTION

    @classmethod
    def get_sentiment_analysis_config(cls) -> ModelConfig:
        return cls.SENTIMENT_ANALYSIS