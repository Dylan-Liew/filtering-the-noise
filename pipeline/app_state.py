from typing import Optional
from sentence_transformers import SentenceTransformer
from transformers import AutoImageProcessor, AutoModelForImageClassification, MobileNetV2ImageProcessor, \
    MobileNetV2ForImageClassification, AutoTokenizer, AutoModelForSequenceClassification, RobertaTokenizer, \
    RobertaForSequenceClassification
import torch
from config.models import ModelsConfig


class AppState:
    """
    Centralized state management for ML models and configurations.
    Handles model initialization, device management, and provides dependency injection.
    """

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # None cause lazy loading
        self._sentence_transformer: Optional[SentenceTransformer] = None
        self._image_processor: Optional[AutoImageProcessor] = None
        self._image_model: Optional[AutoModelForImageClassification] = None
        self._spam_tokenizer: Optional[AutoTokenizer] = None
        self._spam_model: Optional[AutoModelForSequenceClassification] = None
        self._sentiment_tokenizer: Optional[AutoTokenizer] = None
        self._sentiment_model: Optional[AutoModelForSequenceClassification] = None

    @property
    def sentence_transformer(self) -> SentenceTransformer:
        if self._sentence_transformer is None:
            config = ModelsConfig.get_sentence_transformer_config()

            model_kwargs = config.model_kwargs.copy()
            if "device_map" in model_kwargs:
                model_kwargs["device_map"] = self.device

            self._sentence_transformer = SentenceTransformer(
                config.name,
                model_kwargs=model_kwargs,
                tokenizer_kwargs=config.tokenizer_kwargs,
            )
        return self._sentence_transformer

    @property
    def image_processor(self) -> MobileNetV2ImageProcessor:
        if self._image_processor is None:
            config = ModelsConfig.get_image_processor_config()
            self._image_processor: MobileNetV2ImageProcessor = (AutoImageProcessor
                                                                .from_pretrained(config.name))
        return self._image_processor

    @property
    def image_model(self) -> MobileNetV2ForImageClassification:
        if self._image_model is None:
            config = ModelsConfig.get_image_classifier_config()
            self._image_model: MobileNetV2ForImageClassification = (AutoModelForImageClassification.
                                                                    from_pretrained(config.name))

        return self._image_model

    @property
    def spam_tokenizer(self) -> RobertaTokenizer:
        if self._spam_tokenizer is None:
            config = ModelsConfig.get_spam_detection_config()
            self._spam_tokenizer: RobertaTokenizer = AutoTokenizer.from_pretrained(
                config.name,
                **config.tokenizer_kwargs
            )
        return self._spam_tokenizer

    @property
    def spam_model(self) -> RobertaForSequenceClassification:
        if self._spam_model is None:
            config = ModelsConfig.get_spam_detection_config()
            
            model_kwargs = config.model_kwargs.copy()
            if "device_map" in model_kwargs:
                model_kwargs["device_map"] = self.device
                
            self._spam_model: RobertaForSequenceClassification = AutoModelForSequenceClassification.from_pretrained(
                config.name, 
                **model_kwargs
            )
        return self._spam_model

    @property
    def sentiment_tokenizer(self) -> RobertaTokenizer:
        if self._sentiment_tokenizer is None:
            config = ModelsConfig.get_sentiment_analysis_config()
            self._sentiment_tokenizer: RobertaTokenizer = AutoTokenizer.from_pretrained(
                config.name,
                **config.tokenizer_kwargs
            )
        return self._sentiment_tokenizer

    @property
    def sentiment_model(self) -> RobertaForSequenceClassification:
        if self._sentiment_model is None:
            config = ModelsConfig.get_sentiment_analysis_config()
            
            model_kwargs = config.model_kwargs.copy()
            if "device_map" in model_kwargs:
                model_kwargs["device_map"] = self.device
                
            self._sentiment_model: RobertaForSequenceClassification = AutoModelForSequenceClassification.from_pretrained(
                config.name, 
                **model_kwargs
            )
        return self._sentiment_model
