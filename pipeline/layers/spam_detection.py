from typing import Dict, Any
import torch
from .base_layer import BaseLayer


class SpamDetectionInput:
    """Input data structure for spam detection analysis."""
    
    def __init__(self, review_text: str):
        self.review_text = review_text


class SpamDetectionOutput:
    """Output data structure for spam detection results."""
    
    def __init__(self, is_spam: bool, spam_probability: float, predicted_label: str, confidence_scores: Dict[str, float]):
        self.is_spam = is_spam
        self.spam_probability = spam_probability
        self.predicted_label = predicted_label
        self.confidence_scores = confidence_scores


class SpamDetectionLayer(BaseLayer):
    """
    Layer for detecting spam content in reviews using RoBERTa-based classification.
    Uses the mshenoda/roberta-spam model for accurate spam/ham classification.
    """
    
    def __init__(self, app_state, layer_name=None):
        super().__init__(app_state, layer_name or "SpamDetectionLayer")
    
    def _classify_text(self, text: str) -> Dict[str, Any]:
        """Classify text as spam or ham using the RoBERTa model."""
        tokenizer = self.app_state.spam_tokenizer
        model = self.app_state.spam_model
        
        # Tokenize the input text
        inputs = tokenizer([text], padding=True, truncation=True, return_tensors="pt")
        
        # Move inputs to the same device as the model
        inputs = {key: value.to(model.device) for key, value in inputs.items()}
        
        # Model inference
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Post-processing
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        predicted_class_id = torch.argmax(probabilities, dim=-1)
        
        # Get the predicted label
        predicted_label = model.config.id2label[predicted_class_id.item()]
        
        # Get confidence scores for all labels
        confidence_scores = {
            model.config.id2label[i]: score.item() 
            for i, score in enumerate(probabilities[0])
        }
        
        label_mapping = {
            "LABEL_0": "HAM",
            "LABEL_1": "SPAM"
        }
        
        mapped_confidence_scores = {}
        mapped_predicted_label = predicted_label
        
        for label, score in confidence_scores.items():
            mapped_label = label_mapping.get(label, label)
            mapped_confidence_scores[mapped_label] = score
            if label == predicted_label:
                mapped_predicted_label = mapped_label
        
        return {
            "predicted_label": mapped_predicted_label,
            "confidence_scores": mapped_confidence_scores,
            "probabilities": probabilities[0]
        }
    
    def run(self, input_data: SpamDetectionInput) -> SpamDetectionOutput:
        """
        Analyze review text for spam content.
        
        Args:
            input_data: SpamDetectionInput containing review text
            
        Returns:
            SpamDetectionOutput with spam classification and confidence scores
        """
        try:
            self._log_execution_start(f"Analyzing text for spam detection")
            
            # Classify the text
            classification_result = self._classify_text(input_data.review_text)
            
            predicted_label = classification_result["predicted_label"]
            confidence_scores = classification_result["confidence_scores"]
            
            # Determine if text is spam
            is_spam = predicted_label == "SPAM"
            spam_probability = confidence_scores.get("SPAM", 0.0)
            
            output = SpamDetectionOutput(
                is_spam=is_spam,
                spam_probability=spam_probability,
                predicted_label=predicted_label,
                confidence_scores=confidence_scores
            )
            
            self._log_execution_end(
                f"Spam Detection: {'SPAM' if is_spam else 'HAM'} "
                f"(confidence: {spam_probability:.3f})"
            )
            
            return output
            
        except Exception as e:
            self._handle_error(e, f"text_length={len(input_data.review_text)}")
            raise