"""
Sentiment analysis layer for detecting rants and emotional content.
Uses transformer-based models for sentiment classification.
"""

from typing import List, Dict, Any, Union
import torch
from .base_layer import BaseLayer


class SentimentAnalysisLayer(BaseLayer):
    """
    Layer for analyzing sentiment and detecting rants using transformer models.
    Identifies extremely negative, emotionally-charged, or unconstructive content.
    """
    
    def __init__(self, app_state, layer_name=None):
        super().__init__(app_state, layer_name or "SentimentAnalysisLayer")
        self._init_rant_indicators()
        
    def _init_rant_indicators(self):
        """Initialize rant detection indicators."""
        # Define rant keywords
        self.rant_keywords = [
            'terrible', 'awful', 'worst', 'horrible', 'disgusting', 'never again',
            'waste of money', 'scam', 'rude', 'unprofessional', 'incompetent'
        ]
        
    def _analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment using centralized transformer model."""
        try:
            # Use centralized models from app_state
            tokenizer = self.app_state.sentiment_tokenizer
            model = self.app_state.sentiment_model
            
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            
            with torch.no_grad():
                outputs = model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                
            # Get sentiment scores
            sentiment_scores = {
                'negative': float(predictions[0][0]),
                'neutral': float(predictions[0][1]), 
                'positive': float(predictions[0][2])
            }
            
            # Determine primary sentiment
            primary_sentiment = max(sentiment_scores, key=sentiment_scores.get)
            confidence = sentiment_scores[primary_sentiment]
            
            return {
                'primary_sentiment': primary_sentiment,
                'confidence': confidence,
                'scores': sentiment_scores
            }
            
        except Exception as e:
            self.logger.error(f"Sentiment analysis failed: {e}")
            return {
                'primary_sentiment': 'neutral',
                'confidence': 0.0,
                'scores': {'negative': 0.33, 'neutral': 0.34, 'positive': 0.33}
            }
    
    def _detect_rant_indicators(self, text: str) -> Dict[str, Any]:
        """Detect rant-specific indicators in text."""
        text_lower = text.lower()
        
        # Count rant keywords
        rant_keyword_count = sum(1 for keyword in self.rant_keywords if keyword in text_lower)
        
        # Count exclamation marks and caps
        exclamation_count = text.count('!')
        caps_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
        
        # Check for repetitive patterns
        words = text.split()
        repetitive_score = 0
        if len(words) > 1:
            repetitive_score = len(words) - len(set(w.lower() for w in words))
            repetitive_score = repetitive_score / len(words)
        
        return {
            'rant_keyword_count': rant_keyword_count,
            'exclamation_count': exclamation_count,
            'caps_ratio': caps_ratio,
            'repetitive_score': repetitive_score,
            'text_length': len(text)
        }
    
    def _compute_rant_score(self, sentiment_result: Dict[str, Any], 
                           indicators: Dict[str, Any]) -> float:
        """Compute overall rant score."""
        # Base score from negative sentiment
        base_score = sentiment_result['scores']['negative']
        
        # Add indicators
        rant_modifier = (
            indicators['rant_keyword_count'] * 0.2 +
            min(indicators['exclamation_count'] / 5, 1.0) * 0.15 +
            indicators['caps_ratio'] * 0.1 +
            indicators['repetitive_score'] * 0.1
        )
        
        # Length penalty for very short texts
        if indicators['text_length'] < 20:
            rant_modifier *= 0.5
            
        final_score = min(base_score + rant_modifier, 1.0)
        return final_score
    
    def run(self, input_data: Union[str, List[str]]) -> List[Dict[str, Any]]:
        """
        Analyze sentiment and detect rants in input texts.
        
        Args:
            input_data: Single text string or list of text strings
            
        Returns:
            List of sentiment analysis results
        """
        if isinstance(input_data, str):
            input_data = [input_data]
            
        try:
            self._log_execution_start(f"Analyzing sentiment for {len(input_data)} texts")
            
            results = []
            
            for i, text in enumerate(input_data):
                sentiment_result = self._analyze_sentiment(text)
                indicators = self._detect_rant_indicators(text)
                rant_score = self._compute_rant_score(sentiment_result, indicators)
                
                is_rant = (
                    rant_score > 0.7 and 
                    sentiment_result['primary_sentiment'] == 'negative' and
                    sentiment_result['confidence'] > 0.6
                )
                
                results.append({
                    'text_index': i,
                    'sentiment': sentiment_result['primary_sentiment'],
                    'sentiment_confidence': sentiment_result['confidence'],
                    'sentiment_scores': sentiment_result['scores'],
                    'is_rant': is_rant,
                    'rant_score': rant_score,
                    'rant_indicators': indicators,
                    'text_preview': text[:100] + "..." if len(text) > 100 else text
                })
            
            rant_count = sum(1 for r in results if r['is_rant'])
            self._log_execution_end(
                f"Detected {rant_count}/{len(results)} rants"
            )
            
            return results
            
        except Exception as e:
            self._handle_error(e, f"num_texts={len(input_data)}")
            raise
