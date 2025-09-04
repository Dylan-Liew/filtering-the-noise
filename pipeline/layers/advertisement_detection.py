from typing import List, Dict, Any
import re
from .base_layer import BaseLayer


class AdvertisementDetectionInput:
    """Input data structure for advertisement detection."""
    
    def __init__(self, texts: List[str], metadata: List[Dict[str, Any]] = None):
        self.texts = texts
        self.metadata = metadata or [{}] * len(texts)


class AdvertisementDetectionOutput:
    """Output data structure for advertisement detection results."""
    
    def __init__(self, detections: List[Dict[str, Any]]):
        self.detections = detections


class AdvertisementDetectionLayer(BaseLayer):
    """
    Layer for detecting advertisements and promotional content in text reviews.
    Uses both rule-based patterns and semantic similarity to identify ads.
    """
    
    def __init__(self, app_state, layer_name=None):
        super().__init__(app_state, layer_name)
        self._init_patterns()
        
    def _init_patterns(self):
        """Initialize advertisement detection patterns."""
        self.promotional_patterns = [
            r'\b(call now|limited time|special offer|discount|sale|promotion)\b',
            r'\b(visit our|check out our|follow us|subscribe)\b',
            r'\b(www\.|http|\.com|\.org|\.net)\b',
            r'\b(\d{3}[-.]?\d{3}[-.]?\d{4})\b',  # Phone numbers
            r'\b(email|contact|reach out)\b.*@',
            r'\b(free|cheap|affordable|best price|lowest price)\b',
            r'\b(buy now|order now|shop now|click here)\b',
        ]
        
        self.promotional_keywords = [
            'advertisement', 'sponsored', 'promo', 'deal', 'coupon',
            'affiliate', 'referral', 'partnership', 'collaboration'
        ]
        
    def _detect_promotional_patterns(self, text: str) -> Dict[str, Any]:
        """Detect promotional patterns in text using regex."""
        text_lower = text.lower()
        
        pattern_matches = []
        for pattern in self.promotional_patterns:
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            if matches:
                pattern_matches.extend(matches)
        
        keyword_matches = [kw for kw in self.promotional_keywords if kw in text_lower]
        
        return {
            'pattern_matches': pattern_matches,
            'keyword_matches': keyword_matches,
            'pattern_score': len(pattern_matches),
            'keyword_score': len(keyword_matches)
        }
    
    def _compute_semantic_similarity(self, text: str, ad_templates: List[str]) -> float:
        """Compute semantic similarity with known advertisement templates."""
        try:
            model = self.app_state.sentence_transformer
            
            text_embedding = model.encode([text])
            template_embeddings = model.encode(ad_templates)
            
            similarities = model.similarity(text_embedding, template_embeddings)
            max_similarity = float(similarities.max())
            
            return max_similarity
        except Exception as e:
            self.logger.warning(f"Failed to compute semantic similarity: {e}")
            return 0.0
    
    def run(self, input_data: AdvertisementDetectionInput) -> List[Dict[str, Any]]:
        """
        Detect advertisements in input texts.
        
        Args:
            input_data: AdvertisementDetectionInput containing texts and metadata
            
        Returns:
            AdvertisementDetectionOutput with detection results
        """
        try:
            self._log_execution_start(f"Processing {len(input_data.texts)} texts for ads")
            
            ad_templates = [
                "Visit our website for more information and special offers",
                "Call us now for the best deals and discounts available",
                "Check out our amazing products and services today",
                "Limited time offer - don't miss this opportunity",
                "Follow us on social media for exclusive promotions"
            ]
            
            detections = []
            
            for i, text in enumerate(input_data.texts):
                metadata = input_data.metadata[i]
                
                pattern_result = self._detect_promotional_patterns(text)
                
                semantic_score = self._compute_semantic_similarity(text, ad_templates)
                
                total_score = (
                    pattern_result['pattern_score'] * 0.4 +
                    pattern_result['keyword_score'] * 0.3 +
                    semantic_score * 0.3
                )
                
                is_advertisement = total_score > 1.0
                confidence = min(total_score / 3.0, 1.0)
                
                detections.append({
                    'text_index': i,
                    'is_advertisement': is_advertisement,
                    'confidence': confidence,
                    'total_score': total_score,
                    'pattern_matches': pattern_result['pattern_matches'],
                    'keyword_matches': pattern_result['keyword_matches'],
                    'semantic_score': semantic_score,
                    'metadata': metadata
                })
            

            ad_count = sum(1 for d in detections if d['is_advertisement'])
            self._log_execution_end(
                f"Detected {ad_count}/{len(detections)} advertisements"
            )
            
            return detections
            
        except Exception as e:
            self._handle_error(e, f"num_texts={len(input_data.texts)}")
            raise