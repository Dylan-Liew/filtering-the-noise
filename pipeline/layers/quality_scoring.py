from typing import List, Dict, Any, Union
import re
from sentence_transformers import util
from .base_layer import BaseLayer


class QualityScoringLayer(BaseLayer):
    """
    Layer for scoring review usefulness using transformer-based analysis.
    Evaluates informativeness, helpfulness, and overall value.
    """
    
    def __init__(self, app_state, layer_name=None):
        super().__init__(app_state, layer_name or "QualityScoringLayer")
        self._init_scoring_criteria()
        
    def _init_scoring_criteria(self):
        """Initialize scoring criteria and templates."""
        self.informative_templates = [
            "The review describes the service quality and staff behavior.",
            "The review mentions specific details about the location, parking, or accessibility.",
            "The review discusses the atmosphere, ambiance, or cleanliness of the place.",
            "The review provides information about pricing, costs, or value for money.",
            "The review mentions wait times, busy periods, or reservation details.",
            "The review describes food quality, menu items, or dining experience.",
            "The review discusses business hours, availability, or scheduling.",
            "The review compares this business to other similar establishments.",
            "The review shares a specific incident, experience, or visit details.",
            "The review provides practical tips, recommendations, or advice for future visitors.",
            "The review mentions facilities like restrooms, Wi-Fi, or special amenities.",
            "The review describes the location's convenience, transportation, or neighborhood.",
            "The review discusses customer service interactions or staff helpfulness.",
            "The review mentions special events, promotions, or seasonal offerings.",
            "The review provides details about product/service availability or selection.",
            "The review describes safety, security, or comfort aspects of the location."
        ]
        
        self.quality_aspects = [
            "specific details about products or services",
            "helpful information for other customers", 
            "constructive feedback and suggestions",
            "clear description of experience",
            "balanced and fair evaluation",
            "useful tips and recommendations"
        ]
        
        self.low_quality_indicators = [
            "very short and uninformative",
            "purely emotional without details",
            "generic and vague comments",
            "irrelevant personal information",
            "spam or promotional content"
        ]
        
        try:
            model = self.app_state.sentence_transformer
            instructed_templates = [
                f"Represent this review template for semantic search: {template}"
                for template in self.informative_templates
            ]
            self.template_embeddings = model.encode(
                instructed_templates, 
                convert_to_tensor=True
            )
        except Exception as e:
            self.logger.error(f"Failed to precompute template embeddings: {e}")
            raise RuntimeError(f"QualityScoringLayer initialization failed: Unable to compute template embeddings. {e}") from e
        
    def _analyze_informativeness_semantic(self, text: str, threshold: float = 0.25) -> Dict[str, Any]:
        """Analyze informativeness using semantic similarity with Qwen3 embeddings."""
        
        try:
            # Split review into sentences for more granular matching
            import re
            sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
            if not sentences:
                words = text.split()
                return {
                    'matched_templates': [],
                    'informativeness_score': 0.0,
                    'word_count': len(words),
                    'detail_score': 0.0,
                    'question_score': 0.0,
                    'high_confidence_matches': 0
                }

            model = self.app_state.sentence_transformer
            # Use instruction-aware prompting for review sentences too
            instructed_sentences = [
                f"Represent this review sentence for semantic search: {sentence}"
                for sentence in sentences
            ]
            review_embeddings = model.encode(instructed_sentences, convert_to_tensor=True)

            cosine_scores = util.cos_sim(review_embeddings, self.template_embeddings)

            matched_templates = set()
            high_confidence_matches = 0
            
            # Find the best match for each sentence
            for i in range(len(sentences)):
                # Get the highest score for the current sentence across all templates
                best_score_for_sentence = cosine_scores[i].max()
                if best_score_for_sentence > threshold:
                    template_index = cosine_scores[i].argmax()
                    matched_templates.add(self.informative_templates[template_index])
                    if best_score_for_sentence > 0.5:  # High confidence match
                        high_confidence_matches += 1
            
            # Informativeness score logic update:
            # We don't expect a review to cover ALL 16 templates.
            # If it covers 3-4 distinct informative aspects, it's a very good review.
            SATURATION_POINT = 3.0
            base_informativeness_score = min(len(matched_templates) / SATURATION_POINT, 1.0)
            
            # Boost score based on high-confidence matches
            # If we have strong matches, we boost confidence
            confidence_boost = min(high_confidence_matches * 0.1, 0.2)
            informativeness_score = min(base_informativeness_score + confidence_boost, 1.0)
            
            words = text.split()
            word_count = len(words)
            
            # Compute descriptive indicators for location reviews (expanded list)
            descriptive_indicators = ['specifically', 'located', 'visited', 'experience', 'recommend', 'suggest', 
                                    'definitely', 'exactly', 'found', 'counted', 'made', 'bought', 'went', 'happy']
            question_indicators = ['what', 'how', 'when', 'where', 'why', 'which']
            
            descriptive_score = min(sum(1 for word in descriptive_indicators if word in text.lower()) / 2.0, 1.0)
            question_score = sum(1 for q in question_indicators if q in text.lower()) / len(question_indicators)
            
            # Combine semantic and traditional scoring
            detail_score = (informativeness_score * 0.8) + (descriptive_score * 0.2)
            
            return {
                'matched_templates': list(matched_templates),
                'informativeness_score': round(informativeness_score, 3),
                'word_count': word_count,
                'detail_score': round(detail_score, 3),
                'question_score': round(question_score, 3),
                'high_confidence_matches': high_confidence_matches
            }
            
        except Exception as e:
            self.logger.error(f"Semantic informativeness analysis failed: {e}")
            raise
    
    
    
    def _compute_semantic_quality(self, text: str) -> float:
        """Compute semantic quality using sentence transformer."""
        try:
            model = self.app_state.sentence_transformer
            
            # Create quality templates
            high_quality_templates = [
                "This review provides specific and helpful information about the business",
                "The reviewer shares detailed experience with useful insights",
                "This is a constructive and informative review with specific details"
            ]
            
            low_quality_templates = [
                "This is a short and uninformative review",
                "The review is vague and doesn't provide useful information",
                "This is an emotional rant without constructive details"
            ]
            
            text_embedding = model.encode([text])
            high_quality_embeddings = model.encode(high_quality_templates)
            low_quality_embeddings = model.encode(low_quality_templates)
            
            high_quality_similarity = model.similarity(text_embedding, high_quality_embeddings).max()
            low_quality_similarity = model.similarity(text_embedding, low_quality_embeddings).max()
            
            # Compute relative quality score
            quality_score = float(high_quality_similarity) - float(low_quality_similarity) * 0.5
            quality_score = max(0.0, min(1.0, (quality_score + 1) / 2))
            
            return quality_score
            
        except Exception as e:
            self.logger.warning(f"Semantic quality computation failed: {e}")
            return 0.5
    
    def _compute_helpfulness_score(self, text: str, informativeness: Dict[str, Any], 
                                  semantic_quality: float) -> Dict[str, float]:
        """Compute overall helpfulness and usefulness scores based purely on semantic content."""
        
        # Base score from informativeness (rebalanced for real-world reviews)
        info_score = (
            informativeness['detail_score'] * 0.4 +
            informativeness['question_score'] * 0.1 +
            informativeness['informativeness_score'] * 0.3 +
            semantic_quality * 0.2
        )
        
        # Usefulness score (more specific criteria)
        usefulness_score = info_score
        if 'recommend' in text.lower() or 'suggest' in text.lower():
            usefulness_score += 0.1
        if any(word in text.lower() for word in ['tip', 'advice', 'warning', 'note']):
            usefulness_score += 0.1
            
        return {
            'helpfulness_score': min(info_score, 1.0),
            'usefulness_score': min(usefulness_score, 1.0),
            'informativeness_score': info_score
        }
    
    def run(self, input_data: Union[str, List[str]]) -> List[Dict[str, Any]]:
        """
        Score reviews for usefulness and quality.
        
        Args:
            input_data: Single text string or list of text strings
            
        Returns:
            List of scoring results
        """
        if isinstance(input_data, str):
            input_data = [input_data]
            
        try:
            self._log_execution_start(f"Scoring {len(input_data)} reviews")
            
            results = []
            
            for i, text in enumerate(input_data):
                informativeness = self._analyze_informativeness_semantic(text)
                semantic_quality = self._compute_semantic_quality(text)
                scores = self._compute_helpfulness_score(text, informativeness, semantic_quality)
                
                # Overall quality rating (1-5 scale)
                overall_score = scores['helpfulness_score']
                quality_rating = max(1, min(5, int(overall_score * 4) + 1))
                
                results.append({
                    'text_index': i,
                    'helpfulness_score': scores['helpfulness_score'],
                    'usefulness_score': scores['usefulness_score'],
                    'informativeness_score': scores['informativeness_score'],
                    'semantic_quality_score': semantic_quality,
                    'quality_rating': quality_rating,
                    'word_count': informativeness['word_count'],
                    'detail_score': informativeness['detail_score'],
                    'matched_templates': informativeness.get('matched_templates', []),
                    'high_confidence_matches': informativeness.get('high_confidence_matches', 0),
                    'text_preview': text[:100] + "..." if len(text) > 100 else text
                })
            
            avg_quality = sum(r['quality_rating'] for r in results) / len(results)
            self._log_execution_end(
                f"Average quality rating: {avg_quality:.2f}/5"
            )
            
            return results
            
        except Exception as e:
            self._handle_error(e, f"num_texts={len(input_data)}")
            raise
