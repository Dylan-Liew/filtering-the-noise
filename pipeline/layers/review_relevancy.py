from typing import List, Dict, Any
import torch
from .base_layer import BaseLayer


class ReviewRelevancyInput:
    """Input data structure for review relevancy analysis."""
    
    def __init__(self, review_text: str, business_info: Dict[str, Any]):
        self.review_text = review_text
        self.business_info = business_info


class ReviewRelevancyOutput:
    """Output data structure for review relevancy results."""
    
    def __init__(self, is_relevant: bool, relevancy_score: float, analysis: Dict[str, Any]):
        self.is_relevant = is_relevant
        self.relevancy_score = relevancy_score
        self.analysis = analysis


class ReviewRelevancyLayer(BaseLayer):
    """
    Layer for determining if a review is relevant to the business/location being reviewed.
    Uses semantic similarity to compare review content with business information.
    """
    
    def __init__(self, app_state, layer_name=None):
        super().__init__(app_state, layer_name or "ReviewRelevancyLayer")
        self._init_relevancy_criteria()
    
    def _init_relevancy_criteria(self):
        """Initialize relevancy analysis criteria."""
        self.business_aspects = [
            "food", "service", "atmosphere", "location", "price", "quality",
            "staff", "cleanliness", "experience", "value", "menu", "drinks",
            "ambiance", "decor", "parking", "accessibility", "hours", "wait time"
        ]
    
    def _generate_business_context(self, business_info: Dict[str, Any]) -> List[str]:
        """Generate contextual descriptions based on business description and category list."""
        contexts = []
        
        # Use business description if available
        if business_info.get("business_description") and business_info["business_description"].strip():
            contexts.append(f"Review about {business_info['business_description']}")
        
        # Use category information (category is a list)
        if "category" in business_info and business_info["category"]:
            categories = business_info["category"]

            # Generate context for each category
            for category in categories:
                contexts.append(f"Experience at a {category}")
                
                # Add specific contexts based on category type
                category_lower = category.lower()
                
                if "restaurant" in category_lower or "food" in category_lower or "bar" in category_lower:
                    contexts.extend([
                        "Food quality and taste experience",
                        "Restaurant service and dining atmosphere",
                        "Menu options and pricing evaluation"
                    ])
                elif "hotel" in category_lower:
                    contexts.extend([
                        "Hotel stay and accommodation experience",
                        "Room quality and hotel services",
                        "Check-in process and staff interactions"
                    ])
                elif "store" in category_lower and "department" not in category_lower:
                    contexts.extend([
                        "Shopping experience and product quality",
                        "Customer service and store atmosphere", 
                        "Product selection and pricing"
                    ])
                elif "museum" in category_lower:
                    contexts.extend([
                        "Museum visit and exhibit experience",
                        "Educational content and displays",
                        "Facility and visitor services"
                    ])
                elif "department store" in category_lower or "supermarket" in category_lower:
                    contexts.extend([
                        "Shopping experience and product selection",
                        "Store layout and checkout process",
                        "Staff assistance and customer service"
                    ])
                elif "dealer" in category_lower:
                    contexts.extend([
                        "Vehicle purchase or service experience",
                        "Sales staff and customer service",
                        "Dealership facilities and process"
                    ])
        
        return contexts
    
    def _compute_relevancy_score(self, review_text: str, business_contexts: List[str]) -> Dict[str, float]:
        """Compute relevancy scores using semantic similarity."""
        model = self.app_state.sentence_transformer
        
        review_embedding = model.encode([review_text])
        context_embeddings = model.encode(business_contexts)
        
        similarities = model.similarity(review_embedding, context_embeddings)
        max_similarity = float(similarities.max())
        avg_similarity = float(similarities.mean())
        
        business_aspect_similarities = []
        for aspect in self.business_aspects:
            aspect_context = f"Discussion about {aspect} at this business"
            aspect_embedding = model.encode([aspect_context])
            aspect_sim = model.similarity(review_embedding, aspect_embedding)
            business_aspect_similarities.append(float(aspect_sim))
        
        business_aspect_score = max(business_aspect_similarities)
        
        return {
            "max_business_similarity": max_similarity,
            "avg_business_similarity": avg_similarity,
            "business_aspect_score": business_aspect_score,
            "business_aspect_similarities": business_aspect_similarities
        }
    
    
    def run(self, input_data: ReviewRelevancyInput) -> ReviewRelevancyOutput:
        """
        Analyze review relevancy to the business.
        
        Args:
            input_data: ReviewRelevancyInput containing review text and business info
            
        Returns:
            ReviewRelevancyOutput with relevancy determination and analysis
        """
        try:
            self._log_execution_start(
                f"Analyzing relevancy for review of {input_data.business_info.get('business_description', 'unknown business')}"
            )
            
            business_contexts = self._generate_business_context(input_data.business_info)
            similarity_scores = self._compute_relevancy_score(input_data.review_text, business_contexts)
            
            relevancy_score = (
                similarity_scores["max_business_similarity"] * 0.6 +
                similarity_scores["business_aspect_score"] * 0.4
            )
            
            relevancy_score = max(0.0, min(1.0, relevancy_score))
            
            is_relevant = (
                relevancy_score > 0.5 and
                similarity_scores["max_business_similarity"] > 0.3
            )
            
            analysis = {
                "similarity_scores": similarity_scores,
                "business_contexts_used": len(business_contexts),
                "primary_factors": {
                    "business_similarity": similarity_scores["max_business_similarity"],
                    "aspect_relevance": similarity_scores["business_aspect_score"]
                }
            }
            
            output = ReviewRelevancyOutput(is_relevant, relevancy_score, analysis)
            
            self._log_execution_end(
                f"Relevancy: {'RELEVANT' if is_relevant else 'IRRELEVANT'} "
                f"(score: {relevancy_score:.3f})"
            )
            
            return output
            
        except Exception as e:
            self._handle_error(e, f"business={input_data.business_info.get('business_description', 'unknown')}")
            raise
