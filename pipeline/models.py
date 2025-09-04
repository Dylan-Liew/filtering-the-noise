from typing import Optional, List, Union

from pydantic import BaseModel


class ReviewAnalysisInput(BaseModel):
    """Input data structure for single review analysis.
    
    Fields match the dataset structure:
    - review_cleaned: The cleaned review text content
    - uuid: Unique review identifier 
    - rating: Review rating score (1-5)
    - has_image: Whether the review has associated images
    - images: List of image URLs (if any)
    - business_description: Business description text
    - category: Business category (can be string or list)
    """
    review_cleaned: str
    uuid: Optional[str] = None
    rating: Optional[int] = None
    has_image: Optional[bool] = False
    images: Optional[List[str]] = None
    business_description: Optional[str] = None
    category: Optional[Union[str, List[str]]] = None


class ReviewAnalysisOutput(BaseModel):
    """Output data structure for single review analysis results."""
    uuid: Optional[str]
    is_advertisement: bool
    ad_confidence: float
    is_spam: bool
    spam_probability: float
    is_rant: bool
    rant_confidence: float
    is_relevant: bool
    relevancy_score: float
    quality_rating: int
    helpfulness_score: float
    usefulness_score: float
    informativeness_score: float
    semantic_quality_score: float
    detail_score: float
    word_count: int
    high_confidence_matches: int
    matched_templates: List[str]
    should_filter: bool
    reason: str
    final_verdict: str