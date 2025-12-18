from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import logging
import uuid
from pipeline.orchestrator import PipelineOrchestrator
from pipeline.models import ReviewAnalysisInput

app = FastAPI(title="Filtering The Noise", version="1.0.0")

# Initialize pipeline orchestrator
logging.basicConfig(level=logging.INFO)
orchestrator = PipelineOrchestrator()

class ReviewInput(BaseModel):
    review_cleaned: str
    business_description: Optional[str] = None
    category: Optional[List[str]] = None
    quality_rating: Optional[int] = None

class BulkReviewInput(BaseModel):
    reviews: List[ReviewInput]

class ContentTaggingOutput(BaseModel):
    is_advertisement: bool
    ad_confidence: float
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
    matched_templates_count: int
    matched_templates: List[str]
    should_filter: bool
    reason: str
    final_verdict: str

class BulkContentTaggingOutput(BaseModel):
    results: List[ContentTaggingOutput]
    total_processed: int
    total_filtered: int
    processing_summary: dict

def process_single_review_internal(input_data: ReviewInput) -> ContentTaggingOutput:
    """Internal function to process a single review"""
    try:
        # Create pipeline input
        pipeline_input = ReviewAnalysisInput(
            review_cleaned=input_data.review_cleaned,
            business_description=input_data.business_description,
            category=input_data.category[0] if input_data.category else None,
            uuid=str(uuid.uuid4()),
            has_image=False
        )
        
        # Run analysis through pipeline
        result = orchestrator.analyze_single_review(pipeline_input)
        
        # Convert to API response format
        final_verdict = "APPROVE" if result.final_verdict == "KEEP" else "FILTER"
        
        return ContentTaggingOutput(
            is_advertisement=result.is_advertisement,
            ad_confidence=round(result.ad_confidence, 8),
            is_rant=result.is_rant,
            rant_confidence=round(result.rant_confidence, 8),
            is_relevant=result.is_relevant,
            relevancy_score=round(result.relevancy_score, 2),
            quality_rating=result.quality_rating,
            helpfulness_score=round(result.helpfulness_score, 3),
            usefulness_score=round(result.usefulness_score, 3),
            informativeness_score=round(result.informativeness_score, 3),
            semantic_quality_score=round(result.semantic_quality_score, 3),
            detail_score=round(result.detail_score, 3),
            word_count=result.word_count,
            high_confidence_matches=result.high_confidence_matches,
            matched_templates_count=len(result.matched_templates),
            matched_templates=result.matched_templates,
            should_filter=result.should_filter,
            reason=result.reason,
            final_verdict=final_verdict
        )
        
    except Exception as e:
        logging.error(f"Error analyzing review: {e}")
        # Return safe default on error with semantic defaults
        return ContentTaggingOutput(
            is_advertisement=False,
            ad_confidence=0.0,
            is_rant=False,
            rant_confidence=0.0,
            is_relevant=True,
            relevancy_score=0.5,
            quality_rating=input_data.quality_rating,
            helpfulness_score=0.5,
            usefulness_score=0.5,
            informativeness_score=0.5,
            semantic_quality_score=0.5,
            detail_score=0.5,
            word_count=0,
            high_confidence_matches=0,
            matched_templates_count=0,
            matched_templates=[],
            should_filter=False,
            reason=f"Analysis failed: {str(e)}",
            final_verdict="APPROVE"
        )

@app.post("/analyze-review", response_model=ContentTaggingOutput)
async def analyze_review(input_data: ReviewInput):
    """
    Analyze a single review content using the ML pipeline
    """
    return process_single_review_internal(input_data)

@app.post("/analyze-reviews-batch", response_model=BulkContentTaggingOutput)
async def analyze_reviews_batch(bulk_input: BulkReviewInput):
    """
    Analyze multiple reviews in batch using the ML pipeline
    """
    try:
        logging.info(f"Processing batch of {len(bulk_input.reviews)} reviews")
        
        results = []
        filtered_count = 0
        
        # Process each review
        for i, review_input in enumerate(bulk_input.reviews):
            try:
                result = process_single_review_internal(review_input)
                results.append(result)
                
                if result.should_filter:
                    filtered_count += 1
                    
            except Exception as e:
                logging.error(f"Error processing review {i}: {e}")
                # Add error result with semantic defaults
                error_result = ContentTaggingOutput(
                    is_advertisement=False,
                    ad_confidence=0.0,
                    is_rant=False,
                    rant_confidence=0.0,
                    is_relevant=True,
                    relevancy_score=0.5,
                    quality_rating=review_input.quality_rating,
                    helpfulness_score=0.5,
                    usefulness_score=0.5,
                    informativeness_score=0.5,
                    semantic_quality_score=0.5,
                    detail_score=0.5,
                    word_count=0,
                    high_confidence_matches=0,
                    matched_templates_count=0,
                    matched_templates=[],
                    should_filter=False,
                    reason=f"Processing failed: {str(e)}",
                    final_verdict="APPROVE"
                )
                results.append(error_result)
        
        # Calculate summary statistics
        total_processed = len(results)
        approved_count = total_processed - filtered_count
        
        # Categorize filtering reasons
        filtering_summary = {}
        for result in results:
            if result.should_filter:
                reason = result.reason
                filtering_summary[reason] = filtering_summary.get(reason, 0) + 1
        
        processing_summary = {
            "approved": approved_count,
            "filtered": filtered_count,
            "approval_rate": round(approved_count / total_processed * 100, 2) if total_processed > 0 else 0,
            "filtering_reasons": filtering_summary,
            "average_ad_confidence": round(sum(r.ad_confidence for r in results) / total_processed, 4) if total_processed > 0 else 0,
            "average_relevancy_score": round(sum(r.relevancy_score for r in results) / total_processed, 2) if total_processed > 0 else 0,
            "average_informativeness": round(sum(r.informativeness_score for r in results) / total_processed, 3) if total_processed > 0 else 0,
            "average_helpfulness": round(sum(r.helpfulness_score for r in results) / total_processed, 3) if total_processed > 0 else 0,
            "average_semantic_quality": round(sum(r.semantic_quality_score for r in results) / total_processed, 3) if total_processed > 0 else 0,
            "high_confidence_reviews": sum(1 for r in results if r.high_confidence_matches > 0),
        }
        
        logging.info(f"Batch processing completed: {approved_count} approved, {filtered_count} filtered")
        
        return BulkContentTaggingOutput(
            results=results,
            total_processed=total_processed,
            total_filtered=filtered_count,
            processing_summary=processing_summary
        )
        
    except Exception as e:
        logging.error(f"Error in batch processing: {e}")
        # Return error response
        return BulkContentTaggingOutput(
            results=[],
            total_processed=0,
            total_filtered=0,
            processing_summary={"error": str(e)}
        )

@app.get("/")
async def root():
    return {"message": "Filtering The Noise"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
