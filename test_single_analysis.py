#!/usr/bin/env python3

"""
Test script for single review analysis functionality
"""

import logging
import sys
from pipeline.orchestrator import PipelineOrchestrator
from pipeline.models import ReviewAnalysisInput

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_single_review_analysis():
    """Test the single review analysis functionality."""
    
    try:
        # Initialize orchestrator
        logger.info("Initializing PipelineOrchestrator...")
        orchestrator = PipelineOrchestrator()
        
        # Test case 1: Normal review
        test_review_1 = ReviewAnalysisInput(
            review_cleaned="Great restaurant! The food was delicious and the service was excellent. Highly recommend the pasta dishes.",
            business_description="Italian Restaurant",
            category="Restaurant",
            uuid="test-review-1"
        )
        
        logger.info("Testing normal review...")
        result_1 = orchestrator.analyze_single_review(test_review_1)
        print("=== Test Case 1: Normal Review ===")
        print(f"Review: {test_review_1.review_cleaned[:50]}...")
        print(f"Is Advertisement: {result_1.is_advertisement}")
        print(f"Is Rant: {result_1.is_rant}")
        print(f"Is Relevant: {result_1.is_relevant}")
        print(f"Quality Rating: {result_1.quality_rating}")
        print(f"Final Verdict: {result_1.final_verdict}")
        print(f"Reason: {result_1.reason}")
        print()
        
        # Test case 2: Potential advertisement
        test_review_2 = ReviewAnalysisInput(
            review_cleaned="Visit our amazing store for the best deals! Call now 555-1234 or check out our website www.bestdeals.com for exclusive offers!",
            business_description="Electronics Store",
            category="Retail",
            uuid="test-review-2"
        )
        
        logger.info("Testing potential advertisement...")
        result_2 = orchestrator.analyze_single_review(test_review_2)
        print("=== Test Case 2: Potential Advertisement ===")
        print(f"Review: {test_review_2.review_cleaned[:50]}...")
        print(f"Is Advertisement: {result_2.is_advertisement}")
        print(f"Ad Confidence: {result_2.ad_confidence}")
        print(f"Final Verdict: {result_2.final_verdict}")
        print(f"Reason: {result_2.reason}")
        print()
        
        # Test case 3: Irrelevant review
        test_review_3 = ReviewAnalysisInput(
            review_cleaned="I love cats and dogs. My favorite color is blue. The weather is nice today.",
            business_description="Pizza Restaurant",
            category="Restaurant",
            uuid="test-review-3"
        )
        
        logger.info("Testing irrelevant review...")
        result_3 = orchestrator.analyze_single_review(test_review_3)
        print("=== Test Case 3: Irrelevant Review ===")
        print(f"Review: {test_review_3.review_cleaned[:50]}...")
        print(f"Is Relevant: {result_3.is_relevant}")
        print(f"Relevancy Score: {result_3.relevancy_score}")
        print(f"Final Verdict: {result_3.final_verdict}")
        print(f"Reason: {result_3.reason}")
        print()
        
        logger.info("All tests completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_single_review_analysis()
    sys.exit(0 if success else 1)