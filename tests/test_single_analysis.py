#!/usr/bin/env python3

"""
Test script for single review analysis functionality
"""

import logging
import sys
import os
# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pipeline.orchestrator import PipelineOrchestrator
from pipeline.models import ReviewAnalysisInput

# Set up logging
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

def print_report(title, review_text, result):
    print("\n" + "="*80)
    print(f" TEST CASE: {title}")
    print("="*80)
    print(f"Review Preview: \"{review_text[:100]}...\"")
    print("-" * 80)
    
    # Classification
    print(f"{'METRIC':<25} | {'VALUE':<15} | {'CONFIDENCE/SCORE':<15}")
    print("-" * 60)
    print(f"{'Advertisement':<25} | {str(result.is_advertisement):<15} | {result.ad_confidence:.3f}")
    print(f"{'Spam':<25} | {str(result.is_spam):<15} | {result.spam_probability:.3f}")
    print(f"{'Rant':<25} | {str(result.is_rant):<15} | {result.rant_confidence:.3f}")
    print(f"{'Relevant':<25} | {str(result.is_relevant):<15} | {result.relevancy_score:.3f}")
    print("-" * 60)
    
    # Quality
    print(f"Quality Rating:      {result.quality_rating}/5")
    print(f"Helpfulness Score:   {result.helpfulness_score:.3f}")
    print(f"Informativeness:     {result.informativeness_score:.3f}")
    
    print("-" * 80)
    # Verdict
    verdict_color = "🔴" if result.should_filter else "🟢"
    print(f"FINAL VERDICT: {verdict_color} {result.final_verdict}")
    if result.should_filter:
        print(f"REASON:        {result.reason}")
    print("="*80 + "\n")

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
        print_report("Normal Review (Should Review/Keep)", test_review_1.review_cleaned, result_1)
        
        # Test case 2: Potential advertisement
        test_review_2 = ReviewAnalysisInput(
            review_cleaned="Visit our amazing store for the best deals! Call now 555-1234 or check out our website www.bestdeals.com for exclusive offers!",
            business_description="Electronics Store",
            category="Retail",
            uuid="test-review-2"
        )
        
        logger.info("Testing potential advertisement...")
        result_2 = orchestrator.analyze_single_review(test_review_2)
        print_report("Potential Advertisement", test_review_2.review_cleaned, result_2)
        
        # Test case 3: Irrelevant review
        test_review_3 = ReviewAnalysisInput(
            review_cleaned="I love cats and dogs. My favorite color is blue. The weather is nice today.",
            business_description="Pizza Restaurant",
            category="Restaurant",
            uuid="test-review-3"
        )
        
        logger.info("Testing irrelevant review...")
        result_3 = orchestrator.analyze_single_review(test_review_3)
        print_report("Irrelevant Review", test_review_3.review_cleaned, result_3)
        
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