#!/usr/bin/env python3
"""
Test script for verifying running API endpoints.
Requires the API to be running separately (e.g. uvicorn app.main:app --reload).
"""

import requests
import json
import sys
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

API_URL = "http://127.0.0.1:8000"

def check_api_status():
    """Check if API is running."""
    try:
        response = requests.get(f"{API_URL}/")
        if response.status_code == 200:
            logger.info("✅ API is responding")
            return True
        else:
            logger.error(f"❌ API responded with status code: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        logger.error("❌ Could not connect to API. Is it running? (uvicorn app.main:app --reload)")
        return False

def print_result_report(title, review_text, result):
    """Print a standardized report table, similar to test_single_analysis.py."""
    print("\n" + "="*80)
    print(f" TEST CASE: {title}")
    print("="*80)
    print(f"Review Preview: \"{review_text[:100]}...\"")
    print("-" * 80)
    
    # Classification
    print(f"{'METRIC':<25} | {'VALUE':<15} | {'CONFIDENCE/SCORE':<15}")
    print("-" * 60)
    print(f"{'Advertisement':<25} | {str(result.get('is_advertisement', False)):<15} | {result.get('ad_confidence', 0.0):.3f}")
    print(f"{'Spam':<25} | {str(result.get('is_spam', False)):<15} | {result.get('spam_probability', 0.0):.3f}")
    print(f"{'Rant':<25} | {str(result.get('is_rant', False)):<15} | {result.get('rant_confidence', 0.0):.3f}")
    print(f"{'Relevant':<25} | {str(result.get('is_relevant', False)):<15} | {result.get('relevancy_score', 0.0):.3f}")
    print("-" * 60)
    
    # Quality
    print(f"Quality Rating:      {result.get('quality_rating', 0)}/5")
    print(f"Helpfulness Score:   {result.get('helpfulness_score', 0.0):.3f}")
    print(f"Informativeness:     {result.get('informativeness_score', 0.0):.3f}")
    
    print("-" * 80)
    # Verdict
    should_filter = result.get('should_filter', False)
    verdict_color = "🔴" if should_filter else "🟢"
    print(f"FINAL VERDICT: {verdict_color} {result.get('final_verdict', 'UNKNOWN')}")
    if should_filter:
        print(f"REASON:        {result.get('reason', '')}")
    print("="*80 + "\n")

def test_api_endpoints():
    """Test API endpoints with sample reviews."""
    
    if not check_api_status():
        return False

    endpoint = f"{API_URL}/analyze-review"
    
    # Test case 1: Normal review
    review_1 = {
        "review_cleaned": "Great restaurant! The food was delicious and the service was excellent. Highly recommend the pasta dishes.",
        "business_description": "Italian Restaurant",
        "category": ["Restaurant"],
        "uuid": "api-test-1"
    }
    
    logger.info("Testing normal review...")
    try:
        response = requests.post(endpoint, json=review_1)
        if response.status_code == 200:
            print_result_report("Normal Review (Should Keep)", review_1['review_cleaned'], response.json())
        else:
            logger.error(f"Failed to analyze review 1. Status: {response.status_code}, Detail: {response.text}")
    except Exception as e:
        logger.error(f"Error testing review 1: {e}")

    # Test case 2: Potential advertisement
    review_2 = {
        "review_cleaned": "Visit our amazing store for the best deals! Call now 555-1234 or check out our website www.bestdeals.com for exclusive offers!",
        "business_description": "Electronics Store",
        "category": ["Retail"],
        "uuid": "api-test-2"
    }
    
    logger.info("Testing potential advertisement...")
    try:
        response = requests.post(endpoint, json=review_2)
        if response.status_code == 200:
            print_result_report("Potential Advertisement", review_2['review_cleaned'], response.json())
        else:
            logger.error(f"Failed to analyze review 2. Status: {response.status_code}, Detail: {response.text}")
    except Exception as e:
        logger.error(f"Error testing review 2: {e}")

    # Test case 3: Irrelevant review
    review_3 = {
        "review_cleaned": "I love cats and dogs. My favorite color is blue. The weather is nice today.",
        "business_description": "Pizza Restaurant",
        "category": ["Restaurant"],
        "uuid": "api-test-3"
    }
    
    logger.info("Testing irrelevant review...")
    try:
        response = requests.post(endpoint, json=review_3)
        if response.status_code == 200:
            print_result_report("Irrelevant Review", review_3['review_cleaned'], response.json())
        else:
            logger.error(f"Failed to analyze review 3. Status: {response.status_code}, Detail: {response.text}")
    except Exception as e:
        logger.error(f"Error testing review 3: {e}")

    return True

if __name__ == "__main__":
    success = test_api_endpoints()
    sys.exit(0 if success else 1)
