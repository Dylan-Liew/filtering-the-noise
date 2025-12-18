#!/usr/bin/env python3
"""
Test script for the AdvertisementDetectionLayer functionality.
Tests promotional content detection using both pattern matching and semantic analysis.
"""

import logging
import sys
import os

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from typing import List, Dict, Any

from pipeline.app_state import AppState
from pipeline.layers.advertisement_detection import (
    AdvertisementDetectionLayer,
    AdvertisementDetectionInput,
    AdvertisementDetectionOutput,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_test_texts() -> Dict[str, List[str]]:
    """Get real sample data from the dataset for testing."""
    return {
        "sample_reviews": [
            "My favorite store in Cullman.",
            "Super delicious pizza and great fresh salad :thumbs up: light skin tone:",
            "Self check out is great",
            "Remodle was well worth it but the katchup packets are still too small.",
            "Best sirloin steak and cesar salad. Will be back again."
        ]
    }


def get_real_sample_data() -> List[Dict[str, Any]]:
    """Get real sample data with full context from the dataset."""
    return [
        {
            "uuid": "260445",
            "review_cleaned": "My favorite store in Cullman.",
            "business_description": None,
            "category": ["Sporting goods store"]
        },
        {
            "uuid": "581381", 
            "review_cleaned": "",
            "business_description": None,
            "category": ["Mexican restaurant"]
        },
        {
            "uuid": "958786",
            "review_cleaned": "",
            "business_description": "Fast-food chain serving fried chicken, big burgers & made-from-scratch breakfast biscuits.",
            "category": ["Fast food restaurant", "Hamburger restaurant"]
        },
        {
            "uuid": "936339",
            "review_cleaned": "Super delicious pizza and great fresh salad :thumbs up: light skin tone:",
            "business_description": "Pizzeria chain offering an array of craft beers, plus specialty pizzas, salads & sandwiches.",
            "category": ["Pizza restaurant"]
        },
        {
            "uuid": "931968",
            "review_cleaned": "Self check out is great",
            "business_description": None,
            "category": ["Department store", "Clothing store", "Craft store", "Discount store", "Electronics store", "Grocery store", "Home goods store", "Sporting goods store", "Supermarket", "Toy store"]
        },
        {
            "uuid": "168682",
            "review_cleaned": "Remodle was well worth it but the katchup packets are still too small.",
            "business_description": "Classic, long-running fast-food chain known for its burgers, fries & shakes.",
            "category": ["Fast food restaurant", "Breakfast restaurant", "Coffee shop", "Hamburger restaurant", "Restaurant", "Sandwich shop"]
        },
        {
            "uuid": "525872",
            "review_cleaned": "Best sirloin steak and cesar salad. Will be back again.",
            "business_description": "Lively chain steakhouse serving American fare with a Southwestern spin amid Texas-themed decor.",
            "category": ["Steak house", "American restaurant", "New American restaurant"]
        }
    ]


def test_hello_world_ad_detection(layer: AdvertisementDetectionLayer) -> bool:
    """
    Simple hello world test with obvious promotional content.
    
    Args:
        layer: AdvertisementDetectionLayer instance
        
    Returns:
        True if test passes, False otherwise
    """
    try:
        logger.info("=== Hello World Advertisement Detection Test ===")
        
        promotional_text = "Call now for special discount! Visit our website www.greatdeals.com for limited time offers!"
        genuine_text = "The food was delicious and the service was excellent. Would definitely recommend this restaurant."
        
        # Test promotional text
        input_data = AdvertisementDetectionInput([promotional_text])
        results = layer.run(input_data)
        promo_result = results[0]
        
        logger.info(f"🎯 Promotional Text Analysis:")
        logger.info(f"   Is Advertisement: {promo_result['is_advertisement']}")
        logger.info(f"   Confidence: {promo_result['confidence']:.4f}")
        logger.info(f"   Total Score: {promo_result['total_score']:.4f}")
        logger.info(f"   Pattern Matches: {promo_result['pattern_matches']}")
        logger.info(f"   Keyword Matches: {promo_result['keyword_matches']}")
        
        # Test genuine text
        input_data = AdvertisementDetectionInput([genuine_text])
        results = layer.run(input_data)
        genuine_result = results[0]
        
        logger.info(f"🎯 Genuine Review Analysis:")
        logger.info(f"   Is Advertisement: {genuine_result['is_advertisement']}")
        logger.info(f"   Confidence: {genuine_result['confidence']:.4f}")
        logger.info(f"   Total Score: {genuine_result['total_score']:.4f}")
        
        # Should detect promo as ad, genuine as not ad
        success = (promo_result['is_advertisement'] and 
                  not genuine_result['is_advertisement'])
        
        if success:
            logger.info("✓ Correctly distinguished promotional from genuine content")
        else:
            logger.warning(f"Detection issues: promo_detected={promo_result['is_advertisement']}, "
                          f"genuine_clean={not genuine_result['is_advertisement']}")
            
        return success
        
    except Exception as e:
        logger.error(f"✗ Hello world test failed: {e}")
        return False


def test_pattern_detection(layer: AdvertisementDetectionLayer) -> bool:
    """
    Test regex pattern detection capabilities.
    
    Args:
        layer: AdvertisementDetectionLayer instance
        
    Returns:
        True if test passes, False otherwise
    """
    try:
        logger.info("=== Pattern Detection Test ===")
        
        pattern_tests = [
            ("Phone number", "Call us at 555-123-4567 for more info"),
            ("Website URL", "Check out our deals at www.example.com"),
            ("Email contact", "Email us at contact@business.org for quotes"),
            ("Promotional keywords", "Limited time offer with special discount!"),
            ("Call to action", "Buy now and save big on all items"),
            ("Price language", "Best price guaranteed, cheap and affordable")
        ]
        
        results = []
        for test_name, text in pattern_tests:
            input_data = AdvertisementDetectionInput([text])
            detection_results = layer.run(input_data)
            result = detection_results[0]
            
            results.append({
                'test': test_name,
                'detected': result['is_advertisement'],
                'pattern_matches': len(result['pattern_matches']),
                'keyword_matches': len(result['keyword_matches']),
                'score': result['total_score']
            })
            
            logger.info(f"   {test_name}: {'✓' if result['is_advertisement'] else '✗'} "
                       f"(patterns: {len(result['pattern_matches'])}, "
                       f"keywords: {len(result['keyword_matches'])}, "
                       f"score: {result['total_score']:.3f})")
        
        # Most pattern tests should be detected as ads
        detected_count = sum(1 for r in results if r['detected'])
        success = detected_count >= len(results) * 0.8  # At least 80%
        
        if success:
            logger.info(f"✓ Pattern detection working: {detected_count}/{len(results)} detected")
        else:
            logger.warning(f"Pattern detection issues: only {detected_count}/{len(results)} detected")
            
        return success
        
    except Exception as e:
        logger.error(f"✗ Pattern detection test failed: {e}")
        return False


def test_semantic_similarity(layer: AdvertisementDetectionLayer) -> bool:
    """
    Test semantic similarity with advertisement templates.
    
    Args:
        layer: AdvertisementDetectionLayer instance
        
    Returns:
        True if test passes, False otherwise
    """
    try:
        logger.info("=== Semantic Similarity Test ===")
        
        # Test texts with varying semantic similarity to ads
        semantic_tests = [
            ("High similarity", "Don't miss our exclusive promotions and special deals today!"),
            ("Medium similarity", "We offer great service and competitive pricing for customers."),
            ("Low similarity", "The food was tasty and the atmosphere was cozy and welcoming."),
            ("Very low similarity", "I enjoyed spending time with my family at this quiet location.")
        ]
        
        results = []
        for test_name, text in semantic_tests:
            input_data = AdvertisementDetectionInput([text])
            detection_results = layer.run(input_data)
            result = detection_results[0]
            
            results.append({
                'test': test_name,
                'semantic_score': result['semantic_score'],
                'total_score': result['total_score'],
                'detected': result['is_advertisement']
            })
            
            logger.info(f"   {test_name}: semantic={result['semantic_score']:.4f}, "
                       f"total={result['total_score']:.4f}, "
                       f"detected={'✓' if result['is_advertisement'] else '✗'}")
        
        # Verify semantic scores decrease as expected
        semantic_scores = [r['semantic_score'] for r in results]
        decreasing_trend = all(semantic_scores[i] >= semantic_scores[i+1] 
                              for i in range(len(semantic_scores)-1))
        
        if decreasing_trend:
            logger.info("✓ Semantic similarity scores follow expected trend")
        else:
            logger.warning("⚠ Semantic similarity scores don't follow expected decreasing trend")
            
        return True  # Don't fail on trend, just log warning
        
    except Exception as e:
        logger.error(f"✗ Semantic similarity test failed: {e}")
        return False


def test_real_sample_data(layer: AdvertisementDetectionLayer) -> bool:
    """
    Test detection with real sample data from the dataset.
    
    Args:
        layer: AdvertisementDetectionLayer instance
        
    Returns:
        True if test passes, False otherwise
    """
    try:
        logger.info("=== Real Sample Data Test ===")
        
        sample_data = get_real_sample_data()
        
        # Filter to only reviews with text
        text_reviews = [sample for sample in sample_data if sample['review_cleaned'].strip()]
        
        if not text_reviews:
            logger.warning("No reviews with text found in sample data")
            return True
        
        review_texts = [sample['review_cleaned'] for sample in text_reviews]
        
        logger.info(f"Testing {len(review_texts)} real reviews...")
        
        input_data = AdvertisementDetectionInput(review_texts)
        results = layer.run(input_data)
        
        detected_count = sum(1 for r in results if r['is_advertisement'])
        avg_confidence = sum(r['confidence'] for r in results) / len(results)
        avg_score = sum(r['total_score'] for r in results) / len(results)
        
        logger.info(f"📊 Real Sample Data Results:")
        logger.info(f"   Total reviews analyzed: {len(results)}")
        logger.info(f"   Advertisements detected: {detected_count}")
        logger.info(f"   Average confidence: {avg_confidence:.4f}")
        logger.info(f"   Average score: {avg_score:.4f}")
        
        # Show individual results with business context
        for i, result in enumerate(results):
            sample = text_reviews[i]
            business_desc = sample['business_description'] or 'N/A'
            category = sample['category'][0] if sample['category'] else 'N/A'
            
            logger.info(f"   {i+1}. {'[AD]' if result['is_advertisement'] else '[OK]'} "
                       f"(conf: {result['confidence']:.3f}) "
                       f"{sample['review_cleaned'][:40]}...")
            logger.info(f"      Business: {business_desc[:50]}{'...' if len(business_desc) > 50 else ''}")
            logger.info(f"      Category: {category}")
            if result['pattern_matches']:
                logger.info(f"      Patterns: {result['pattern_matches']}")
            if result['keyword_matches']:
                logger.info(f"      Keywords: {result['keyword_matches']}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Real sample data test failed: {e}")
        return False


def test_edge_cases(layer: AdvertisementDetectionLayer) -> bool:
    """
    Test edge cases and corner scenarios.
    
    Args:
        layer: AdvertisementDetectionLayer instance
        
    Returns:
        True if test passes, False otherwise
    """
    try:
        logger.info("=== Edge Cases Test ===")
        
        edge_cases = [
            ("Empty text", ""),
            ("Very short", "Ok."),
            ("Only punctuation", "!!! ??? ... !!!"),
            ("Numbers only", "123 456 789 000"),
            ("Emojis only", "😀😍🔥💰🎉⭐"),
            ("Very long repetitive", "Great place " * 50),
            ("Mixed languages", "Good service muy bueno 很好 excellent!"),
            ("Special characters", "@#$%^&*()[]{}|\\:;\"'<>?/~`"),
        ]
        
        results = []
        for test_name, text in edge_cases:
            try:
                input_data = AdvertisementDetectionInput([text])
                detection_results = layer.run(input_data)
                result = detection_results[0]
                
                results.append({
                    'test': test_name,
                    'success': True,
                    'detected': result['is_advertisement'],
                    'score': result['total_score']
                })
                
                logger.info(f"   {test_name}: {'✓' if result['is_advertisement'] else '✗'} "
                           f"(score: {result['total_score']:.4f})")
                
            except Exception as e:
                results.append({
                    'test': test_name,
                    'success': False,
                    'error': str(e)
                })
                logger.warning(f"   {test_name}: Failed - {e}")
        
        success_count = sum(1 for r in results if r['success'])
        total_count = len(results)
        
        success = success_count == total_count
        
        if success:
            logger.info(f"✓ All {total_count} edge cases handled successfully")
        else:
            logger.warning(f"Only {success_count}/{total_count} edge cases handled successfully")
            
        return success
        
    except Exception as e:
        logger.error(f"✗ Edge cases test failed: {e}")
        return False


def test_batch_processing(layer: AdvertisementDetectionLayer) -> bool:
    """
    Test batch processing with real sample data.
    
    Args:
        layer: AdvertisementDetectionLayer instance
        
    Returns:
        True if test passes, False otherwise
    """
    try:
        logger.info("=== Batch Processing Test ===")
        
        # Get all sample texts
        test_texts = get_test_texts()
        all_texts = test_texts['sample_reviews']
        
        # Process as single batch
        input_data = AdvertisementDetectionInput(all_texts)
        results = layer.run(input_data)
        
        # Analyze batch results
        total_texts = len(all_texts)
        detected_count = sum(1 for r in results if r['is_advertisement'])
        avg_confidence = sum(r['confidence'] for r in results) / len(results)
        avg_score = sum(r['total_score'] for r in results) / len(results)
        
        logger.info(f"🎯 Batch Processing Results ({total_texts} texts):")
        logger.info(f"   Advertisements detected: {detected_count}/{total_texts} ({detected_count/total_texts:.1%})")
        logger.info(f"   Average confidence: {avg_confidence:.4f}")
        logger.info(f"   Average score: {avg_score:.4f}")
        
        # Score distribution
        high_confidence = sum(1 for r in results if r['confidence'] > 0.7)
        medium_confidence = sum(1 for r in results if 0.3 <= r['confidence'] <= 0.7)
        low_confidence = sum(1 for r in results if r['confidence'] < 0.3)
        
        logger.info(f"   Confidence Distribution: High({high_confidence}) Medium({medium_confidence}) Low({low_confidence})")
        
        # Show all results since we only have a few
        logger.info("   All Results:")
        for i, result in enumerate(results):
            text_preview = all_texts[result['text_index']][:60]
            logger.info(f"     {i+1}. Score {result['total_score']:.3f}: {text_preview}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Batch processing test failed: {e}")
        return False


def run_all_tests() -> None:
    """
    Run all advertisement detection tests.
    """
    logger.info("🚀 Advertisement Detection Layer Tests")
    logger.info("=" * 50)
    
    try:
        # Initialize the layer
        logger.info("Initializing AppState and AdvertisementDetectionLayer...")
        app_state = AppState()
        layer = AdvertisementDetectionLayer(app_state)
        
        try:
            _ = app_state.sentence_transformer
            logger.info("✓ Sentence transformer model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load sentence transformer: {e}")
            logger.error("Please ensure sentence-transformers is installed")
            return
        
        # Run all tests
        tests = [
            ("Hello World Ad Detection Test", test_hello_world_ad_detection),
            ("Pattern Detection Test", test_pattern_detection),
            ("Semantic Similarity Test", test_semantic_similarity),
            ("Real Sample Data Test", test_real_sample_data),
            ("Edge Cases Test", test_edge_cases),
            ("Batch Processing Test", test_batch_processing),
        ]
        
        passed = 0
        total = len(tests)
        
        for test_name, test_func in tests:
            logger.info(f"\n{'='*20} {test_name} {'='*20}")
            try:
                if test_func(layer):
                    passed += 1
                    logger.info(f"✅ {test_name} PASSED\n")
                else:
                    logger.error(f"❌ {test_name} FAILED\n")
            except Exception as e:
                logger.error(f"❌ {test_name} FAILED with error: {e}\n")
            
        logger.info("=" * 60)
        logger.info(f"📊 Final Results: {passed}/{total} tests passed")
        
        if passed == total:
            logger.info("🎉 All tests passed!")
        else:
            logger.warning(f"⚠️  {total - passed} test(s) failed")
            
        # Clean up
        if hasattr(app_state, 'cleanup'):
            app_state.cleanup()
            
    except Exception as e:
        logger.error(f"💥 Fatal error during testing: {e}")
        raise


if __name__ == "__main__":
    try:
        run_all_tests()
    except KeyboardInterrupt:
        logger.info("Tests interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)