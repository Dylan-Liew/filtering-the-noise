import logging
import sys
import os

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from typing import List, Dict, Any

from pipeline.app_state import AppState
from pipeline.layers.quality_scoring import QualityScoringLayer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_real_sample_data() -> List[Dict[str, Any]]:
    """Get real sample data with various quality levels for testing."""
    return [
        {
            "uuid": "high_quality_1",
            "review_cleaned": "I visited this restaurant last Tuesday evening around 7pm. The staff was incredibly friendly and attentive throughout our meal. The atmosphere was cozy with dim lighting and soft jazz music. We ordered the ribeye steak which was cooked perfectly to medium-rare as requested. The parking was convenient with plenty of spaces available. The wait time was about 15 minutes which was reasonable for a busy night. I would definitely recommend this place to anyone looking for a quality dining experience. The prices are fair for the portion sizes - around $25 per entree."
        },
        {
            "uuid": "medium_quality_1", 
            "review_cleaned": "Good food and service. The restaurant was clean and the staff was helpful. We had the pasta which was tasty. Would come back again."
        },
        {
            "uuid": "low_quality_1",
            "review_cleaned": "Great place!"
        },
        {
            "uuid": "high_quality_2",
            "review_cleaned": "Went to this coffee shop on Saturday morning around 10am. They have excellent WiFi which is perfect for working remotely. The baristas are knowledgeable about their coffee selections and helped me choose a perfect medium roast. The seating area has comfortable chairs and tables with power outlets. Parking can be challenging during weekends, so I'd recommend arriving early. They offer oat milk and almond milk alternatives. The croissants are freshly baked and pair well with their house blend. Prices are reasonable - $4.50 for a large latte. The atmosphere is quiet enough for concentration but has a nice buzz of activity."
        },
        {
            "uuid": "medium_quality_2",
            "review_cleaned": "Nice place to work from. Good coffee and WiFi. Staff is friendly. Gets busy on weekends."
        },
        {
            "uuid": "low_quality_2",
            "review_cleaned": "Bad experience. Won't return."
        },
        {
            "uuid": "detailed_experience_1",
            "review_cleaned": "Had an appointment at this dental office yesterday. The reception area was clean and modern with comfortable seating. Check-in was quick and efficient. Dr. Smith was thorough in explaining the procedure and answered all my questions patiently. The dental hygienist was gentle during the cleaning. They use modern equipment and the office feels very professional. The appointment ran on time - scheduled for 2pm and was seen promptly. Parking is free in the adjacent lot. They accept most insurance plans and the billing was transparent with no surprise charges. I appreciate that they send reminder texts the day before appointments."
        },
        {
            "uuid": "promotional_content_1",
            "review_cleaned": "Amazing deals every day! Visit our store for the best prices in town. Call 555-0123 for more information. Don't miss out on our special offers!"
        }
    ]


def test_quality_scoring_analysis():
    """Test quality scoring analysis with real sample data."""
    logger.info("🧪 Testing Quality Scoring Analysis")
    logger.info("=" * 60)
    
    try:
        # Initialize the layer
        app_state = AppState()
        layer = QualityScoringLayer(app_state)
        
        sample_data = get_real_sample_data()
        
        # Filter to only reviews with text
        text_reviews = [sample for sample in sample_data if sample['review_cleaned'].strip()]
        
        if not text_reviews:
            logger.warning("No reviews with text found in sample data")
            return
        
        logger.info(f"Testing {len(text_reviews)} reviews for quality scoring...")
        
        results_summary = []
        
        for i, sample in enumerate(text_reviews):
            logger.info(f"\n{'='*50} Sample {i+1} {'='*50}")
            
            review_text = sample['review_cleaned']
            word_count = len(review_text.split())
            
            logger.info(f"📝 Review ({sample['uuid']}): '{review_text[:100]}{'...' if len(review_text) > 100 else ''}'")
            logger.info(f"📊 Word count: {word_count}")
            
            # Run quality scoring
            results = layer.run([review_text])
            result = results[0] if results else {}
            
            # Extract key metrics
            informativeness_score = result.get('informativeness_score', 0.0)
            helpfulness_score = result.get('helpfulness_score', 0.0)
            usefulness_score = result.get('usefulness_score', 0.0)
            semantic_quality_score = result.get('semantic_quality_score', 0.0)
            detail_score = result.get('detail_score', 0.0)
            quality_rating = result.get('quality_rating', 1)
            high_confidence_matches = result.get('high_confidence_matches', 0)
            matched_templates = result.get('matched_templates', [])
            
            logger.info(f"\n📈 Quality Metrics:")
            logger.info(f"   Quality Rating: {quality_rating}/5")
            logger.info(f"   Informativeness Score: {informativeness_score:.3f}")
            logger.info(f"   Helpfulness Score: {helpfulness_score:.3f}")
            logger.info(f"   Usefulness Score: {usefulness_score:.3f}")
            logger.info(f"   Semantic Quality Score: {semantic_quality_score:.3f}")
            logger.info(f"   Detail Score: {detail_score:.3f}")
            logger.info(f"   High Confidence Matches: {high_confidence_matches}")
            
            logger.info(f"\n🎯 Matched Templates ({len(matched_templates)}):")
            for j, template in enumerate(matched_templates[:5]):  # Show first 5
                logger.info(f"   {j+1}: {template}")
            if len(matched_templates) > 5:
                logger.info(f"   ... and {len(matched_templates) - 5} more")
            
            # Determine quality classification
            if quality_rating >= 4:
                quality_class = "🟢 HIGH QUALITY"
            elif quality_rating >= 3:
                quality_class = "🟡 MEDIUM QUALITY"
            else:
                quality_class = "🔴 LOW QUALITY"
            
            logger.info(f"\n✅ Final Assessment: {quality_class}")
            
            results_summary.append({
                'uuid': sample['uuid'],
                'word_count': word_count,
                'quality_rating': quality_rating,
                'informativeness_score': informativeness_score,
                'helpfulness_score': helpfulness_score,
                'high_confidence_matches': high_confidence_matches,
                'matched_templates_count': len(matched_templates),
                'quality_class': quality_class
            })
        
        # Summary analysis
        logger.info(f"\n{'='*60}")
        logger.info("📊 SUMMARY ANALYSIS")
        logger.info("=" * 60)
        
        total_reviews = len(results_summary)
        high_quality = sum(1 for r in results_summary if r['quality_rating'] >= 4)
        medium_quality = sum(1 for r in results_summary if 2 <= r['quality_rating'] < 4)
        low_quality = sum(1 for r in results_summary if r['quality_rating'] < 2)
        
        avg_informativeness = sum(r['informativeness_score'] for r in results_summary) / total_reviews
        avg_helpfulness = sum(r['helpfulness_score'] for r in results_summary) / total_reviews
        avg_quality_rating = sum(r['quality_rating'] for r in results_summary) / total_reviews
        total_high_conf_matches = sum(r['high_confidence_matches'] for r in results_summary)
        
        logger.info(f"Total reviews processed: {total_reviews}")
        logger.info(f"Quality distribution:")
        logger.info(f"  🟢 High quality (4-5): {high_quality} ({high_quality/total_reviews*100:.1f}%)")
        logger.info(f"  🟡 Medium quality (2-3): {medium_quality} ({medium_quality/total_reviews*100:.1f}%)")
        logger.info(f"  🔴 Low quality (1): {low_quality} ({low_quality/total_reviews*100:.1f}%)")
        logger.info(f"")
        logger.info(f"Average metrics:")
        logger.info(f"  Quality rating: {avg_quality_rating:.2f}/5")
        logger.info(f"  Informativeness: {avg_informativeness:.3f}")
        logger.info(f"  Helpfulness: {avg_helpfulness:.3f}")
        logger.info(f"  Total high-confidence matches: {total_high_conf_matches}")
        
        logger.info(f"\n📋 Individual Results:")
        for result in results_summary:
            logger.info(f"  {result['uuid']}: {result['quality_class']} "
                       f"(Rating: {result['quality_rating']}/5, "
                       f"Info: {result['informativeness_score']:.3f}, "
                       f"Matches: {result['high_confidence_matches']})")
        
        logger.info("\n" + "="*60)
        logger.info("✅ Quality scoring analysis completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Quality scoring analysis failed: {e}")
        raise


def test_batch_processing():
    """Test batch processing functionality."""
    logger.info("🔄 Testing Batch Processing")
    logger.info("=" * 60)
    
    try:
        app_state = AppState()
        layer = QualityScoringLayer(app_state)
        
        sample_data = get_real_sample_data()
        
        # Extract all review texts
        review_texts = [sample['review_cleaned'] for sample in sample_data if sample['review_cleaned'].strip()]
        
        logger.info(f"Processing {len(review_texts)} reviews in batch...")
        
        # Process all reviews in a single batch
        batch_results = layer.run(review_texts)
        
        logger.info(f"✅ Batch processing completed. Got {len(batch_results)} results.")
        
        # Verify results consistency
        for i, result in enumerate(batch_results):
            logger.info(f"Result {i+1}: Quality {result['quality_rating']}/5, "
                       f"Templates: {len(result.get('matched_templates', []))}, "
                       f"High-conf: {result.get('high_confidence_matches', 0)}")
        
        logger.info("✅ Batch processing test completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Batch processing test failed: {e}")
        raise


def test_template_matching():
    """Test specific template matching scenarios."""
    logger.info("🎯 Testing Template Matching Scenarios")
    logger.info("=" * 60)
    
    try:
        app_state = AppState()
        layer = QualityScoringLayer(app_state)
        
        # Test specific scenarios that should match different templates
        test_scenarios = [
            {
                "name": "Service Quality",
                "text": "The staff was incredibly friendly and provided excellent customer service throughout our visit."
            },
            {
                "name": "Location Details", 
                "text": "The restaurant is conveniently located with ample parking and easy accessibility via public transportation."
            },
            {
                "name": "Pricing Information",
                "text": "The prices are very reasonable - around $15 per entree with generous portion sizes. Great value for money."
            },
            {
                "name": "Wait Times",
                "text": "We had to wait about 20 minutes for a table, but it was worth it during the busy Friday evening rush."
            },
            {
                "name": "Specific Recommendations",
                "text": "I highly recommend trying their signature burger and arriving early to avoid crowds. The outdoor seating is perfect for warm evenings."
            }
        ]
        
        for scenario in test_scenarios:
            logger.info(f"\n🧪 Testing: {scenario['name']}")
            logger.info(f"Text: {scenario['text']}")
            
            results = layer.run([scenario['text']])
            result = results[0] if results else {}
            
            matched_templates = result.get('matched_templates', [])
            high_confidence_matches = result.get('high_confidence_matches', 0)
            informativeness_score = result.get('informativeness_score', 0.0)
            
            logger.info(f"✅ Informativeness: {informativeness_score:.3f}")
            logger.info(f"✅ High-confidence matches: {high_confidence_matches}")
            logger.info(f"✅ Matched templates ({len(matched_templates)}):")
            for template in matched_templates[:3]:
                logger.info(f"   • {template}")
        
        logger.info("\n✅ Template matching test completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Template matching test failed: {e}")
        raise


if __name__ == "__main__":
    try:
        test_quality_scoring_analysis()
        print("\n" + "="*80 + "\n")
        test_batch_processing()
        print("\n" + "="*80 + "\n")
        test_template_matching()
    except KeyboardInterrupt:
        logger.info("Tests interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)