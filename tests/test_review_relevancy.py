import logging
import sys
import os

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from typing import List, Dict, Any

from pipeline.app_state import AppState
from pipeline.layers.review_relevancy import ReviewRelevancyLayer, ReviewRelevancyInput

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


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
            "uuid": "test_sample_1",
            "review_cleaned": "10 adults & 1 child. The service was quick in the beginning and got our drinks right away. Then it took a long time to get two bowls of salad, 1/2 full on one, the other approximately quarter full. No one ever came back to check so we had to ask for a server to come and refill our salad. Nobody offered to bring more breadsticks so we had to wait on our server and ask for a few more.( originally given 10 breadsticks for the 11 of us) 9 adult meals were delivered at one time. The last adult male came approximately 17 minutes after everyone started eating. The food was pretty good and the servers were nice. It was just terribly slow.",
            "business_description": None,
            "category": ["Italian restaurant", "Caterer", "Family restaurant", "Gluten-free restaurant", "Takeout Restaurant", "Seafood restaurant", "Soup restaurant", "Wine bar"]
        },
        # {
        #     "uuid": "936339",
        #     "review_cleaned": "Super delicious pizza and great fresh salad :thumbs up: light skin tone:",
        #     "business_description": "Pizzeria chain offering an array of craft beers, plus specialty pizzas, salads & sandwiches.",
        #     "category": ["Pizza restaurant"]
        # },
        # {
        #     "uuid": "931968",
        #     "review_cleaned": "Self check out is great",
        #     "business_description": None,
        #     "category": ["Department store", "Clothing store", "Craft store", "Discount store", "Electronics store", "Grocery store", "Home goods store", "Sporting goods store", "Supermarket", "Toy store"]
        # },
        # {
        #     "uuid": "168682",
        #     "review_cleaned": "Remodle was well worth it but the katchup packets are still too small.",
        #     "business_description": "Classic, long-running fast-food chain known for its burgers, fries & shakes.",
        #     "category": ["Fast food restaurant", "Breakfast restaurant", "Coffee shop", "Hamburger restaurant", "Restaurant", "Sandwich shop"]
        # },
        # {
        #     "uuid": "525872",
        #     "review_cleaned": "Best sirloin steak and cesar salad. Will be back again.",
        #     "business_description": "Lively chain steakhouse serving American fare with a Southwestern spin amid Texas-themed decor.",
        #     "category": ["Steak house", "American restaurant", "New American restaurant"]
        # }
    ]


def test_real_sample_data():
    """Test relevancy analysis with real sample data from the dataset."""
    logger.info("🧪 Testing Real Sample Data Relevancy Analysis")
    logger.info("=" * 60)
    
    try:
        # Initialize the layer
        app_state = AppState()
        layer = ReviewRelevancyLayer(app_state)
        
        sample_data = get_real_sample_data()
        
        # Filter to only reviews with text
        text_reviews = [sample for sample in sample_data if sample['review_cleaned'].strip()]
        
        if not text_reviews:
            logger.warning("No reviews with text found in sample data")
            return
        
        logger.info(f"Testing {len(text_reviews)} real reviews for relevancy...")
        
        for i, sample in enumerate(text_reviews):
            logger.info(f"\n{'='*50} Sample {i+1} {'='*50}")
            print(sample)
            
            # Prepare business info
            business_info = {
                'business_description': sample['business_description'] if sample['business_description'] else "",
                'category': sample['category'] if sample['category'] else ["Other"],
            }


            review_text = sample['review_cleaned']
            
            logger.info(f"📝 Review: '{review_text}'")
            logger.info(f"🏢 Business: {business_info['business_description'][:60]}{'...' if len(business_info['business_description']) > 60 else ''}")
            logger.info(f"🏷️  Category: {business_info['category']}")

            # Generate business contexts
            business_contexts = layer._generate_business_context(business_info)
            logger.info(f"\n📋 Business contexts generated: {len(business_contexts)}")
            for j, context in enumerate(business_contexts[:3]):  # Show first 3
                logger.info(f"   {j+1}: {context}")
            if len(business_contexts) > 3:
                logger.info(f"   ... and {len(business_contexts) - 3} more")
            
            # Test similarity score computation
            similarity_scores = layer._compute_relevancy_score(review_text, business_contexts)
            
            logger.info(f"\n📊 Similarity Score Components:")
            logger.info(f"   max_business_similarity: {similarity_scores['max_business_similarity']:.4f}")
            logger.info(f"   avg_business_similarity: {similarity_scores['avg_business_similarity']:.4f}")
            logger.info(f"   business_aspect_score: {similarity_scores['business_aspect_score']:.4f}")
            logger.info(f"   irrelevant_score: {similarity_scores['irrelevant_score']:.4f}")
            
            # Calculate final relevancy score using updated formula
            final_score = (
                similarity_scores["max_business_similarity"] * 0.6 +
                similarity_scores["business_aspect_score"] * 0.5 -
                similarity_scores["irrelevant_score"] * 0.1
            )
            final_score = max(0.0, min(1.0, final_score))
            
            # Test relevancy determination criteria
            is_relevant = (
                final_score > 0.5 and
                similarity_scores["max_business_similarity"] > 0.3 and
                similarity_scores["irrelevant_score"] < 0.7
            )
            
            logger.info(f"\n✅ Final Relevancy Analysis:")
            logger.info(f"   Final Score: {final_score:.4f}")
            logger.info(f"   Score > 0.5: {final_score > 0.5}")
            logger.info(f"   Max business similarity > 0.3: {similarity_scores['max_business_similarity'] > 0.3}")
            logger.info(f"   Irrelevant score < 0.7: {similarity_scores['irrelevant_score'] < 0.7}")
            logger.info(f"   Final determination: {'✅ RELEVANT' if is_relevant else '❌ IRRELEVANT'}")
        
        logger.info("\n" + "="*60)
        logger.info("✅ Real sample data testing completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Real sample data testing failed: {e}")
        raise


def test_full_pipeline_integration():
    """Test full pipeline integration with real sample data."""
    logger.info("🔄 Testing Full Pipeline Integration")
    logger.info("=" * 60)
    
    try:
        app_state = AppState()
        layer = ReviewRelevancyLayer(app_state)
        
        sample_data = get_real_sample_data()
        
        # Filter to only reviews with text
        text_reviews = [sample for sample in sample_data if sample['review_cleaned'].strip()]
        
        logger.info(f"Testing full pipeline integration with {len(text_reviews)} reviews...")
        
        results_summary = []
        
        for sample in text_reviews:
            business_info = {
                'business_description': sample['business_description'],
                'category': sample['category']
            }
            
            # Create input and run through layer
            input_data = ReviewRelevancyInput(sample['review_cleaned'], business_info)
            result = layer.run(input_data)
            
            results_summary.append({
                'uuid': sample['uuid'],
                'review': sample['review_cleaned'],
                'category': business_info['category'][0] if business_info['category'] else 'Unknown',
                'is_relevant': result.is_relevant,
                'relevancy_score': result.relevancy_score
            })
        
        # Analyze results
        relevant_count = sum(1 for r in results_summary if r['is_relevant'])
        avg_score = sum(r['relevancy_score'] for r in results_summary) / len(results_summary)
        
        logger.info(f"\n📊 Pipeline Integration Results:")
        logger.info(f"   Total reviews processed: {len(results_summary)}")
        logger.info(f"   Relevant reviews: {relevant_count}/{len(results_summary)} ({relevant_count/len(results_summary)*100:.1f}%)")
        logger.info(f"   Average relevancy score: {avg_score:.4f}")
        
        logger.info(f"\n📋 Individual Results:")
        for result in results_summary:
            status = '✅ RELEVANT' if result['is_relevant'] else '❌ IRRELEVANT'
            logger.info(f"   {result['uuid']}: {status} (score: {result['relevancy_score']:.4f})")
            logger.info(f"      Review: {result['review'][:60]}{'...' if len(result['review']) > 60 else ''}")
            logger.info(f"      Category: {result['category']}")
            
    except Exception as e:
        logger.error(f"❌ Pipeline integration testing failed: {e}")
        raise


if __name__ == "__main__":
    try:
        test_real_sample_data()
        # test_full_pipeline_integration()
    except KeyboardInterrupt:
        logger.info("Tests interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)