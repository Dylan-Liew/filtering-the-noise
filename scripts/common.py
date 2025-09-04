"""
Common utilities for scripts, including data loading functions.
"""

import pandas as pd
from pathlib import Path
from typing import Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_reviews_data(data_path: Optional[str] = None) -> pd.DataFrame:
    """
    Load the processed reviews data from parquet file.
    
    Args:
        data_path: Optional path to the data file. If None, uses default path.
        
    Returns:
        pandas DataFrame with review data containing columns:
        - review_uuid: Unique identifier for the review
        - review_text: The actual review text content
        - has_image: Boolean indicating if review has associated images
        - batch_id: Batch identifier for processing
        - original_index: Original index in source dataset
        - review_time: Timestamp of review
        - rating: Star rating given
        - original_text: Original unprocessed text
        - original_category: Business category
        - business_avg_rating: Average rating of business
        - business_num_of_reviews: Number of reviews for business
        - is_spam: Boolean spam classification
        - reason: Detailed reason for spam classification
        - short_reason: Short reason for spam classification
        - classification_timestamp: When classification was done
        - classified_review_text: Processed review text
        
    Raises:
        FileNotFoundError: If data file doesn't exist
        ValueError: If data format is invalid
    """
    try:
        if data_path is None:
            # Default path relative to script location
            script_dir = Path(__file__).parent
            project_root = script_dir.parent
            data_path = project_root / "data" / "processed_reviews.parquet"
        
        data_path = Path(data_path)
        
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found at: {data_path}")
            
        logger.info(f"Loading reviews data from: {data_path}")
        
        # Load the parquet file
        df = pd.read_parquet(data_path)
        
        # Validate expected columns exist
        expected_columns = [
            'review_uuid', 'review_text', 'has_image', 'batch_id', 'original_index',
            'review_time', 'rating', 'original_text', 'original_category',
            'business_avg_rating', 'business_num_of_reviews', 'is_spam', 'reason',
            'short_reason', 'classification_timestamp', 'classified_review_text'
        ]
        
        missing_columns = [col for col in expected_columns if col not in df.columns]
        if missing_columns:
            logger.warning(f"Missing expected columns: {missing_columns}")
            
        logger.info(f"Loaded {len(df)} reviews with {len(df.columns)} columns")
        logger.info(f"Reviews with images: {df['has_image'].sum() if 'has_image' in df.columns else 'unknown'}")
        
        return df
        
    except Exception as e:
        logger.error(f"Error loading reviews data: {e}")
        raise


def get_reviews_with_images(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter reviews that have associated images.
    
    Args:
        df: DataFrame with review data
        
    Returns:
        Filtered DataFrame containing only reviews with images
    """
    if 'has_image' not in df.columns:
        logger.warning("'has_image' column not found, returning empty DataFrame")
        return pd.DataFrame()
    
    image_reviews = df[df['has_image'] == True].copy()
    logger.info(f"Found {len(image_reviews)} reviews with images out of {len(df)} total reviews")
    
    return image_reviews


def get_sample_reviews(df: pd.DataFrame, n: int = 10, seed: Optional[int] = 42) -> pd.DataFrame:
    """
    Get a random sample of reviews for testing.
    
    Args:
        df: DataFrame with review data
        n: Number of samples to return
        seed: Random seed for reproducibility
        
    Returns:
        Sample DataFrame
    """
    if len(df) == 0:
        logger.warning("Empty DataFrame provided")
        return df
    
    if len(df) < n:
        logger.warning(f"DataFrame has only {len(df)} rows, returning all")
        return df
    
    sample_df = df.sample(n=n, random_state=seed).copy()
    logger.info(f"Selected {len(sample_df)} sample reviews")
    
    return sample_df


def display_review_summary(df: pd.DataFrame) -> None:
    """
    Display a summary of the reviews data.
    
    Args:
        df: DataFrame with review data
    """
    print(f"\n=== Review Data Summary ===")
    print(f"Total reviews: {len(df)}")
    
    if 'has_image' in df.columns:
        print(f"Reviews with images: {df['has_image'].sum()}")
        
    if 'is_spam' in df.columns:
        print(f"Spam reviews: {df['is_spam'].sum()}")
        
    if 'rating' in df.columns:
        print(f"Average rating: {df['rating'].mean():.2f}")
        
    if 'original_category' in df.columns:
        print(f"Business categories: {df['original_category'].nunique()}")
        
    print(f"Columns: {list(df.columns)}")
    print("=" * 30)