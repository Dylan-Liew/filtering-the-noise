import logging
import re
from typing import Optional
import os

import pandas as pd
from dotenv import dotenv_values

logger = logging.getLogger(__name__)


class ReviewDataLoader:
    """Loads preprocessed review data from CSV files."""
    
    def __init__(self, config_file: str = ".env"):
        self.config = dotenv_values(config_file)
        
    def _clean_review_text(self, text: str) -> str:
        """Clean review text by removing Google Translate patterns.
        
        Removes patterns like:
        - (Translated by Google) ... (Original) ...
        - Any text after (Original) marker
        
        Args:
            text: Raw review text
            
        Returns:
            Cleaned review text with only the translated portion
        """
        if not text or not isinstance(text, str):
            return text
            
        google_translate_pattern = r'\(Translated by Google\)\s*(.*?)\s*\(Original\).*?$'
        
        alt_pattern = r'\(Translated by Google\)(.*?)\(Original\)'
        
        match = re.search(google_translate_pattern, text, re.DOTALL | re.IGNORECASE)
        if not match:
            match = re.search(alt_pattern, text, re.DOTALL | re.IGNORECASE)
            
        if match:
            cleaned_text = match.group(1).strip()
            # logger.debug(f"Cleaned Google Translate pattern: '{text[:50]}...' -> '{cleaned_text[:50]}...'")
            return cleaned_text
        
        return text
    
    def load_from_csv(self, file_path: str, limit: Optional[int] = None, filter_null_reviews: bool = True) -> pd.DataFrame:
        """Load review data from CSV.
        
        Args:
            file_path: Path to CSV file
            limit: Maximum number of records to load
            filter_null_reviews: If True, only load reviews with non-null review_cleaned
        """
        logger.info(f"Loading review data from {file_path}")
        
        if not os.path.exists(file_path):
             logger.error(f"File not found: {file_path}")
             return pd.DataFrame()

        try:
            df = pd.read_csv(file_path)
            
            # Ensure required columns exist
            required_columns = ['review_cleaned', 'quality_rating']
            for col in required_columns:
                if col not in df.columns:
                     # dynamic fallback or error?
                     if col == 'review_cleaned' and 'text' in df.columns:
                         df['review_cleaned'] = df['text']
                     elif col == 'quality_rating' and 'rating' in df.columns:
                         df['quality_rating'] = df['rating']
            
            # Filter for non-null review_cleaned if requested
            if filter_null_reviews and 'review_cleaned' in df.columns:
                df = df[df['review_cleaned'].notna()]
                df = df[df['review_cleaned'] != '']
                logger.info("Filtering for reviews with non-null review_cleaned")
            
            if limit:
                df = df.head(limit)
                
            # Clean text
            if 'review_cleaned' in df.columns:
                df['review_cleaned'] = df['review_cleaned'].apply(self._clean_review_text)

            # Ensure uuid exists
            if 'uuid' not in df.columns:
                import uuid
                df['uuid'] = [str(uuid.uuid4()) for _ in range(len(df))]

            logger.info(f"Loaded {len(df)} reviews from CSV")
            
            return df
        except Exception as e:
            logger.error(f"Error loading CSV: {e}")
            return pd.DataFrame()
    
    def load_and_preprocess(self, limit: Optional[int] = None, filter_null_reviews: bool = True, source_file: str = "data/reviews.csv") -> pd.DataFrame:
        """Load data from source file.
        
        Args:
            limit: Maximum number of records to load
            filter_null_reviews: If True, only load reviews with non-null review_cleaned
            source_file: Source CSV file path
        """
        return self.load_from_csv(source_file, limit, filter_null_reviews)
    
    def close(self):
        """Close connections (noop for CSV)."""
        pass
