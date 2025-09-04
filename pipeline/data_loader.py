import logging
import re
from typing import Optional

import pandas as pd
from dotenv import dotenv_values
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from config.database_models import Data

logger = logging.getLogger(__name__)


class ReviewDataLoader:
    """Loads preprocessed review data from CockroachDB database."""
    
    def __init__(self, config_file: str = ".env"):
        self.config = dotenv_values(config_file)
        self.engine = None
        self.SessionLocal = None
        self._setup_database()
        
    def _setup_database(self):
        """Initialize database connection and session factory."""
        if "DB_URI" not in self.config:
            raise ValueError("DB_URI not found in configuration")
            
        self.engine = create_engine(self.config["DB_URI"])
        self.SessionLocal = sessionmaker(bind=self.engine)
        logger.info("Database connection established")
    
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
            logger.debug(f"Cleaned Google Translate pattern: '{text[:50]}...' -> '{cleaned_text[:50]}...'")
            return cleaned_text
        
        return text
    
    def load_from_database(self, limit: Optional[int] = None, filter_null_reviews: bool = True) -> pd.DataFrame:
        """Load review data from database.
        
        Args:
            limit: Maximum number of records to load
            filter_null_reviews: If True, only load reviews with non-null review_cleaned
        """
        logger.info("Loading review data from database")
        
        with self.SessionLocal() as session:
            query = select(Data)
            
            # Filter for non-null review_cleaned if requested
            if filter_null_reviews:
                query = query.where(Data.review_cleaned.isnot(None))
                query = query.where(Data.review_cleaned != '')
                logger.info("Filtering for reviews with non-null review_cleaned")
            
            if limit:
                query = query.limit(limit)
                
            result = session.execute(query)
            records = result.fetchall()
            
            # Convert to list of dictionaries
            data_list = []
            for record in records:
                data_obj = record[0]  # Extract the Data object from the tuple
                data_list.append({
                    'rowid': data_obj.rowid,
                    'review_time': data_obj.review_time,
                    'rating': data_obj.rating,
                    'uuid': data_obj.uuid,
                    'has_image': data_obj.has_image,
                    'images': data_obj.images,
                    'review_cleaned': self._clean_review_text(data_obj.review_cleaned),
                    'business_description': data_obj.business_description,
                    'category': data_obj.category
                })
            
            df = pd.DataFrame(data_list)
            logger.info(f"Loaded {len(df)} reviews from database")
            
            return df
    
    def load_and_preprocess(self, limit: Optional[int] = None, filter_null_reviews: bool = True) -> pd.DataFrame:
        """Load data from database (no preprocessing needed as data is already cleaned).
        
        Args:
            limit: Maximum number of records to load
            filter_null_reviews: If True, only load reviews with non-null review_cleaned
        """
        return self.load_from_database(limit, filter_null_reviews)
    
    def close(self):
        """Close database connections."""
        if self.engine:
            self.engine.dispose()
            logger.info("Database connection closed")
