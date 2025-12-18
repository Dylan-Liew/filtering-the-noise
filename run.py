#!/usr/bin/env python3
"""
Main execution script for the review filtering pipeline.
Processes Google review data through ML layers and exports results.
"""

import logging
import sys
import argparse
from pathlib import Path
from pipeline.orchestrator import PipelineOrchestrator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    encoding="utf-8"
)

logger = logging.getLogger(__name__)


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Review Quality Assessment Pipeline')
    parser.add_argument('--config-file', default='.env', help='Configuration file')
    parser.add_argument('--input-file', default='data/reviews.csv', help='Input CSV file containing reviews')
    parser.add_argument('--output-file', default='processed_reviews.csv', help='Output CSV file name')
    parser.add_argument('--batch-size', type=int, default=100, help='Batch size for processing')
    parser.add_argument('--limit', type=int, help='Limit number of reviews to process (for testing)')
    parser.add_argument('--include-null-reviews', action='store_true', 
                        help='Include reviews with null/empty review_cleaned (default: filter them out)')
    
    args = parser.parse_args()
    
    try:
        logger.info("🚀 Starting Review Quality Assessment Pipeline")
        logger.info("=" * 60)
        
        # Verify config file exists
        config_file = Path(args.config_file)
        # if not config_file.exists():
        #     logger.warning(f"Configuration file not found: {config_file}")

        # Initialize orchestrator
        logger.info("Initializing pipeline orchestrator...")
        orchestrator = PipelineOrchestrator(args.config_file)
        
        # Load data
        filter_nulls = not args.include_null_reviews
        if filter_nulls:
            logger.info(f"Loading review data from {args.input_file} (filtering null/empty reviews)...")
        else:
            logger.info(f"Loading review data from {args.input_file} (including all reviews)...")
        df = orchestrator.load_data(args.limit, filter_null_reviews=filter_nulls, source_file=args.input_file)
        
        if df.empty:
            logger.error("No review data loaded. Please check your data files.")
            return 1
        
        logger.info(f"Loaded {len(df)} reviews for processing")
        
        # Process reviews through pipeline
        logger.info("Processing reviews through ML pipeline...")
        processed_df = orchestrator.process_dataframe(df, batch_size=args.batch_size)
        
        # Export results
        logger.info("Exporting results...")
        orchestrator.export_results(processed_df, args.output_file)
        
        logger.info("✅ Pipeline execution completed successfully!")
        logger.info(f"Results saved to: {args.output_file}")
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
