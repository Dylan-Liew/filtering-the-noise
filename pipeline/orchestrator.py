import logging
from typing import List, Dict, Any

import pandas as pd

from .app_state import AppState
from .data_loader import ReviewDataLoader
from .layers.advertisement_detection import AdvertisementDetectionLayer, AdvertisementDetectionInput
from .layers.image_classification import ImageClassificationLayer
from .layers.quality_scoring import QualityScoringLayer
from .layers.review_relevancy import ReviewRelevancyLayer, ReviewRelevancyInput
from .layers.sentiment_analysis import SentimentAnalysisLayer
from .layers.spam_detection import SpamDetectionLayer, SpamDetectionInput
from .models import ReviewAnalysisInput, ReviewAnalysisOutput


class PipelineOrchestrator:
    """
    Orchestrates the execution of all processing layers in the review filtering pipeline.
    Manages AppState and coordinates data flow between layers.
    """
    
    def __init__(self, config_file: str = ".env"):
        self.app_state = AppState()
        self.logger = logging.getLogger("pipeline.orchestrator")
        self.data_loader = ReviewDataLoader(config_file)
        
        # Initialize all layers
        self.relevancy_layer = ReviewRelevancyLayer(self.app_state)
        self.image_layer = ImageClassificationLayer(self.app_state)
        self.ad_detection_layer = AdvertisementDetectionLayer(self.app_state)
        self.sentiment_layer = SentimentAnalysisLayer(self.app_state)
        self.quality_scoring_layer = QualityScoringLayer(self.app_state)
        self.spam_detection_layer = SpamDetectionLayer(self.app_state)
        
    def load_data(self, limit: int = None, filter_null_reviews: bool = True, source_file: str = "data/reviews.csv") -> pd.DataFrame:
        """Load review data from file.
        
        Args:
            limit: Maximum number of records to load
            filter_null_reviews: If True, only load reviews with non-null review_cleaned
            source_file: Path to source CSV file
        """
        return self.data_loader.load_and_preprocess(limit, filter_null_reviews, source_file)
    
    def process_dataframe(self, df: pd.DataFrame, batch_size: int = 100) -> pd.DataFrame:
        """
        Process entire DataFrame through the pipeline.
        
        Args:
            df: DataFrame with review data
            batch_size: Number of reviews to process in each batch
            
        Returns:
            DataFrame with all analysis results
        """
        self.logger.info(f"Processing {len(df)} reviews in batches of {batch_size}")
        
        results = []
        total_batches = (len(df) + batch_size - 1) // batch_size
        
        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(df))
            batch_df = df.iloc[start_idx:end_idx].copy()
            
            self.logger.info(f"Processing batch {batch_idx + 1}/{total_batches}")
            
            batch_results = self._process_batch(batch_df)
            results.extend(batch_results)
        
        # Convert results back to DataFrame
        results_df = self._combine_results(df, results)
        return results_df
    
    def _process_batch(self, batch_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Process a batch of reviews."""
        batch_results = []
        
        # Extract texts for batch processing
        texts = batch_df['review_cleaned'].tolist()
        
        # Run text-based analyses in batch
        ad_results = self._run_advertisement_detection_batch(texts)
        sentiment_results = self._run_sentiment_analysis_batch(texts)
        quality_results = self._run_quality_scoring_batch(texts)
        spam_results = self._run_spam_detection_batch(texts)
        
        # Process each review individually for relevancy and images
        for idx, (_, row) in enumerate(batch_df.iterrows()):
            result = {
                'uuid': row.get('uuid'),
                'original_index': row.get('original_index'),
                'advertisement_analysis': ad_results[idx] if idx < len(ad_results) else {},
                'sentiment_analysis': sentiment_results[idx] if idx < len(sentiment_results) else {},
                'quality_analysis': quality_results[idx] if idx < len(quality_results) else {},
                'spam_analysis': spam_results[idx] if idx < len(spam_results) else {},
                'relevancy_analysis': {},
                'image_analysis': {},
                'final_verdict': {}
            }
            
            # Relevancy analysis - pass data as-is
            business_info = {
                'business_description': row.get('business_description'),
                'category': row.get('category', [])
            }
            
            if row['review_cleaned'].strip():
                try:
                    relevancy_input = ReviewRelevancyInput(row['review_cleaned'], business_info)
                    relevancy_output = self.relevancy_layer.run(relevancy_input)
                    result['relevancy_analysis'] = {
                        'is_relevant': relevancy_output.is_relevant,
                        'relevancy_score': relevancy_output.relevancy_score,
                        'analysis': relevancy_output.analysis
                    }
                except Exception as e:
                    self.logger.error(f"Relevancy analysis failed for review {idx}: {e}")
                    result['relevancy_analysis'] = {'error': str(e)}
            
            # Image analysis (placeholder - would need actual image loading)
            if row.get('has_image', False):  # Updated to match database column
                # In a real implementation, you'd load and process actual images
                result['image_analysis'] = {'has_images': True, 'processed': False}
            
            # Compute final verdict
            result['final_verdict'] = self._compute_final_verdict(result)
            
            batch_results.append(result)
        
        return batch_results
    
    def _run_advertisement_detection_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Run advertisement detection on batch of texts."""
        try:
            input_data = AdvertisementDetectionInput(texts)
            detections = self.ad_detection_layer.run(input_data)
            return detections
        except Exception as e:
            self.logger.error(f"Advertisement detection batch failed: {e}")
            return [{'error': str(e)} for _ in texts]
    
    def _run_sentiment_analysis_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Run sentiment analysis on batch of texts."""
        try:
            return self.sentiment_layer.run(texts)
        except Exception as e:
            self.logger.error(f"Sentiment analysis batch failed: {e}")
            return [{'error': str(e)} for _ in texts]
    
    def _run_quality_scoring_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Run quality scoring on batch of texts."""
        try:
            return self.quality_scoring_layer.run(texts)
        except Exception as e:
            self.logger.error(f"Quality scoring batch failed: {e}")
            return [{'error': str(e)} for _ in texts]
    
    def _run_spam_detection_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Run spam detection on batch of texts."""
        try:
            spam_results = []
            for text in texts:
                input_data = SpamDetectionInput(text)
                result = self.spam_detection_layer.run(input_data)
                spam_results.append({
                    'is_spam': result.is_spam,
                    'spam_probability': result.spam_probability,
                    'predicted_label': result.predicted_label,
                    'confidence_scores': result.confidence_scores
                })
            return spam_results
        except Exception as e:
            self.logger.error(f"Spam detection batch failed: {e}")
            return [{'is_spam': False, 'spam_probability': 0.0, 'error': str(e)} for _ in texts]
    
    def _compute_final_verdict(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Compute final filtering verdict based on all layer results."""
        reasons = []
        should_filter = False
        confidence_scores = []
        
        # Advertisement detection
        ad_analysis = result.get('advertisement_analysis', {})
        if ad_analysis.get('is_advertisement', False):
            should_filter = True
            reasons.append("Advertisement content detected")
            confidence_scores.append(ad_analysis.get('confidence', 0))
        
        # Spam detection
        spam_analysis = result.get('spam_analysis', {})
        if spam_analysis.get('is_spam', False):
            should_filter = True
            reasons.append("Spam content detected")
            confidence_scores.append(spam_analysis.get('spam_probability', 0))
        
        # Relevancy analysis
        relevancy_analysis = result.get('relevancy_analysis', {})
        if not relevancy_analysis.get('is_relevant', True):
            should_filter = True
            reasons.append("Not relevant to business")
            confidence_scores.append(1 - relevancy_analysis.get('relevancy_score', 0.5))
        
        # Sentiment analysis (rant detection)
        sentiment_analysis = result.get('sentiment_analysis', {})
        if sentiment_analysis.get('is_rant', False):
            should_filter = True
            reasons.append("Rant content detected")
            confidence_scores.append(sentiment_analysis.get('rant_score', 0))
        
        # Image analysis
        image_analysis = result.get('image_analysis', {})
        if image_analysis.get('has_inappropriate_content', False):
            should_filter = True
            reasons.append("Inappropriate image content")
        
        # Enhanced quality analysis using semantic insights
        quality_analysis = result.get('quality_analysis', {})
        quality_rating = quality_analysis.get('quality_rating', 3)
        informativeness_score = quality_analysis.get('informativeness_score', 0.5)
        helpfulness_score = quality_analysis.get('helpfulness_score', 0.5)
        high_confidence_matches = quality_analysis.get('high_confidence_matches', 0)
        
        # Filter low quality reviews with enhanced criteria
        if quality_rating <= 2:
            should_filter = True
            reasons.append(f"Low quality content ({quality_rating}/5)")
            confidence_scores.append(1 - quality_rating / 5)
        
        # Filter reviews with very low informativeness (more lenient)
        if informativeness_score < 0.15:
            should_filter = True
            reasons.append("Very low informativeness")
            confidence_scores.append(1 - informativeness_score)
        
        # Filter reviews with very low helpfulness and no high-confidence semantic matches (more lenient)
        if helpfulness_score < 0.25 and high_confidence_matches == 0:
            should_filter = True
            reasons.append("Unhelpful content with no clear informative elements")
            confidence_scores.append(1 - helpfulness_score)
        
        # Overall confidence
        overall_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
        
        return {
            'should_filter': should_filter,
            'reason': '; '.join(reasons) if reasons else 'No issues detected',
            'final_verdict': 'FILTER' if should_filter else 'KEEP'
        }
    
    def _combine_results(self, original_df: pd.DataFrame, 
                        results: List[Dict[str, Any]]) -> pd.DataFrame:
        """Combine original data with analysis results."""
        # Create results DataFrame
        flattened_results = []
        
        for result in results:
            flat_result = {
                'uuid': result.get('uuid'),
                'original_index': result.get('original_index'),
            }
            
            # Advertisement detection results
            ad = result.get('advertisement_analysis', {})
            flat_result.update({
                'is_advertisement': ad.get('is_advertisement', False),
                'ad_confidence': ad.get('confidence', 0.0)
            })
            
            # Spam detection results
            spam = result.get('spam_analysis', {})
            flat_result.update({
                'is_spam': spam.get('is_spam', False),
                'spam_probability': spam.get('spam_probability', 0.0)
            })
            
            # Sentiment analysis results (rant detection)
            sentiment = result.get('sentiment_analysis', {})
            flat_result.update({
                'is_rant': sentiment.get('is_rant', False),
                'rant_confidence': sentiment.get('rant_score', 0.0)
            })
            
            # Relevancy analysis results
            relevancy = result.get('relevancy_analysis', {})
            flat_result.update({
                'is_relevant': relevancy.get('is_relevant', True),
                'relevancy_score': relevancy.get('relevancy_score', 0.5)
            })
            
            # Quality scoring results with enhanced semantic analysis
            quality = result.get('quality_analysis', {})
            flat_result.update({
                'quality_rating': quality.get('quality_rating', 3),
                'helpfulness_score': quality.get('helpfulness_score', 0.5),
                'usefulness_score': quality.get('usefulness_score', 0.5),
                'informativeness_score': quality.get('informativeness_score', 0.5),
                'semantic_quality_score': quality.get('semantic_quality_score', 0.5),
                'detail_score': quality.get('detail_score', 0.5),
                'word_count': quality.get('word_count', 0),
                'high_confidence_matches': quality.get('high_confidence_matches', 0),
                'matched_templates_count': len(quality.get('matched_templates', [])),
                'matched_templates': '; '.join(quality.get('matched_templates', []))
            })
            
            # Final verdict
            verdict = result.get('final_verdict', {})
            flat_result.update({
                'should_filter': verdict.get('should_filter', False),
                'reason': verdict.get('reason', 'No issues detected'),
                'final_verdict': verdict.get('final_verdict', 'KEEP')
            })
            
            flattened_results.append(flat_result)
        
        results_df = pd.DataFrame(flattened_results)
        
        # Merge with original data using uuid as the key
        final_df = original_df.merge(results_df, on='uuid', how='left')
        
        return final_df
    
    def analyze_single_review(self, review_input: ReviewAnalysisInput) -> ReviewAnalysisOutput:
        """
        Analyze a single review through the entire pipeline.
        
        Args:
            review_input: ReviewAnalysisInput containing review data
            
        Returns:
            ReviewAnalysisOutput with analysis results
        """
        self.logger.info(f"Analyzing single review: {review_input.uuid or 'unnamed'}")
        
        try:
            # Run text-based analyses
            ad_result = self._run_single_advertisement_detection(review_input.review_cleaned)
            sentiment_result = self._run_single_sentiment_analysis(review_input.review_cleaned)
            quality_result = self._run_single_quality_scoring(review_input.review_cleaned)
            spam_result = self._run_single_spam_detection(review_input.review_cleaned)
            
            # Run relevancy analysis - pass data as-is
            business_info = {
                'business_description': review_input.business_description,
                'category': review_input.category if isinstance(review_input.category, list) else [review_input.category] if review_input.category else []
            }
            
            relevancy_result = {}
            if review_input.review_cleaned.strip():
                try:
                    relevancy_input = ReviewRelevancyInput(review_input.review_cleaned, business_info)
                    relevancy_output = self.relevancy_layer.run(relevancy_input)
                    relevancy_result = {
                        'is_relevant': relevancy_output.is_relevant,
                        'relevancy_score': relevancy_output.relevancy_score,
                        'analysis': relevancy_output.analysis
                    }
                except Exception as e:
                    self.logger.error(f"Relevancy analysis failed: {e}")
                    relevancy_result = {'is_relevant': True, 'relevancy_score': 0.5, 'analysis': {}, 'error': str(e)}
            
            # Image analysis (placeholder)
            image_result = {}
            if review_input.has_image:
                image_result = {'has_images': True, 'processed': False}
            
            # Combine results
            combined_result = {
                'uuid': review_input.uuid,
                'advertisement_analysis': ad_result,
                'sentiment_analysis': sentiment_result,
                'quality_analysis': quality_result,
                'spam_analysis': spam_result,
                'relevancy_analysis': relevancy_result,
                'image_analysis': image_result,
                'final_verdict': {}
            }
            
            # Compute final verdict
            combined_result['final_verdict'] = self._compute_final_verdict(combined_result)
            
            # Convert to structured output with enhanced semantic fields
            return ReviewAnalysisOutput(
                uuid=review_input.uuid,
                is_advertisement=ad_result.get('is_advertisement', False),
                ad_confidence=ad_result.get('confidence', 0.0),
                is_spam=spam_result.get('is_spam', False),
                spam_probability=spam_result.get('spam_probability', 0.0),
                is_rant=sentiment_result.get('is_rant', False),
                rant_confidence=sentiment_result.get('rant_score', 0.0),
                is_relevant=relevancy_result.get('is_relevant', True),
                relevancy_score=relevancy_result.get('relevancy_score', 0.5),
                quality_rating=quality_result.get('quality_rating', 3),
                helpfulness_score=quality_result.get('helpfulness_score', 0.5),
                usefulness_score=quality_result.get('usefulness_score', 0.5),
                informativeness_score=quality_result.get('informativeness_score', 0.5),
                semantic_quality_score=quality_result.get('semantic_quality_score', 0.5),
                detail_score=quality_result.get('detail_score', 0.5),
                word_count=quality_result.get('word_count', 0),
                high_confidence_matches=quality_result.get('high_confidence_matches', 0),
                matched_templates=quality_result.get('matched_templates', []),
                should_filter=combined_result['final_verdict'].get('should_filter', False),
                reason=combined_result['final_verdict'].get('reason', 'No issues detected'),
                final_verdict=combined_result['final_verdict'].get('final_verdict', 'KEEP')
            )
            
        except Exception as e:
            self.logger.error(f"Single review analysis failed: {e}")
            # Return default "keep" result on error with semantic defaults
            return ReviewAnalysisOutput(
                uuid=review_input.uuid,
                is_advertisement=False,
                ad_confidence=0.0,
                is_spam=False,
                spam_probability=0.0,
                is_rant=False,
                rant_confidence=0.0,
                is_relevant=True,
                relevancy_score=0.5,
                quality_rating=3,
                helpfulness_score=0.5,
                usefulness_score=0.5,
                informativeness_score=0.5,
                semantic_quality_score=0.5,
                detail_score=0.5,
                word_count=0,
                high_confidence_matches=0,
                matched_templates=[],
                should_filter=False,
                reason=f"Analysis failed: {str(e)}",
                final_verdict="KEEP"
            )
    
    def _run_single_advertisement_detection(self, text: str) -> Dict[str, Any]:
        """Run advertisement detection on single text."""
        try:
            input_data = AdvertisementDetectionInput([text])
            detections = self.ad_detection_layer.run(input_data)
            return detections[0] if detections else {'is_advertisement': False, 'confidence': 0.0}
        except Exception as e:
            self.logger.error(f"Single advertisement detection failed: {e}")
            return {'is_advertisement': False, 'confidence': 0.0, 'error': str(e)}
    
    def _run_single_sentiment_analysis(self, text: str) -> Dict[str, Any]:
        """Run sentiment analysis on single text."""
        try:
            results = self.sentiment_layer.run([text])
            return results[0] if results else {'is_rant': False, 'rant_score': 0.0}
        except Exception as e:
            self.logger.error(f"Single sentiment analysis failed: {e}")
            return {'is_rant': False, 'rant_score': 0.0, 'error': str(e)}
    
    def _run_single_quality_scoring(self, text: str) -> Dict[str, Any]:
        """Run quality scoring on single text."""
        try:
            results = self.quality_scoring_layer.run([text])
            return results[0] if results else {'quality_rating': 3}
        except Exception as e:
            self.logger.error(f"Single quality scoring failed: {e}")
            return {'quality_rating': 3, 'error': str(e)}
    
    def _run_single_spam_detection(self, text: str) -> Dict[str, Any]:
        """Run spam detection on single text."""
        try:
            input_data = SpamDetectionInput(text)
            result = self.spam_detection_layer.run(input_data)
            return {
                'is_spam': result.is_spam,
                'spam_probability': result.spam_probability,
                'predicted_label': result.predicted_label,
                'confidence_scores': result.confidence_scores
            }
        except Exception as e:
            self.logger.error(f"Single spam detection failed: {e}")
            return {'is_spam': False, 'spam_probability': 0.0, 'error': str(e)}
    
    def export_results(self, df: pd.DataFrame, output_file: str = "processed_reviews.csv"):
        """Export results to CSV file."""
        self.logger.info(f"Exporting results to {output_file}")
        
        # Select relevant columns for export with enhanced semantic analysis fields
        export_columns = [
            'uuid', 'review_cleaned', 'rating', 'review_time', 'business_description', 'category',
            'is_advertisement', 'ad_confidence', 
            'is_spam', 'spam_probability',
            'is_rant', 'rant_confidence',
            'is_relevant', 'relevancy_score', 
            'quality_rating', 'helpfulness_score', 'usefulness_score', 'informativeness_score',
            'semantic_quality_score', 'detail_score',
            'high_confidence_matches', 'matched_templates_count', 'matched_templates',
            'should_filter', 'reason', 'final_verdict'
        ]
        
        # Include only columns that exist
        available_columns = [col for col in export_columns if col in df.columns]
        export_df = df[available_columns].copy()
        
        export_df.to_csv(output_file, index=False)
        self.logger.info(f"Exported {len(export_df)} reviews to {output_file}")
        
        # Print summary statistics
        self._print_summary_stats(export_df)
    
    def _print_summary_stats(self, df: pd.DataFrame):
        """Print summary statistics of the results."""
        print("\n=== Review Analysis Summary ===")
        print(f"Total reviews processed: {len(df)}")
        
        if 'should_filter' in df.columns:
            filtered_count = df['should_filter'].sum()
            print(f"Reviews to filter: {filtered_count} ({filtered_count/len(df)*100:.1f}%)")
        
        if 'is_advertisement' in df.columns:
            ad_count = df['is_advertisement'].sum()
            print(f"Advertisement reviews: {ad_count} ({ad_count/len(df)*100:.1f}%)")
        
        if 'is_spam' in df.columns:
            spam_count = df['is_spam'].sum()
            print(f"Spam reviews: {spam_count} ({spam_count/len(df)*100:.1f}%)")
        
        if 'is_rant' in df.columns:
            rant_count = df['is_rant'].sum()
            print(f"Rant reviews: {rant_count} ({rant_count/len(df)*100:.1f}%)")
        
        if 'is_relevant' in df.columns:
            irrelevant_count = (~df['is_relevant']).sum()
            print(f"Irrelevant reviews: {irrelevant_count} ({irrelevant_count/len(df)*100:.1f}%)")
        
        if 'quality_rating' in df.columns:
            avg_quality = df['quality_rating'].mean()
            print(f"Average quality rating: {avg_quality:.2f}/5")
        
        # Enhanced semantic analysis statistics
        if 'informativeness_score' in df.columns:
            avg_informativeness = df['informativeness_score'].mean()
            low_info_count = (df['informativeness_score'] < 0.3).sum()
            print(f"Average informativeness: {avg_informativeness:.3f}")
            print(f"Low informativeness reviews: {low_info_count} ({low_info_count/len(df)*100:.1f}%)")
        
        if 'helpfulness_score' in df.columns:
            avg_helpfulness = df['helpfulness_score'].mean()
            unhelpful_count = (df['helpfulness_score'] < 0.4).sum()
            print(f"Average helpfulness: {avg_helpfulness:.3f}")
            print(f"Unhelpful reviews: {unhelpful_count} ({unhelpful_count/len(df)*100:.1f}%)")
        
        if 'high_confidence_matches' in df.columns:
            avg_matches = df['high_confidence_matches'].mean()
            no_matches_count = (df['high_confidence_matches'] == 0).sum()
            print(f"Avg high-confidence semantic matches: {avg_matches:.1f}")
            print(f"Reviews with no clear informative elements: {no_matches_count} ({no_matches_count/len(df)*100:.1f}%)")
        
        
        print("=" * 32)