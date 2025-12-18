#!/usr/bin/env python3
"""
Test script for the ImageClassificationLayer functionality.
Simple tests using real image data.
"""

import logging
import sys
import os

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from typing import List, Dict, Any
from io import BytesIO

import requests

try:
    from PIL import Image
    import numpy as np
except ImportError as e:
    print(f"Missing required dependencies: {e}")
    print("Please install: pip install pillow numpy requests")
    sys.exit(1)

from pipeline.app_state import AppState
from pipeline.layers.image_classification import (
    ImageClassificationLayer,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_image_from_url(url: str) -> Image.Image:
    """
    Load an image from URL.
    
    Args:
        url: Image URL
        
    Returns:
        PIL Image object
    """
    try:
        logger.info(f"Loading image from: {url}")
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content)).convert('RGB')
        logger.info(f"Successfully loaded image: {image.size}")
        return image
    except Exception as e:
        logger.error(f"Failed to load image from {url}: {e}")
        raise


def test_hello_world_image(layer: ImageClassificationLayer) -> bool:
    """
    Simple hello world test with COCO dataset image.
    
    Args:
        layer: ImageClassificationLayer instance
        
    Returns:
        True if test passes, False otherwise
    """
    try:
        logger.info("=== Hello World Image Classification Test ===")
        
        # Load image from COCO dataset
        url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        test_image = load_image_from_url(url)
        
        # Run classification
        predictions = layer.run(test_image)
        
        # Display results
        prediction = predictions[0]
        logger.info(f"🎯 Classification Result:")
        logger.info(f"   Predicted class: {prediction['predicted_class']}")
        logger.info(f"   Confidence: {prediction['confidence']:.4f}")
        logger.info(f"   Class ID: {prediction['class_id']}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Hello world test failed: {e}")
        return False


def test_data_integration(layer: ImageClassificationLayer) -> bool:
    # test with review images
    pass

def test_batch_processing(layer: ImageClassificationLayer) -> bool:
    """
    Test batch processing with multiple images from COCO dataset.
    
    Args:
        layer: ImageClassificationLayer instance
        
    Returns:
        True if test passes, False otherwise
    """
    try:
        logger.info("=== Batch Processing Test ===")
        
        # Multiple COCO images for batch testing
        image_urls = [
            "http://images.cocodataset.org/val2017/000000039769.jpg",  # cats
            "http://images.cocodataset.org/val2017/000000397133.jpg",  # person
            "http://images.cocodataset.org/val2017/000000037777.jpg"   # stop sign
        ]
        
        # Load all images
        images = []
        for i, url in enumerate(image_urls):
            try:
                image = load_image_from_url(url)
                images.append(image)
                logger.info(f"Loaded image {i+1}/{len(image_urls)}")
            except Exception as e:
                logger.warning(f"Failed to load image {i+1}, using placeholder")
                # Create a simple colored placeholder if URL fails
                placeholder = Image.new('RGB', (224, 224), color=(128, 128, 128))
                images.append(placeholder)
        
        # Run batch classification
        predictions = layer.run(images)
        
        # Display batch results
        logger.info(f"🎯 Batch Classification Results:")
        for i, prediction in enumerate(predictions):
            logger.info(f"   Image {i+1}: {prediction['predicted_class']} ({prediction['confidence']:.4f})")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Batch processing test failed: {e}")
        return False


def run_all_tests() -> None:
    """
    Run simplified image classification tests.
    """
    logger.info("🚀 Image Classification Layer Tests")
    logger.info("=" * 50)
    
    try:
        # Initialize the layer
        logger.info("Initializing AppState and ImageClassificationLayer...")
        app_state = AppState()
        layer = ImageClassificationLayer(app_state)
        
        try:
            _ = app_state.image_processor
            _ = app_state.image_model
            logger.info("✓ Models loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            logger.error("Please ensure transformers and required models are installed")
            return
        
        # Run simplified tests
        tests = [
            ("Hello World Image Test", test_hello_world_image),
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