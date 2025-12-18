"""
Test script for spam detection layer.
Tests the SpamDetectionLayer with sample spam and ham texts.
"""

import sys
import os

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pipeline.app_state import AppState
from pipeline.layers.spam_detection import SpamDetectionLayer, SpamDetectionInput

def test_spam_detection():
    """Test the spam detection layer with sample texts."""
    
    # Initialize app state and layer
    app_state = AppState()
    spam_layer = SpamDetectionLayer(app_state)
    
    # Test texts - mix of spam and legitimate reviews
    test_texts = [
        # Obvious spam
        "Congratulations! You've won a $1,000 gift card. Click here to claim your prize now!",
        "URGENT: Your account will be suspended unless you verify your details immediately. Click now!",
        "Make $5000 working from home! No experience needed. Call now!!!",
        
        # Legitimate reviews
        "The food was excellent and the service was outstanding. Highly recommend this restaurant!",
        "Great atmosphere, friendly staff, but the wait was a bit long. Overall a good experience.",
        "Terrible experience. The food was cold and the server was rude. Won't be coming back.",
        "Perfect location, clean rooms, and helpful front desk staff. Will definitely stay here again.",
        
        # Borderline cases
        "Best restaurant EVER!!! You MUST try it!!! Amazing deals every day!!!",
        "Hi team, just a reminder that our meeting is scheduled for 10 AM tomorrow.",
    ]
    
    print("=" * 80)
    print("SPAM DETECTION LAYER TEST")
    print("=" * 80)
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n--- Test {i} ---")
        print(f"Text: '{text[:60]}{'...' if len(text) > 60 else ''}'")
        
        try:
            # Create input and run spam detection
            input_data = SpamDetectionInput(text)
            result = spam_layer.run(input_data)
            
            # Display results
            print(f"Classification: {result.predicted_label}")
            print(f"Is Spam: {'YES' if result.is_spam else 'NO'}")
            print(f"Spam Probability: {result.spam_probability:.3f}")
            print("Confidence Scores:")
            for label, score in result.confidence_scores.items():
                print(f"  {label}: {score:.3f}")
                
        except Exception as e:
            print(f"ERROR: {str(e)}")
    
    print("\n" + "=" * 80)
    print("Test completed!")

if __name__ == "__main__":
    test_spam_detection()