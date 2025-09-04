from typing import List, Union
from PIL import Image
import torch
from .base_layer import BaseLayer



class ImageClassificationLayer(BaseLayer):
    """
    Layer for classifying images using pre-trained MobileNet v2 model.
    Classifies images into ImageNet classes.
    """
    
    def run(self, input_data: Union[Image.Image, List[Image.Image]]) -> List[dict]:
        """
        Classify input images.
        
        Args:
            input_data: ImageClassificationInput containing PIL images
            
        Returns:
            ImageClassificationOutput with classification predictions
        """
        if isinstance(input_data, Image.Image):
            input_data = [input_data]
        try:
            self._log_execution_start(f"Processing {len(input_data)} images")
            
            preprocessor = self.app_state.image_processor
            model = self.app_state.image_model
            
            predictions = []
            
            for i, image in enumerate(input_data):
                inputs = preprocessor(images=image, return_tensors="pt")
                outputs = model(**inputs)
                logits = outputs.logits

                predicted_class_idx = logits.argmax(-1).item()
                confidence = torch.softmax(logits, dim=-1).max().item()
                predicted_class = model.config.id2label[predicted_class_idx]
                
                predictions.append({
                    "image_index": i,
                    "predicted_class": predicted_class,
                    "class_id": predicted_class_idx,
                    "confidence": confidence
                })
            

            self._log_execution_end(
                f"Classified {len(predictions)} images with predictions: "
                f"{[p['predicted_class'] for p in predictions]}"
            )
            
            return predictions
            
        except Exception as e:
            self._handle_error(e, f"num_images={len(input_data)}")
            raise
