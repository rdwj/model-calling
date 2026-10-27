"""
Adapter for the Granite Guardian content moderation model.
"""
import json
from typing import Dict, Any, List

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from model_calling.adapter import ModelAdapter


class GraniteGuardianAdapter(ModelAdapter):
    """Adapter for IBM's Granite Guardian content moderation model"""

    def __init__(self, model_config: Dict[str, Any]):
        super().__init__(model_config)
        # Initialize model and tokenizer
        model_name = "ibm-granite/granite-guardian-3.2-3b-a800m"  # Using smaller model by default
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)

    async def translate_request(self, openai_format_request: Dict[str, Any]) -> Dict[str, Any]:
        """Convert OpenAI moderation format request to model input format"""
        # Extract input text from request
        input_text = openai_format_request.get("input", "")
        if isinstance(input_text, list):
            input_text = " ".join(input_text)  # Combine multiple inputs
            
        return {"text": input_text}

    async def translate_response(self, model_response: Dict[str, Any]) -> Dict[str, Any]:
        """Convert model output to OpenAI moderation format"""
        # Extract probabilities
        probs = model_response["probabilities"]
        
        # Create results entry
        result = {
            "categories": {
                "hate": bool(probs["hate"] > 0.5),
                "hate/threatening": bool(probs["hate/threatening"] > 0.5),
                "harassment": bool(probs["harassment"] > 0.5),
                "self-harm": bool(probs["self-harm"] > 0.5),
                "sexual": bool(probs["sexual"] > 0.5),
                "sexual/minors": bool(probs["sexual/minors"] > 0.5),
                "violence": bool(probs["violence"] > 0.5),
                "violence/graphic": bool(probs["violence/graphic"] > 0.5)
            },
            "category_scores": {
                "hate": float(probs["hate"]),
                "hate/threatening": float(probs["hate/threatening"]),
                "harassment": float(probs["harassment"]),
                "self-harm": float(probs["self-harm"]),
                "sexual": float(probs["sexual"]),
                "sexual/minors": float(probs["sexual/minors"]),
                "violence": float(probs["violence"]),
                "violence/graphic": float(probs["violence/graphic"])
            },
            "flagged": any(probs[k] > 0.5 for k in probs)
        }
        
        return {
            "id": "modr-" + self.model_name,
            "model": self.model_name,
            "results": [result]
        }

    async def _call_model_api(self, translated_request: Dict[str, Any]) -> Dict[str, Any]:
        """Call the Granite Guardian model for content moderation"""
        text = translated_request["text"]
        
        # Tokenize and get model outputs
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        # Get probabilities for each category
        probs = torch.sigmoid(outputs.logits[0]).tolist()
        
        # Map to category names
        category_names = [
            "hate", "hate/threatening", "harassment", "self-harm",
            "sexual", "sexual/minors", "violence", "violence/graphic"
        ]
        probabilities = dict(zip(category_names, probs))
        
        return {"probabilities": probabilities}
