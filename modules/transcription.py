import logging
import os
import json
import re
from typing import List, Dict, Any
from PIL import Image, ImageDraw, ImageFont
import google.generativeai as genai
from modules.interfaces import TranscriptionEngine

logger = logging.getLogger(__name__)

class GeminiTranscription(TranscriptionEngine):
    def __init__(self):
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set.")
        genai.configure(api_key=api_key)

    def transcribe(self, image: Image.Image, regions: List[Dict[str, Any]], config: Dict[str, Any]) -> List[Dict[str, Any]]:
        gemini_config = config.get('transcription', {}).get('gemini', {})
        model_name = gemini_config.get('model', 'gemini-1.5-pro')
        base_prompt = gemini_config.get('prompt', 'Transcribe this text.')
        
        logger.info(f"Initializing Gemini model: {model_name}")
        model = genai.GenerativeModel(model_name)
        
        # 1. Annotate Image (Visual Tagging)
        annotated_image = image.copy()
        draw = ImageDraw.Draw(annotated_image)
        
        # Load font (try default, fallback to simple)
        try:
            font = ImageFont.truetype("Arial.ttf", 20)
        except IOError:
            font = ImageFont.load_default()

        logger.info(f"Annotating {len(regions)} regions for Visual Tagging...")
        
        for i, region in enumerate(regions):
            bbox = region['bbox'] # (x1, y1, x2, y2)
            
            # Draw Red Box
            draw.rectangle(bbox, outline="red", width=2)
            
            # Draw ID Label
            label = str(i + 1)
            # Draw background for label for readability
            text_bbox = draw.textbbox((bbox[0], bbox[1]), label, font=font)
            draw.rectangle(text_bbox, fill="red")
            draw.text((bbox[0], bbox[1]), label, fill="white", font=font)
            
            # Store ID in region for mapping back
            region['visual_id'] = i + 1

        # Save annotated image for debugging
        debug_path = os.path.join(config.get('output_dir', 'output'), "debug_annotated.jpg")
        os.makedirs(os.path.dirname(debug_path), exist_ok=True)
        annotated_image.save(debug_path)
        logger.info(f"Saved annotated image to {debug_path}")

        # 2. Construct Prompt
        system_prompt = (
            "You are an expert transcription system. "
            "I have annotated the image with red boxes and numbered labels (1, 2, 3...).\n"
            "Your task is to transcribe the text inside each numbered box.\n"
            "Return the output as a JSON object where keys are the box numbers (as strings) and values are the transcribed text.\n"
            "Example: {\"1\": \"Text in box 1\", \"2\": \"Text in box 2\"}\n"
            "Do not include any markdown formatting (like ```json). Just the raw JSON string."
        )
        
        full_prompt = [system_prompt, base_prompt, annotated_image]
        
        # 3. Call Gemini
        logger.info("Sending annotated image to Gemini...")
        try:
            gen_config = genai.types.GenerationConfig(
                temperature=0.0,
                response_mime_type="application/json" # Force JSON output
            )
            
            response = model.generate_content(
                full_prompt,
                generation_config=gen_config
            )
            
            raw_text = response.text.strip()
            # Clean up potential markdown code blocks if the model ignores the instruction
            if raw_text.startswith("```json"):
                raw_text = raw_text[7:]
            if raw_text.endswith("```"):
                raw_text = raw_text[:-3]
                
            transcription_map = json.loads(raw_text)
            logger.info("Received JSON response from Gemini.")
            
            # 4. Map Back to Regions
            for region in regions:
                visual_id = str(region['visual_id'])
                if visual_id in transcription_map:
                    region['text'] = transcription_map[visual_id]
                else:
                    logger.warning(f"Gemini did not return text for box {visual_id}")
                    region['text'] = ""
                    
        except Exception as e:
            logger.error(f"Gemini Transcription Failed: {e}")
            # Fallback: Mark all as error
            for region in regions:
                region['text'] = "[Error: Transcription Failed]"
                
        return regions
