import logging
import os
import json
import re
from typing import List, Dict, Any
from PIL import Image, ImageDraw, ImageFont
import google.generativeai as genai
from modules.interfaces import TranscriptionEngine

logger = logging.getLogger(__name__)

def otsu_threshold(histogram):
    """Computes Otsu's threshold from a histogram."""
    total = sum(histogram)
    sum_b = 0
    w_b = 0
    w_f = 0
    sum1 = sum(i * n for i, n in enumerate(histogram))
    max_var = 0.0
    threshold = 0

    for i in range(256):
        w_b += histogram[i]
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
        
        sum_b += i * histogram[i]
        m_b = sum_b / w_b
        m_f = (sum1 - sum_b) / w_f
        
        var_between = w_b * w_f * (m_b - m_f) ** 2
        
        if var_between > max_var:
            max_var = var_between
            threshold = i
            
    return threshold

def preprocess_image(image: Image.Image, config: Dict[str, Any]) -> Image.Image:
    """Applies preprocessing steps to the image based on config."""
    from PIL import ImageOps, ImageEnhance
    
    prep_config = config.get('transcription', {}).get('preprocessing', {})
    
    # 1. Resize (Constrain Dimensions)
    size_setting = prep_config.get('resize_image', 'large')
    max_dim = None
    if size_setting == 'large':
        max_dim = 3000
    elif size_setting == 'medium':
        max_dim = 2000
    elif size_setting == 'small':
        max_dim = 1000
        
    if max_dim:
        w, h = image.size
        if w > max_dim or h > max_dim:
            # Calculate new size maintaining aspect ratio
            ratio = min(max_dim / w, max_dim / h)
            new_size = (int(w * ratio), int(h * ratio))
            image = image.resize(new_size, Image.Resampling.LANCZOS)
            logger.info(f"Resized image to {new_size}")

    # 2. Grayscale (Required for binarization, or if requested)
    # We'll convert to grayscale if binarize is on, or if contrast is applied (usually better on L)
    # But let's stick to user request: "if that is required for binarization, then just make that a multistep conversion"
    # We will convert to 'L' if binarize is True.
    # If just contrast/invert, we can keep RGB or convert. 
    # Usually HTR models work fine with RGB, but 'L' is safer for processing.
    # Let's convert to 'L' if binarize is True.
    if prep_config.get('binarize', False):
         if image.mode != 'L':
             image = image.convert('L')

    # 3. Contrast
    contrast_factor = prep_config.get('contrast', 1.0)
    if contrast_factor != 1.0:
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(contrast_factor)
        logger.info(f"Applied contrast enhancement: {contrast_factor}")

    # 4. Binarize (Otsu)
    if prep_config.get('binarize', False):
        # Ensure grayscale
        if image.mode != 'L':
            image = image.convert('L')
        
        hist = image.histogram()
        thresh = otsu_threshold(hist)
        image = image.point(lambda p: 255 if p > thresh else 0)
        logger.info(f"Applied Otsu binarization (threshold: {thresh})")

    # 5. Invert
    if prep_config.get('invert', False):
        # ImageOps.invert works on RGB and L
        if image.mode == 'RGBA':
            # Handle transparency if needed, or drop alpha
            image = image.convert('RGB')
        image = ImageOps.invert(image)
        logger.info("Applied color inversion")
        
    return image

class GeminiTranscription(TranscriptionEngine):
    def __init__(self):
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set.")
        genai.configure(api_key=api_key)

    def transcribe(self, image: Image.Image, regions: List[Dict[str, Any]], config: Dict[str, Any]) -> tuple[List[Dict[str, Any]], Dict[str, int]]:
        gemini_config = config.get('transcription', {}).get('gemini', {})
        model_name = gemini_config.get('model', 'gemini-1.5-pro')
        base_prompt = gemini_config.get('prompt', 'Transcribe this text.')
        
        logger.info(f"Initializing Gemini model: {model_name}")
        model = genai.GenerativeModel(model_name)
        
        # 0. Preprocess Image
        image = preprocess_image(image, config)
        
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

        # Save annotated image
        prep_config = config.get('transcription', {}).get('preprocessing', {})
        if prep_config.get('debug_image', True):
            image_path = config.get('image_path')
            if image_path:
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                debug_filename = f"{base_name}-debug.jpg"
            else:
                debug_filename = "debug_annotated.jpg"
                
            debug_path = os.path.join(config.get('output_dir', 'output'), debug_filename)
            os.makedirs(os.path.dirname(debug_path), exist_ok=True)
            annotated_image.save(debug_path)
            logger.info(f"Saved debug image to {debug_path}")

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
            # Load generation config from settings
            user_gen_config = gemini_config.get('API_passthrough', {})
            
            # Ensure critical settings are preserved/set
            # We enforce JSON output for this pipeline to work
            user_gen_config['response_mime_type'] = "application/json"
            
            # Set default temperature if not provided
            if 'temperature' not in user_gen_config:
                user_gen_config['temperature'] = 0.0
                
            # Unpack into GenerationConfig
            gen_config = genai.types.GenerationConfig(**user_gen_config)
            
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
            
            # Extract usage metadata
            usage_metadata = {
                'prompt_token_count': response.usage_metadata.prompt_token_count,
                'candidates_token_count': response.usage_metadata.candidates_token_count
            }
                    
        except Exception as e:
            logger.error(f"Gemini Transcription Failed: {e}")
            # Fallback: Mark all as error
            for region in regions:
                region['text'] = "[Error: Transcription Failed]"
            usage_metadata = {'prompt_token_count': 0, 'candidates_token_count': 0}
                
        return regions, usage_metadata

# --- OpenAI Support ---
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

class OpenAITranscription(TranscriptionEngine):
    def __init__(self):
        if OpenAI is None:
            raise ImportError("OpenAI library not installed. Please run `pip install openai`.")
        
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set.")
        self.client = OpenAI(api_key=api_key)

    def transcribe(self, image: Image.Image, regions: List[Dict[str, Any]], config: Dict[str, Any]) -> tuple[List[Dict[str, Any]], Dict[str, int]]:
        openai_config = config.get('transcription', {}).get('openai', {})
        model_name = openai_config.get('model', 'gpt-4o')
        base_prompt = openai_config.get('prompt', 'Transcribe this text.')
        
        logger.info(f"Initializing OpenAI model: {model_name}")
        
        # 0. Preprocess Image
        image = preprocess_image(image, config)
        
        # 1. Annotate Image (Same as Gemini)
        annotated_image = image.copy()
        draw = ImageDraw.Draw(annotated_image)
        try:
            font = ImageFont.truetype("Arial.ttf", 20)
        except IOError:
            font = ImageFont.load_default()

        logger.info(f"Annotating {len(regions)} regions for Visual Tagging...")
        for i, region in enumerate(regions):
            bbox = region['bbox']
            draw.rectangle(bbox, outline="red", width=2)
            label = str(i + 1)
            text_bbox = draw.textbbox((bbox[0], bbox[1]), label, font=font)
            draw.rectangle(text_bbox, fill="red")
            draw.text((bbox[0], bbox[1]), label, fill="white", font=font)
            region['visual_id'] = i + 1
            
        # Save debug image
        prep_config = config.get('transcription', {}).get('preprocessing', {})
        if prep_config.get('debug_image', True):
            image_path = config.get('image_path')
            if image_path:
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                debug_filename = f"{base_name}-debug.jpg"
            else:
                debug_filename = "debug_annotated_openai.jpg"
            debug_path = os.path.join(config.get('output_dir', 'output'), debug_filename)
            os.makedirs(os.path.dirname(debug_path), exist_ok=True)
            annotated_image.save(debug_path)
            logger.info(f"Saved debug image to {debug_path}")
        
        # Encode image to base64
        import base64
        from io import BytesIO
        buffered = BytesIO()
        annotated_image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        # 2. Construct Prompt
        system_prompt = (
            "You are an expert transcription system. "
            "I have annotated the image with red boxes and numbered labels (1, 2, 3...).\n"
            "Your task is to transcribe the text inside each numbered box.\n"
            "Return the output as a JSON object where keys are the box numbers (as strings) and values are the transcribed text.\n"
            "Example: {\"1\": \"Text in box 1\", \"2\": \"Text in box 2\"}\n"
            "Do not include any markdown formatting (like ```json). Just the raw JSON string."
        )
        
        # 3. Call OpenAI
        logger.info("Sending annotated image to OpenAI...")
        try:
            # Map generation config
            user_gen_config = openai_config.get('API_passthrough', {})
            temperature = user_gen_config.get('temperature', 0.0)
            max_tokens = user_gen_config.get('max_output_tokens', 4096) # OpenAI uses max_tokens
            
            response = self.client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": [
                        {"type": "text", "text": base_prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_str}"}}
                    ]}
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"}
            )
            
            raw_text = response.choices[0].message.content
            transcription_map = json.loads(raw_text)
            logger.info("Received JSON response from OpenAI.")
            
            # 4. Map Back
            for region in regions:
                visual_id = str(region['visual_id'])
                if visual_id in transcription_map:
                    region['text'] = transcription_map[visual_id]
                else:
                    logger.warning(f"OpenAI did not return text for box {visual_id}")
                    region['text'] = ""
                    
            usage_metadata = {
                'prompt_token_count': response.usage.prompt_tokens,
                'candidates_token_count': response.usage.completion_tokens
            }
            
        except Exception as e:
            logger.error(f"OpenAI Transcription Failed: {e}")
            for region in regions:
                region['text'] = "[Error: Transcription Failed]"
            usage_metadata = {'prompt_token_count': 0, 'candidates_token_count': 0}
            
        return regions, usage_metadata

# --- Anthropic Support ---
try:
    import anthropic
except ImportError:
    anthropic = None

class AnthropicTranscription(TranscriptionEngine):
    def __init__(self):
        if anthropic is None:
            raise ImportError("Anthropic library not installed. Please run `pip install anthropic`.")
        
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set.")
        self.client = anthropic.Anthropic(api_key=api_key)

    def transcribe(self, image: Image.Image, regions: List[Dict[str, Any]], config: Dict[str, Any]) -> tuple[List[Dict[str, Any]], Dict[str, int]]:
        anthropic_config = config.get('transcription', {}).get('anthropic', {})
        model_name = anthropic_config.get('model', 'claude-3-5-sonnet-20240620')
        base_prompt = anthropic_config.get('prompt', 'Transcribe this text.')
        
        logger.info(f"Initializing Anthropic model: {model_name}")
        
        # 0. Preprocess Image
        image = preprocess_image(image, config)
        
        # 1. Annotate Image (Same as Gemini)
        annotated_image = image.copy()
        draw = ImageDraw.Draw(annotated_image)
        try:
            font = ImageFont.truetype("Arial.ttf", 20)
        except IOError:
            font = ImageFont.load_default()

        logger.info(f"Annotating {len(regions)} regions for Visual Tagging...")
        for i, region in enumerate(regions):
            bbox = region['bbox']
            draw.rectangle(bbox, outline="red", width=2)
            label = str(i + 1)
            text_bbox = draw.textbbox((bbox[0], bbox[1]), label, font=font)
            draw.rectangle(text_bbox, fill="red")
            draw.text((bbox[0], bbox[1]), label, fill="white", font=font)
            region['visual_id'] = i + 1
            
        # Save debug image
        prep_config = config.get('transcription', {}).get('preprocessing', {})
        if prep_config.get('debug_image', True):
            image_path = config.get('image_path')
            if image_path:
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                debug_filename = f"{base_name}-debug.jpg"
            else:
                debug_filename = "debug_annotated_anthropic.jpg"
            debug_path = os.path.join(config.get('output_dir', 'output'), debug_filename)
            os.makedirs(os.path.dirname(debug_path), exist_ok=True)
            annotated_image.save(debug_path)
            logger.info(f"Saved debug image to {debug_path}")
        
        # Encode image to base64
        import base64
        from io import BytesIO
        buffered = BytesIO()
        annotated_image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        # 2. Construct Prompt
        system_prompt = (
            "You are an expert transcription system. "
            "I have annotated the image with red boxes and numbered labels (1, 2, 3...).\n"
            "Your task is to transcribe the text inside each numbered box.\n"
            "Return the output as a JSON object where keys are the box numbers (as strings) and values are the transcribed text.\n"
            "Example: {\"1\": \"Text in box 1\", \"2\": \"Text in box 2\"}\n"
            "Do not include any markdown formatting (like ```json). Just the raw JSON string."
        )
        
        # 3. Call Anthropic
        logger.info("Sending annotated image to Anthropic...")
        try:
            # Map generation config
            user_gen_config = anthropic_config.get('API_passthrough', {})
            temperature = user_gen_config.get('temperature', 0.0)
            max_tokens = user_gen_config.get('max_output_tokens', 4096)
            
            message = self.client.messages.create(
                model=model_name,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_prompt,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": img_str,
                                },
                            },
                            {
                                "type": "text",
                                "text": base_prompt
                            }
                        ],
                    }
                ]
            )
            
            raw_text = message.content[0].text
            # Clean up JSON
            if raw_text.startswith("```json"):
                raw_text = raw_text[7:]
            if raw_text.endswith("```"):
                raw_text = raw_text[:-3]
                
            transcription_map = json.loads(raw_text)
            logger.info("Received JSON response from Anthropic.")
            
            # 4. Map Back
            for region in regions:
                visual_id = str(region['visual_id'])
                if visual_id in transcription_map:
                    region['text'] = transcription_map[visual_id]
                else:
                    logger.warning(f"Anthropic did not return text for box {visual_id}")
                    region['text'] = ""
                    
            usage_metadata = {
                'prompt_token_count': message.usage.input_tokens,
                'candidates_token_count': message.usage.output_tokens
            }
            
        except Exception as e:
            logger.error(f"Anthropic Transcription Failed: {e}")
            for region in regions:
                region['text'] = "[Error: Transcription Failed]"
            usage_metadata = {'prompt_token_count': 0, 'candidates_token_count': 0}
            
        return regions, usage_metadata
