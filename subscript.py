#!/usr/bin/env python3
import argparse
import os
import sys
import json
import yaml
import warnings
from dotenv import load_dotenv
from PIL import Image

# Load environment variables from .env file
load_dotenv()

# Suppress annoying CoreML warnings from Kraken/PyTorch on macOS
warnings.filterwarnings("ignore", category=RuntimeWarning, module="coremltools")

# Suppress Google API Python version warning unless debugging
if os.environ.get("LOG_LEVEL", "INFO").upper() != "DEBUG":
    warnings.filterwarnings("ignore", category=FutureWarning, module="google.api_core.*")
import google.generativeai as genai
from lxml import etree
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

# Try importing kraken, handle missing dependency gracefully for now
try:
    from kraken import blla
except ImportError:
    blla = None

def setup_args():
    parser = argparse.ArgumentParser(description="Manuscript OCR Pipeline")
    parser.add_argument("input_image", nargs='+', help="Path to input image(s) or directory")
    parser.add_argument("--model", required=True, help="Gemini model name or nickname from models.yml")
    parser.add_argument("--api-key", help="Google API Key (or set GEMINI_API_KEY env var)")
    parser.add_argument("--output-dir", default="output", help="Directory to save output")
    parser.add_argument("--context-lines", type=int, default=1, help="Number of previous lines to use as context")
    parser.add_argument("--models-file", default="models.yml", help="Path to models configuration file")
    parser.add_argument("--combine", help="Combine all inputs into a single PDF/TXT with this filename")
    
    # Overrides
    parser.add_argument("--prompt", help="Override prompt from models.yml")
    parser.add_argument("--temperature", type=float, help="Override temperature from models.yml")
    
    return parser.parse_args()

def load_config(args):
    """
    Loads configuration from models.yml and merges with CLI args.
    Returns a dict with 'model', 'prompt', 'temperature'.
    """
    config = {
        'model': args.model,
        'prompt': "Transcribe this handwritten text line exactly as written.",
        'temperature': 0.0
    }

    # 1. Load from YAML if exists
    if os.path.exists(args.models_file):
        try:
            with open(args.models_file, 'r') as f:
                data = yaml.safe_load(f)
                models = data.get('models', {})
                
                if args.model in models:
                    logging.info(f"Using configuration for '{args.model}' from {args.models_file}")
                    model_conf = models[args.model]
                    config['model'] = model_conf.get('model', args.model)
                    config['prompt'] = model_conf.get('prompt', config['prompt'])
                    config['temperature'] = model_conf.get('temperature', config['temperature'])
        except Exception as e:
            logging.warning(f"Warning: Failed to read {args.models_file}: {e}")

    # 2. Override with CLI args
    if args.prompt:
        config['prompt'] = args.prompt
    if args.temperature is not None:
        config['temperature'] = args.temperature
        
    return config

def segment_image(image_path):
    """
    Segments the image using Kraken.
    Returns a list of lines. Each line is a dict with 'baseline', 'boundary', 'text' (empty).
    """
    if blla is None:
        logging.warning("Kraken not installed. Falling back to mock segmentation.")
        return mock_segment_image(image_path)

    logging.info(f"Segmenting {image_path}...")
    try:
        im = Image.open(image_path)
        
        # Manually load default model to avoid importlib bug in kraken 4.x
        from kraken.lib import vgsl
        model_path = os.path.join(os.path.dirname(blla.__file__), 'blla.mlmodel')
        logging.info(f"Loading segmentation model from {model_path}...")
        model = vgsl.TorchVGSLModel.load_model(model_path)
        
        res = blla.segment(im, model=model)
        
        # Convert Kraken Segmentation object to list of dicts
        lines = []
        for line in res.lines:
            lines.append({
                'baseline': line.baseline,
                'boundary': line.boundary,
                'text': ''
            })
        return lines
    except Exception as e:
        logging.error(f"Kraken segmentation failed: {e}. Falling back to mock.")
        return mock_segment_image(image_path)

def draw_page_on_canvas(c, lines, image_path):
    """
    Draws the image and invisible text onto the current PDF canvas page.
    Does NOT save the canvas.
    """
    # Draw image
    with Image.open(image_path) as im:
        w, h = im.size
    
    c.setPageSize((w, h))
    c.drawImage(image_path, 0, 0, width=w, height=h)
    
    # Draw invisible text
    c.setFillColorRGB(0, 0, 0, 0) # Invisible
    
    for line in lines:
        text = line.get('text', '')
        if not text:
            continue
            
        # Calculate bounding box from boundary polygon
        xs = [p[0] for p in line['boundary']]
        ys = [p[1] for p in line['boundary']]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        
        box_width = max_x - min_x
        box_height = max_y - min_y
        
        # ReportLab coordinates are bottom-up, Image is top-down
        pdf_y = h - max_y 
        
        text_object = c.beginText()
        text_object.setTextRenderMode(3) # Invisible text
        font_size = box_height * 0.8
        text_object.setFont("Helvetica", font_size)
        text_object.setTextOrigin(min_x, pdf_y)
        
        # Calculate text width and stretch to fit box_width
        text_width = c.stringWidth(text, "Helvetica", font_size)
        if text_width > 0 and box_width > 0:
            scale = (box_width / text_width) * 100
            text_object.setHorizScale(scale)
        
        text_object.textLine(text)
        c.drawText(text_object)

def save_pdf(lines, image_path, output_path):
    """
    Generates a searchable PDF.
    """
    c = canvas.Canvas(output_path)
    draw_page_on_canvas(c, lines, image_path)
    c.save()
    logging.info(f"Saved PDF to {output_path}")

def detect_repetition(text, threshold=0.2):
    """
    Detects if the text contains excessive repetition.
    Simple heuristic: if the compressed length (zlib) is significantly smaller than original,
    or if specific patterns repeat.
    """
    if len(text) < 50:
        return False
        
    # Check for character repetition (e.g. "I. I. I.")
    # If > 30% of the text is just one or two characters repeated
    from collections import Counter
    counts = Counter(text)
    most_common = counts.most_common(1)
    if most_common and (most_common[0][1] / len(text)) > 0.5:
        return True
        
    # Check for substring repetition
    import zlib
    compressed = zlib.compress(text.encode('utf-8'))
    ratio = len(compressed) / len(text)
    if ratio < threshold:
        return True
        
    return False

def transcribe_line(model, image_slice, base_prompt, temperature, context="", retry_count=0):
    """
    Sends the image slice to Gemini for transcription.
    """
    full_prompt = base_prompt
    if context:
        full_prompt += f" The previous line read: '{context}'. Use this context to resolve ambiguous characters."
    
    if retry_count > 0:
        full_prompt += " WARNING: You previously produced repetitive garbage. Do NOT repeat text. Output ONLY the transcription."
        
    generation_config = genai.types.GenerationConfig(
        temperature=temperature,
        max_output_tokens=1000 # Safe high limit
    )

    try:
        response = model.generate_content(
            [full_prompt, image_slice],
            generation_config=generation_config
        )
        # Check if response has text
        if response.candidates and response.candidates[0].content.parts:
            text = response.text.strip()
            return text
        else:
            finish_reason = response.candidates[0].finish_reason if response.candidates else 'Unknown'
            print(f"Warning: Empty response from model (Finish Reason: {finish_reason})")
            
            # If Finish Reason is 2 (MAX_TOKENS), it likely looped. Retry!
            if str(finish_reason) == "2" or str(finish_reason) == "FinishReason.MAX_TOKENS":
                if retry_count == 0:
                    print("  -> Retrying with high temperature (0.8) to break loop...")
                    return transcribe_line(model, image_slice, base_prompt, 0.8, context, retry_count + 1)
                elif retry_count == 1:
                    print("  -> Retrying without context...")
                    return transcribe_line(model, image_slice, base_prompt, 0.8, "", retry_count + 1)
                else:
                    print("  -> Failed after retries. Marking as [Unreadable].")
                    return "[Unreadable]"
            
            return "[Unreadable]" # Return unreadable for other failures too, to keep line count
    except Exception as e:
        print(f"Error transcribing line: {e}")
        return "[Unreadable]"

def save_page_xml(lines, image_path, output_path):
    """
    Generates PAGE XML from the lines.
    """
    # Basic PAGE XML structure
    NSMAP = {None: "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"}
    root = etree.Element("PcGts", nsmap=NSMAP)
    metadata = etree.SubElement(root, "Metadata")
    etree.SubElement(metadata, "Creator").text = "ManuscriptOCR"
    etree.SubElement(metadata, "Created").text = "2024-01-01T00:00:00" # Placeholder
    
    page = etree.SubElement(root, "Page")
    page.set("imageFilename", os.path.basename(image_path))
    
    # Get image size
    with Image.open(image_path) as im:
        w, h = im.size
    page.set("imageWidth", str(w))
    page.set("imageHeight", str(h))
    
    text_region = etree.SubElement(page, "TextRegion")
    text_region.set("id", "region_0")
    
    for i, line in enumerate(lines):
        text_line = etree.SubElement(text_region, "TextLine")
        text_line.set("id", f"line_{i}")
        
        # Coords
        coords = etree.SubElement(text_line, "Coords")
        # Convert boundary list of tuples [(x,y),...] to string "x,y x,y ..."
        points = " ".join([f"{int(p[0])},{int(p[1])}" for p in line['boundary']])
        coords.set("points", points)
        
        # Baseline
        baseline = etree.SubElement(text_line, "Baseline")
        points = " ".join([f"{int(p[0])},{int(p[1])}" for p in line['baseline']])
        baseline.set("points", points)
        
        # Text
        text_equiv = etree.SubElement(text_line, "TextEquiv")
        unicode_text = etree.SubElement(text_equiv, "Unicode")
        unicode_text.text = line.get('text', '')

    tree = etree.ElementTree(root)
    tree.write(output_path, pretty_print=True, xml_declaration=True, encoding="utf-8")

import logging

# Configure logging
def setup_logging():
    log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format='%(message)s' # Keep it clean, just the message
    )

def save_text(lines, output_path):
    """
    Saves plain text output.
    """
    with open(output_path, "w", encoding="utf-8") as f:
        for line in lines:
            text = line.get('text', '')
            if text:
                f.write(text + "\n")
    logging.info(f"Saved Text to {output_path}")

def process_image_data(image_path, model, args, config):
    """
    Segments and transcribes an image. Returns lines.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Processing {image_path}...")
    
    # 1. Segment
    try:
        lines = segment_image(image_path)
    except Exception as e:
        logger.error(f"Segmentation failed for {image_path}: {e}")
        return None

    logger.info(f"Found {len(lines)} lines in {os.path.basename(image_path)}.")
    
    # 2. Transcribe
    im = Image.open(image_path)
    history = [] 
    
    print(f"\n--- Transcription Output ({os.path.basename(image_path)}) ---")
    
    for i, line in enumerate(lines):
        # Crop line
        xs = [p[0] for p in line['boundary']]
        ys = [p[1] for p in line['boundary']]
        box = (min(xs), min(ys), max(xs), max(ys))
        
        line_im = im.crop(box).convert("RGB")
        
        # Get context
        context = ""
        if i > 0:
            # Get last N lines
            start = max(0, len(history) - args.context)
            context = " ".join(history[start:])
            
        text = transcribe_line(model, line_im, config['prompt'], config['temperature'], context)
        
        line['text'] = text
        history.append(text)
        print(text) # Print actual text to stdout
        
    return lines

def setup_args():
    parser = argparse.ArgumentParser(
        description="Handwritten input images are segmented, transcribed, and saved as a searchable PDFs.",
        usage="./subscript.py model input [options]",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    # Positional arguments
    parser.add_argument("model", help="Model nickname defined in models.yml")
    parser.add_argument("input", nargs='+', help="Path to input image(s) or directory")
    
    # Options
    parser.add_argument("--output-dir", metavar="DIR", default="output", help="Directory for output files. Default: ./output")
    parser.add_argument("--combine", metavar="FILE", help="Combine all inputs into the specified output filename.")
    parser.add_argument("--context", metavar="NUM", type=int, default=5, help="Set number of transcript lines used as context. Default: 5")
    parser.add_argument("--prompt", metavar="TEXT", help="Set custom prompt (overrides value in models.yml).")
    parser.add_argument("--temp", metavar="FLOAT", type=float, help="Set temperature (overrides value in models.yml).")
    
    return parser.parse_args()

def load_config(args):
    """
    Loads configuration from models.yml and merges with CLI args.
    Returns a dict with 'model', 'prompt', 'temperature'.
    """
    config = {
        'model': args.model,
        'prompt': "Transcribe this handwritten text line exactly as written.",
        'temperature': 0.0
    }

    # 1. Load from YAML (hardcoded to models.yml)
    models_file = "models.yml"
    if os.path.exists(models_file):
        try:
            with open(models_file, 'r') as f:
                data = yaml.safe_load(f)
                models = data.get('models', {})
                
                if args.model in models:
                    logging.info(f"Using configuration for '{args.model}' from {models_file}")
                    model_conf = models[args.model]
                    config['model'] = model_conf.get('model', args.model)
                    config['prompt'] = model_conf.get('prompt', config['prompt'])
                    config['temperature'] = model_conf.get('temperature', config['temperature'])
        except Exception as e:
            logging.warning(f"Warning: Failed to read {models_file}: {e}")

    # 2. Override with CLI args
    if args.prompt:
        config['prompt'] = args.prompt
    if args.temp is not None:
        config['temperature'] = args.temp
        
    return config

# ... (load_config, segment_image, draw_page_on_canvas, save_pdf, detect_repetition, transcribe_line, save_page_xml, setup_logging, save_text, process_image_data remain same) ...

def main():
    setup_logging()
    logger = logging.getLogger(__name__)
    
    args = setup_args()
    config = load_config(args)
    
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        logger.error("Error: GEMINI_API_KEY environment variable not set.")
        sys.exit(1)
        
    genai.configure(api_key=api_key)
    
    logger.info(f"Initializing model: {config['model']}")
    model = genai.GenerativeModel(config['model'])
    
    # Collect all images
    image_files = []
    for input_path in args.input:
        if os.path.isdir(input_path):
            # Add all images in directory
            for root, _, files in os.walk(input_path):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
                        image_files.append(os.path.join(root, file))
        elif os.path.isfile(input_path):
            image_files.append(input_path)
        else:
            # Handle glob patterns
            import glob
            matches = glob.glob(input_path)
            if matches:
                image_files.extend(matches)
            else:
                logger.warning(f"Input not found: {input_path}")

    if not image_files:
        logger.error("No valid image files found.")
        sys.exit(1)

    # Sort files to ensure correct order
    image_files.sort()
    logger.info(f"Processing {len(image_files)} images...")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.combine:
        # Use provided filename
        combined_name = args.combine
        # Strip extension if provided, to use for both PDF and TXT
        if combined_name.lower().endswith(('.pdf', '.txt')):
            combined_name = os.path.splitext(combined_name)[0]
            
        combined_pdf_path = os.path.join(args.output_dir, f"{combined_name}.pdf")
        combined_txt_path = os.path.join(args.output_dir, f"{combined_name}.txt")
        
        c = canvas.Canvas(combined_pdf_path)
        logger.info(f"Combining outputs into {combined_pdf_path} and {combined_txt_path}")
        
        all_lines = []
        
        for image_path in image_files:
            lines = process_image_data(image_path, model, args, config)
            if lines:
                # Save individual XML (optional but good for debugging/data)
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                xml_path = os.path.join(args.output_dir, f"{base_name}.xml")
                save_page_xml(lines, image_path, xml_path)
                
                # Add to PDF
                draw_page_on_canvas(c, lines, image_path)
                c.showPage()
                
                # Collect lines for combined text
                all_lines.extend(lines)
        
        c.save()
        save_text(all_lines, combined_txt_path)
        logger.info(f"Saved Combined PDF to {combined_pdf_path}")
        logger.info(f"Saved Combined Text to {combined_txt_path}")
        
    else:
        # Process individually
        for image_path in image_files:
            lines = process_image_data(image_path, model, args, config)
            if lines:
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                
                xml_path = os.path.join(args.output_dir, f"{base_name}.xml")
                save_page_xml(lines, image_path, xml_path)
                logger.info(f"Saved PAGE XML to {xml_path}")
                
                pdf_path = os.path.join(args.output_dir, f"{base_name}.pdf")
                save_pdf(lines, image_path, pdf_path)
                
                txt_path = os.path.join(args.output_dir, f"{base_name}.txt")
                save_text(lines, txt_path)

if __name__ == "__main__":
    main()
