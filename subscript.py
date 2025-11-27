#!/usr/bin/env python3
import argparse
import os
import sys
import yaml
import logging
from dotenv import load_dotenv
from modules.interfaces import SegmentationEngine, TranscriptionEngine, OutputEngine

# Load environment variables
load_dotenv(override=True)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def load_config(config_path):
    if not os.path.exists(config_path):
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def get_segmentation_engine(config):
    engine_type = config.get('segmentation', {}).get('engine', 'kraken')
    if engine_type == 'kraken':
        from modules.segmentation import KrakenSegmentation
        return KrakenSegmentation()
    else:
        logger.warning(f"Unknown segmentation engine: {engine_type}")
        return None

import glob

def get_transcription_engine(config, model_nickname):
    engine_type = config.get('transcription', {}).get('engine', 'gemini')
    
    # Get specific model config
    models = config.get('transcription', {}).get('models', {})
    if model_nickname not in models:
        logger.error(f"Model '{model_nickname}' not found in config. Available models: {list(models.keys())}")
        sys.exit(1)
        
    model_config = models[model_nickname]
    
    # Inject model config into the main config structure expected by the engine
    # This is a bit of a hack to keep the engine signature simple for now
    # Ideally, we'd pass the specific config object
    if engine_type == 'gemini':
        from modules.transcription import GeminiTranscription
        # Create a temporary config with the selected model settings merged in
        # We'll pass this 'effective config' to the engine
        effective_config = config.copy()
        effective_config['transcription']['gemini'] = model_config
        return GeminiTranscription(), effective_config
    else:
        raise ValueError(f"Unknown transcription engine: {engine_type}")

def get_output_engine(config):
    from modules.output import UnifiedOutputEngine
    return UnifiedOutputEngine()

def main():
    parser = argparse.ArgumentParser(description="Subscript 2.0: Full-Page HTR Pipeline")
    parser.add_argument("model", help="Model nickname (defined in config.yml)")
    parser.add_argument("input", nargs='+', help="Input image(s) (supports globs)")
    parser.add_argument("--config", default="config.yml", help="Path to config file")
    parser.add_argument("--segmentation", help="Override segmentation engine (e.g., kraken)")
    parser.add_argument("--combine", help="Combine all output PDFs into a single file (e.g., combined.pdf)")
    args = parser.parse_args()

    config = load_config(args.config)
    
    # CLI Override
    if args.segmentation:
        if 'segmentation' not in config: config['segmentation'] = {}
        config['segmentation']['engine'] = args.segmentation
    
    # Expand Globs
    input_files = []
    for pattern in args.input:
        expanded = glob.glob(pattern)
        if not expanded:
            logger.warning(f"No files found for pattern: {pattern}")
        input_files.extend(expanded)
        
    if not input_files:
        logger.error("No input files found.")
        sys.exit(1)
    
    # Initialize Engines
    try:
        segmentation_engine = get_segmentation_engine(config)
        transcription_engine, effective_config = get_transcription_engine(config, args.model)
        output_engine = get_output_engine(config)
    except Exception as e:
        logger.error(f"Initialization failed: {e}")
        sys.exit(1)

    logger.info(f"Subscript 2.0 Initialized (Model: {args.model})")
    logger.info(f"Processing {len(input_files)} files...")

    # Pipeline Loop
    generated_pdfs = []
    
    for image_path in input_files:
        logger.info(f"Processing {image_path}...")
        
        try:
            # 1. Segmentation
            regions = segmentation_engine.analyze(image_path, config)
            
            # 2. Transcription
            from PIL import Image
            with Image.open(image_path) as im:
                # Pass effective_config which contains the selected model settings
                regions = transcription_engine.transcribe(im, regions, effective_config)
            
            # 3. Output Generation
            output_engine.generate(image_path, regions, config.get('output_dir', 'output'), config)
            
            # Collect PDF path if generated
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            pdf_path = os.path.join(config.get('output_dir', 'output'), f"{base_name}.pdf")
            if os.path.exists(pdf_path):
                generated_pdfs.append(pdf_path)
            
            # Temporary Output
            print(f"\n--- Results for {os.path.basename(image_path)} ---")
            for r in regions:
                print(r.get('text', ''))
                
        except Exception as e:
            logger.error(f"Failed to process {image_path}: {e}")
            import traceback
            traceback.print_exc()

    # Combine PDFs if requested
    if args.combine and generated_pdfs:
        logger.info(f"Combining {len(generated_pdfs)} PDFs into {args.combine}...")
        output_path = os.path.join(config.get('output_dir', 'output'), args.combine)
        output_engine.combine_pdfs(generated_pdfs, output_path)

if __name__ == "__main__":
    main()
