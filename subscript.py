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
    if engine_type == 'google_vision':
        raise NotImplementedError("Google Vision Segmentation not yet implemented")
    elif engine_type == 'kraken':
        from modules.segmentation import KrakenSegmentation
        return KrakenSegmentation()
    else:
        logger.warning("No segmentation engine specified.")
        return None

def get_transcription_engine(config):
    engine_type = config.get('transcription', {}).get('engine', 'gemini')
    if engine_type == 'gemini':
        from modules.transcription import GeminiTranscription
        return GeminiTranscription()
    elif engine_type == 'google_vision':
        raise NotImplementedError("Google Vision Transcription not yet implemented")
    else:
        raise ValueError(f"Unknown transcription engine: {engine_type}")

def get_output_engine(config):
    from modules.output import UnifiedOutputEngine
    return UnifiedOutputEngine()

def main():
    parser = argparse.ArgumentParser(description="Subscript 2.0: Full-Page HTR Pipeline")
    parser.add_argument("input", nargs='+', help="Input image(s)")
    parser.add_argument("--config", default="config.yml", help="Path to config file")
    parser.add_argument("--segmentation", choices=['kraken', 'google_vision'], help="Override segmentation engine")
    args = parser.parse_args()

    config = load_config(args.config)
    
    # CLI Override
    if args.segmentation:
        if 'segmentation' not in config: config['segmentation'] = {}
        config['segmentation']['engine'] = args.segmentation
    
    # Initialize Engines
    try:
        segmentation_engine = get_segmentation_engine(config)
        transcription_engine = get_transcription_engine(config)
        output_engine = get_output_engine(config)
    except Exception as e:
        logger.error(f"Initialization failed: {e}")
        sys.exit(1)

    logger.info("Subscript 2.0 Initialized")
    logger.info(f"Processing {len(args.input)} files...")

    # Pipeline Loop
    for image_path in args.input:
        logger.info(f"Processing {image_path}...")
        
        try:
            # 1. Segmentation
            regions = segmentation_engine.analyze(image_path, config)
            
            # 2. Transcription
            from PIL import Image
            with Image.open(image_path) as im:
                regions = transcription_engine.transcribe(im, regions, config)
            
            # 3. Output Generation
            output_engine.generate(image_path, regions, config.get('output_dir', 'output'), config)
            
            # Temporary Output
            print(f"\n--- Results for {os.path.basename(image_path)} ---")
            for r in regions:
                print(r.get('text', ''))
                
        except Exception as e:
            logger.error(f"Failed to process {image_path}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
