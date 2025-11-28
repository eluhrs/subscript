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

# Suppress CoreML warnings
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, module="coremltools")
# Suppress Google API FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning, module="google.api_core")

def load_config(config_path):
    if not os.path.exists(config_path):
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def get_segmentation_engine(config, model_nickname=None):
    # Default to 'default_segmentation' from config, or 'historical-manuscript' if missing
    if not model_nickname:
        model_nickname = config.get('segmentation', {}).get('default_segmentation', 'historical-manuscript')
    
    seg_config = config.get('segmentation', {}).get('models', {}).get(model_nickname, {})
    provider = seg_config.get('provider', 'kraken')
    
    if provider == 'kraken':
        from modules.segmentation import KrakenSegmentation
        return KrakenSegmentation()
    else:
        logger.warning(f"Unknown segmentation provider: {provider}")
        return None

import glob

def get_transcription_engine(config, model_nickname):
    # Get specific model config
    models = config.get('transcription', {}).get('models', {})
    if model_nickname not in models:
        logger.error(f"Model '{model_nickname}' not found in config. Available models: {list(models.keys())}")
        sys.exit(1)
        
    model_config = models[model_nickname]
    provider = model_config.get('provider', 'gemini')
    
    # Inject model config into the main config structure expected by the engine
    if provider == 'gemini':
        from modules.transcription import GeminiTranscription
        # Create a temporary config with the selected model settings merged in
        # We'll pass this 'effective config' to the engine
        effective_config = config.copy()
        # Ensure the structure exists for the engine to read
        if 'transcription' not in effective_config:
            effective_config['transcription'] = {}
        effective_config['transcription']['gemini'] = model_config
        return GeminiTranscription(), effective_config
    else:
        raise ValueError(f"Unknown transcription provider: {provider}")

def get_output_engine(config):
    from modules.output import UnifiedOutputEngine
    return UnifiedOutputEngine()

def main():
    parser = argparse.ArgumentParser(
        description="Subscript: Full-page manuscript image to searchable PDF pipeline\n\nusage: ./subscript.py [SEGMENTATION-NICKNAME] [MODEL-NICKNAME] INPUT [OPTIONS]",
        usage=argparse.SUPPRESS,
        formatter_class=argparse.RawTextHelpFormatter,
        add_help=False
    )
    
    # Positional arguments
    # We make 'model' optional and 'input' greedy.
    # However, argparse is tricky with optional positional followed by greedy positional.
    # Strategy: Parse known args first to get config path, load config, then manually inspect sys.argv
    
    parser.add_argument("--help", action="help", help="Show this help message and exit")
    parser.add_argument("--config", default="config.yml", metavar="", help="Path to alternate config file (default: ./config.yml)")
    parser.add_argument("--output", metavar="", help="Path to alternate output directory (default: ./output)")
    parser.add_argument("--combine", metavar="", help="Combine multiple input images into specified PDF filename")
    parser.add_argument("--prompt", metavar="", help="Override system prompt defined in config.xml")
    parser.add_argument("--temp", type=float, metavar="", help="Override the temperature defined in config.xml")
    
    # Add blank line before help output
    if '--help' in sys.argv or '-h' in sys.argv:
        print("")
        
    # We use parse_known_args to get the flags, leaving the positionals
    try:
        args, remaining_args = parser.parse_known_args()
    except SystemExit:
        # Add blank line after help output (or error)
        print("")
        raise

    config = load_config(args.config)
    
    # Determine Models and Input
    # 1. Get list of valid models
    valid_trans_models = list(config.get('transcription', {}).get('models', {}).keys())
    valid_seg_models = list(config.get('segmentation', {}).get('models', {}).keys())
    
    default_trans_model = config.get('transcription', {}).get('default_model')
    default_seg_model = config.get('segmentation', {}).get('default_segmentation')
    
    selected_trans_model = default_trans_model
    selected_seg_model = default_seg_model
    input_patterns = []
    
    if not remaining_args:
        if '--help' not in sys.argv and '-h' not in sys.argv:
            logger.error("No input files provided.")
            sys.exit(1)
        else:
            sys.exit(0)

    # Parsing Logic
    # Check arg 0
    arg0 = remaining_args[0]
    
    # Case A: arg0 is Segmentation Model
    if arg0 in valid_seg_models:
        selected_seg_model = arg0
        
        # Check arg 1 for Transcription Model
        if len(remaining_args) > 1:
            arg1 = remaining_args[1]
            if arg1 in valid_trans_models:
                selected_trans_model = arg1
                input_patterns = remaining_args[2:]
            else:
                # arg1 is likely input
                input_patterns = remaining_args[1:]
        else:
            # Only one arg provided, and it was a seg model? That means no input.
            logger.error("No input files provided.")
            sys.exit(1)
            
    # Case B: arg0 is Transcription Model (and NOT a segmentation model, or we prioritize seg if overlap? 
    # Assuming distinct names for now as per plan)
    elif arg0 in valid_trans_models:
        selected_trans_model = arg0
        input_patterns = remaining_args[1:]
        
    # Case C: arg0 is Input
    else:
        input_patterns = remaining_args

    if not input_patterns:
        logger.error("No input files provided.")
        sys.exit(1)

    # CLI Override
    if args.output:
        config['output_dir'] = args.output
        
    # Model-specific overrides
    if selected_trans_model in config.get('transcription', {}).get('models', {}):
        model_config = config['transcription']['models'][selected_trans_model]
        
        if args.prompt:
            model_config['prompt'] = args.prompt
            
        if args.temp is not None:
            if 'generation_config' not in model_config:
                model_config['generation_config'] = {}
            model_config['generation_config']['temperature'] = args.temp
    
    # Expand Globs
    input_files = []
    for pattern in input_patterns:
        expanded = glob.glob(pattern)
        if not expanded:
            logger.warning(f"No files found for pattern: {pattern}")
        input_files.extend(expanded)
        
    if not input_files:
        logger.error("No input files found.")
        sys.exit(1)
    
    # Initialize Engines
    try:
        segmentation_engine = get_segmentation_engine(config, selected_seg_model)
        transcription_engine, effective_config = get_transcription_engine(config, selected_trans_model)
        output_engine = get_output_engine(config)
    except Exception as e:
        logger.error(f"Initialization failed: {e}")
        sys.exit(1)

    logger.info(f"\nSubscript.py initialized segmentation provider: {selected_seg_model}")
    logger.info(f"Subscript.py Initialized transcription provider: {selected_trans_model}")
    logger.info(f"Processing {len(input_files)} files...")

    # Pipeline Loop
    generated_pdfs = []
    import time
    import datetime
    
    total_start_time = time.time()
    total_input_tokens = 0
    total_output_tokens = 0
    total_cost = 0.0
    
    for i, image_path in enumerate(input_files):
        page_start_time = time.time()
        logger.info(f"Processing {image_path}...")
        
        try:
            # 1. Segmentation
            seg_start = time.time()
            regions = segmentation_engine.analyze(image_path, config)
            seg_duration = time.time() - seg_start
            
            # 2. Transcription
            trans_start = time.time()
            from PIL import Image
            with Image.open(image_path) as im:
                # Pass effective_config which contains the selected model settings
                # Inject image_path for output naming
                effective_config['image_path'] = image_path
                regions, usage = transcription_engine.transcribe(im, regions, effective_config)
            trans_duration = time.time() - trans_start
            
            # 3. Output Generation
            out_start = time.time()
            output_engine.generate(image_path, regions, config.get('output_dir', 'output'), config)
            out_duration = time.time() - out_start
            
            # Collect PDF path if generated
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            pdf_path = os.path.join(config.get('output_dir', 'output'), f"{base_name}.pdf")
            if os.path.exists(pdf_path):
                generated_pdfs.append(pdf_path)
            
            # Calculate Page Time
            page_end_time = time.time()
            page_duration = page_end_time - page_start_time
            
            # Calculate Costs
            input_tokens = usage.get('prompt_token_count', 0)
            output_tokens = usage.get('candidates_token_count', 0)
            
            # Get cost rates from the specific model config we used
            # effective_config has the merged settings
            model_config = effective_config.get('transcription', {}).get('gemini', {})
            cost_config = model_config.get('cost_config', {})
            input_cost_rate = cost_config.get('input_token_cost', 0.0)
            output_cost_rate = cost_config.get('output_token_cost', 0.0)
            
            page_cost = (input_tokens * input_cost_rate) + (output_tokens * output_cost_rate)
            
            # Accumulate Totals
            total_input_tokens += input_tokens
            total_output_tokens += output_tokens
            total_cost += page_cost
            
            # Formatted Output
            print(f"\n------------- {selected_trans_model} output for {os.path.basename(image_path)} -------------")
            for r in regions:
                print(r.get('text', ''))
            print("----------------------------------------------------------------")
            print("") # Blank line
            
            # Saved messages (Reordered: XML, TXT, PDF)
            xml_path = os.path.join(config.get('output_dir', 'output'), f"{base_name}.xml")
            if os.path.exists(xml_path):
                print(f"Saved PageXML to {xml_path}")
            
            print(f"Saved TXT to {os.path.join(config.get('output_dir', 'output'), f'{base_name}.txt')}")
            
            if os.path.exists(pdf_path):
                print(f"Saved PDF to {pdf_path}")

            print("") # Blank line
            
            # Format time as HH:MM:SS
            # datetime.timedelta string format is usually [D day[s], ][H]H:MM:SS[.UUUUUU]
            # We want to force 00:00:00 if it's short, but timedelta default str is okay for >1h.
            # For <1h it is H:MM:SS. User asked for 00:00:00 (HH:MM:SS).
            # Let's write a helper or just use strftime logic manually.
            def format_timedelta(seconds):
                m, s = divmod(int(seconds), 60)
                h, m = divmod(m, 60)
                return f"{h:02d}:{m:02d}:{s:02d}"

            print(f"Time:   {format_timedelta(page_duration)}")
            print(f"Input:  {input_tokens} tokens")
            print(f"Output: {output_tokens} tokens")
            print(f"Cost:   ${page_cost:.4f}")
            
            # Multi-page Totals (only on last page if multiple files)
            if len(input_files) > 1 and i == len(input_files) - 1:
                total_end_time = time.time()
                total_duration = total_end_time - total_start_time
                
                print("") # Blank line
                print(f"Total time:   {format_timedelta(total_duration)}")
                print(f"Total input:  {total_input_tokens} tokens")
                print(f"Total output: {total_output_tokens} tokens")
                print(f"Total cost:   ${total_cost:.4f}")
            
            print("") # Blank line
            print("----------------------------------------------------------------")
            print("") # Blank line at end
                
        except Exception as e:
            logger.error(f"Failed to process {image_path}: {e}")
            import traceback
            traceback.print_exc()

    # Combine PDFs if requested
    if args.combine and generated_pdfs:
        logger.info(f"Combining {len(generated_pdfs)} PDFs into {args.combine}...")
        output_path = os.path.join(config.get('output_dir', 'output'), args.combine)
        output_engine.combine_pdfs(generated_pdfs, output_path)
        # Note: We already printed totals above if multi-page.
        # If combine is used, it's usually multi-page.
        # The user didn't specify where to put combine logs.
        # I'll leave them as logger.info which goes to stderr/log.

if __name__ == "__main__":
    main()
