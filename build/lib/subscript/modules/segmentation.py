import logging
import os
from PIL import Image
from typing import List, Dict, Any
from modules.interfaces import SegmentationEngine

# Try importing kraken, handle missing dependency gracefully
try:
    from kraken import blla
    from kraken.lib import vgsl
except ImportError:
    blla = None
    vgsl = None

logger = logging.getLogger(__name__)

class KrakenSegmentation(SegmentationEngine):
    def __init__(self):
        if blla is None:
            logger.warning("Kraken not installed. KrakenSegmentation will fail if used.")

    def analyze(self, image_path: str, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        if blla is None:
            raise ImportError("Kraken is not installed. Please install it to use KrakenLayout.")

        logger.info(f"Segmenting {image_path} with Kraken...")
        
        # Load settings
        kraken_config = config.get('layout', {}).get('kraken', {})
        model_name = kraken_config.get('model', 'default')
        padding = kraken_config.get('padding', 0) # Pre-segmentation padding (if any)
        
        # Open Image
        im = Image.open(image_path)
        
        # Load Model
        # Note: We might want to cache the model in __init__ if reusing the engine
        if model_name == 'default':
             model_path = os.path.join(os.path.dirname(blla.__file__), 'blla.mlmodel')
        else:
             model_path = model_name
             
        # Try to show relative path for cleaner logging
        display_path = model_path
        try:
            rel_path = os.path.relpath(model_path)
            if not rel_path.startswith('..'):
                display_path = f"./{rel_path}"
        except ValueError:
            pass
            
        logger.info(f"Loading segmentation model from {display_path}...")
        model = vgsl.TorchVGSLModel.load_model(model_path)
        
        # Segment
        # TODO: Implement pre-segmentation padding if needed (from config)
        res = blla.segment(im, model=model)
        
        regions = []
        for i, line in enumerate(res.lines):
            # Kraken returns baseline and boundary (polygon)
            # We need to convert to our standard region format
            
            # Calculate BBox from boundary
            xs = [p[0] for p in line.boundary]
            ys = [p[1] for p in line.boundary]
            bbox = (min(xs), min(ys), max(xs), max(ys))
            
            regions.append({
                'id': f"line_{i}",
                'type': 'line',
                'bbox': bbox,
                'polygon': line.boundary,
                'baseline': line.baseline,
                'confidence': 1.0 # Kraken doesn't give confidence per line easily
            })
            
        logger.info(f"Found {len(regions)} lines.")
        return regions
