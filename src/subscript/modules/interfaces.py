from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple
from PIL import Image

class SegmentationEngine(ABC):
    """
    Abstract base class for segmentation engines (e.g., Kraken, Google Vision).
    Responsible for detecting text regions/lines in an image.
    """
    @abstractmethod
    def analyze(self, image_path: str, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Analyzes the image and returns a list of text regions.
        
        Args:
            image_path: Path to the image file.
            config: Configuration dictionary.
            
        Returns:
            List of dicts, where each dict represents a region (line/block) and contains:
            - 'bbox': (x1, y1, x2, y2)
            - 'polygon': List of (x, y) tuples (optional)
            - 'type': 'line', 'block', etc.
            - 'confidence': float (optional)
        """
        pass

class TranscriptionEngine(ABC):
    """
    Abstract base class for transcription engines (e.g., Gemini, Google Vision).
    Responsible for converting image regions to text.
    """
    @abstractmethod
    def transcribe(self, image: Image.Image, regions: List[Dict[str, Any]], config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Transcribes the text in the given regions.
        
        Args:
            image: The full PIL Image object.
            regions: List of region dicts (from SegmentationEngine).
            config: Configuration dictionary.
            
        Returns:
            The input list of regions, updated with a 'text' field.
        """
        pass

class OutputEngine(ABC):
    """
    Abstract base class for output generation.
    Responsible for creating PDF, TXT, XML files.
    """
    @abstractmethod
    def generate(self, image_path: str, regions: List[Dict[str, Any]], output_dir: str, config: Dict[str, Any]):
        """
        Generates output files from the image and transcribed regions.
        
        Args:
            image_path: Path to the source image.
            regions: List of region dicts containing 'text' and coordinates.
            output_dir: Directory to save outputs.
            config: Configuration dictionary.
        """
        pass
