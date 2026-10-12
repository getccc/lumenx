from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple


class ImageGenModel(ABC):
    """Abstract base class for image generation models.

    Lives here rather than in ``image.py`` so a provider adapter can
    subclass it without importing that module's DashScope SDK dependency.
    ``image.py`` re-exports this name for backward compatibility.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    @abstractmethod
    def generate(self, prompt: str, output_path: str, **kwargs) -> Tuple[str, float]:
        """
        Generates an image from a prompt.

        Args:
            prompt: The input text prompt.
            output_path: The path to save the generated image.
            **kwargs: Additional arguments.

        Returns:
            A tuple containing:
            - The path to the generated image file.
            - The duration of the API generation process in seconds.
        """
        pass


class VideoGenModel(ABC):
    """Abstract base class for video generation models."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config

    @abstractmethod
    def generate(self, prompt: str, output_path: str, **kwargs) -> Tuple[str, float]:
        """
        Generates a video from a prompt.
        
        Args:
            prompt: The input text prompt.
            output_path: The path to save the generated video.
            **kwargs: Additional arguments.
            
        Returns:
            A tuple containing:
            - The path to the generated video file.
            - The duration of the API generation process in seconds (excluding download).
        """
        pass
