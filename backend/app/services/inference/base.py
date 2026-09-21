from abc import ABC, abstractmethod
from typing import Any, Dict, List


class BaseInferenceEngine(ABC):
    """
    Abstract interface for object detection inference engines.
    Allows swappable implementations (Mock vs. PyTorch HydroWatch).
    """

    @property
    @abstractmethod
    def model_info(self) -> Dict[str, Any]:
        """Describe the active inference backend without exposing implementation details."""
        raise NotImplementedError

    @abstractmethod
    def detect_image(self, image_path: str, confidence_threshold: float = 0.5) -> List[Dict[str, Any]]:
        """
        Run inference on an image file and return detections list.
        Each detection dict should contain: id, label, confidence, bbox, size.
        """
        pass
