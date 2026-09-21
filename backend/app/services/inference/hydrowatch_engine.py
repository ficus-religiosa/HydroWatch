from typing import Any, Dict, List

from app.services.inference.base import BaseInferenceEngine


class HydroWatchInferenceEngine(BaseInferenceEngine):
    """Reserved adapter for the real model after its architecture is finalized."""

    def __init__(self) -> None:
        self._model_loaded = False

    @property
    def model_info(self) -> Dict[str, Any]:
        return {
            "name": "hydrowatch",
            "is_mock": False,
            "status": "not_implemented",
        }

    def detect_image(self, image_path: str, confidence_threshold: float = 0.5) -> List[Dict[str, Any]]:
        raise NotImplementedError(
            "The HydroWatch model adapter is not implemented. Define the model architecture, "
            "checkpoint loading, and preprocessing contract before enabling INFERENCE_BACKEND=hydrowatch."
        )