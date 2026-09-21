from app.core.config import settings
from app.services.inference.base import BaseInferenceEngine
from app.services.inference.hydrowatch_engine import HydroWatchInferenceEngine
from app.services.inference.mock_engine import MockInferenceEngine


_engine: BaseInferenceEngine | None = None


def get_inference_engine() -> BaseInferenceEngine:
    """Create the configured engine once per process."""
    global _engine
    if _engine is None:
        backend = settings.INFERENCE_BACKEND.lower().strip()
        if backend == "mock":
            _engine = MockInferenceEngine()
        elif backend == "hydrowatch":
            _engine = HydroWatchInferenceEngine()
        else:
            raise ValueError("INFERENCE_BACKEND must be 'mock' or 'hydrowatch'.")
    return _engine