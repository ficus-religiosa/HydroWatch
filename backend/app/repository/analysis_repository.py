from abc import ABC, abstractmethod
import json
import os
import tempfile
import threading
import time
from pathlib import Path
from typing import Optional

from app.schemas.analysis import AnalysisResponse


class AnalysisRepository(ABC):
    @abstractmethod
    def get(self, analysis_id: str) -> Optional[AnalysisResponse]:
        raise NotImplementedError

    @abstractmethod
    def save(self, analysis: AnalysisResponse) -> None:
        raise NotImplementedError

    @abstractmethod
    def list_all(self) -> list[AnalysisResponse]:
        raise NotImplementedError


class FileAnalysisRepository(AnalysisRepository):
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    def _path(self, analysis_id: str) -> Path:
        path = (self.root / analysis_id / "result.json").resolve()
        if self.root not in path.parents:
            raise ValueError("Invalid analysis identifier")
        return path

    def get(self, analysis_id: str) -> Optional[AnalysisResponse]:
        with self._lock:
            path = self._path(analysis_id)
            if not path.exists():
                return None
            try:
                return AnalysisResponse.model_validate(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, ValueError, json.JSONDecodeError):
                return None

    def save(self, analysis: AnalysisResponse) -> None:
        with self._lock:
            path = self._path(analysis.analysis_id)
            path.parent.mkdir(parents=True, exist_ok=True)
            fd, temp_name = tempfile.mkstemp(prefix="result-", suffix=".json", dir=path.parent)
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as file:
                    json.dump(analysis.model_dump(mode="json"), file, indent=2)
                    file.flush()
                    os.fsync(file.fileno())
                for attempt in range(3):
                    try:
                        os.replace(temp_name, path)
                        break
                    except PermissionError:
                        if attempt == 2:
                            raise
                        time.sleep(0.05 * (attempt + 1))
            finally:
                if os.path.exists(temp_name):
                    os.unlink(temp_name)

    def list_all(self) -> list[AnalysisResponse]:
        records = []
        for path in self.root.glob("*/result.json"):
            analysis = self.get(path.parent.name)
            if analysis:
                records.append(analysis)
        return records