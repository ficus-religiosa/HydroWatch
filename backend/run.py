"""Start the HydroWatch backend:  python run.py"""
import os
import sys
from pathlib import Path

import uvicorn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from app.config import settings  # noqa: E402,F401  (loads backend/.env into the environment)

if __name__ == "__main__":
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app.main:app", host=host, port=port, workers=1)
