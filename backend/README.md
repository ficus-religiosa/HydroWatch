# HydroWatch Backend

FastAPI backend service for marine debris detection, video analysis, pollution hotspot mapping, and environmental reporting.

## Setup & Running

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Environment Configuration**:
   ```bash
   cp .env.example .env
   ```

3. **Run the Development Server**:
   ```bash
   python run.py
   ```
   Or using Uvicorn directly:
   ```bash
   uvicorn app.main:app --reload --port 8000
   ```

4. **API Documentation**:
   - Interactive Swagger UI: `http://localhost:8000/api/v1/docs`
   - ReDoc: `http://localhost:8000/api/v1/redoc`
   - Health Check: `http://localhost:8000/api/v1/health`
