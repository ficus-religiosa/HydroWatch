from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(default="ok", description="Operational status of the service")
    service: str = Field(default="HydroWatch Backend", description="Service name")
    version: str = Field(default="0.1.0", description="Service version")
