"""ANGGOTA 1 - Skema request/response (Pydantic)."""
from typing import Literal, Optional
from pydantic import BaseModel, Field


class ReadingIn(BaseModel):
    machine_id: str = Field(..., examples=["M-01"])
    product_id: Optional[str] = None
    type: Literal["L", "M", "H"] = "L"
    air_temp: float = Field(..., ge=250, le=350, description="Kelvin")
    process_temp: float = Field(..., ge=250, le=400, description="Kelvin")
    rpm: float = Field(..., ge=0, le=5000)
    torque: float = Field(..., ge=0, le=200, description="Nm")
    tool_wear: float = Field(..., ge=0, le=500, description="menit")
    actual_failure: Optional[int] = Field(None, ge=0, le=1)
