from typing import Literal, Optional
from pydantic import BaseModel, Field


ModeType = Literal["natural", "landcover", "geology", "water"]


class ColorizeRequest(BaseModel):
    mode: ModeType = "natural"
    enable_uncertainty: bool = True
    gamma: float = Field(default=1.0, ge=0.5, le=2.0)


class ColorizeResponse(BaseModel):
    status: str
    message: str
    output_image: Optional[str] = None
    confidence_image: Optional[str] = None
    mode: ModeType = "natural"
    width: Optional[int] = None
    height: Optional[int] = None


class HealthResponse(BaseModel):
    status: str
    app: str
