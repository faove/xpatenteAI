from pydantic import BaseModel


class DetectionPlate(BaseModel):
    image_path: str
    confidence: float    
    