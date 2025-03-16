from pydantic import BaseModel, field_validator


class PredictNumber(BaseModel):
    timestamp: str
    label: int
    prediction: int
    confidence: float

    class Config:
        from_attributes = True

    @field_validator("timestamp", mode="before")
    @classmethod
    def parse_timestamp(cls, date):
        return date.strftime("%Y-%m-%d %H:%M:%S")


class PredictNumberIn(BaseModel):
    b64_encoded: str
    label: int
