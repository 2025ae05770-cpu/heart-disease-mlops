"""Request / response schemas with clinically plausible validation ranges."""

from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class PatientRecord(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "age": 63, "sex": 1, "cp": 1, "trestbps": 145, "chol": 233, "fbs": 1,
                "restecg": 2, "thalach": 150, "exang": 0, "oldpeak": 2.3, "slope": 3,
                "ca": 0, "thal": 6,
            }
        },
    )

    age: int = Field(..., ge=1, le=120, description="Age in years")
    sex: Literal[0, 1] = Field(..., description="1 = male, 0 = female")
    cp: Literal[1, 2, 3, 4] = Field(..., description="Chest pain: 1 typical, 2 atypical, 3 non-anginal, 4 asymptomatic")
    trestbps: float = Field(..., ge=50, le=250, description="Resting blood pressure (mm Hg)")
    chol: float = Field(..., ge=50, le=700, description="Serum cholesterol (mg/dl)")
    fbs: Literal[0, 1] = Field(..., description="Fasting blood sugar > 120 mg/dl")
    restecg: Literal[0, 1, 2] = Field(..., description="Resting ECG: 0 normal, 1 ST-T abnormality, 2 LVH")
    thalach: float = Field(..., ge=40, le=250, description="Maximum heart rate achieved")
    exang: Literal[0, 1] = Field(..., description="Exercise induced angina")
    oldpeak: float = Field(..., ge=0, le=10, description="ST depression induced by exercise")
    slope: Literal[1, 2, 3] = Field(..., description="Slope of peak exercise ST: 1 up, 2 flat, 3 down")
    ca: Optional[int] = Field(None, ge=0, le=3, description="Major vessels coloured by fluoroscopy (optional)")
    thal: Optional[Literal[3, 6, 7]] = Field(None, description="3 normal, 6 fixed defect, 7 reversible (optional)")


class PredictionResponse(BaseModel):
    prediction: int = Field(..., description="1 = heart disease predicted, 0 = not predicted")
    label: str
    confidence: float = Field(..., description="Probability of the predicted class")
    probability_heart_disease: float
    risk_level: Literal["low", "moderate", "high"]
    model_version: str
    request_id: str


class BatchRequest(BaseModel):
    records: List[PatientRecord] = Field(..., min_length=1, max_length=500)


class BatchResponse(BaseModel):
    predictions: List[PredictionResponse]
    count: int


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: Optional[str] = None
