from pydantic import BaseModel, Field
from typing import Optional, Literal

class ChurnPredictionRequest(BaseModel):
    """
    Input schema for Churn Prediction.
    Matches the raw data format expected by preprocessing pipeline.
    """
    # Customer Demographics
    gender: Optional[str] = Field(None, description="Customer gender")
    SeniorCitizen: int = Field(..., description="0 or 1")
    Partner: str = Field(..., description="Yes/No")
    Dependents: str = Field(..., description="Yes/No")
    
    # Account Information
    tenure: int = Field(..., ge=0, description="Number of months with company")
    Contract: str = Field(..., description="Month-to-month, One year, Two year")
    PaperlessBilling: str = Field(..., description="Yes/No")
    PaymentMethod: str = Field(..., description="Electronic check, Mailed check, etc.")
    MonthlyCharges: float = Field(..., gt=0, description="Monthly amount charged")
    TotalCharges: float = Field(..., ge=0, description="Total amount charged")
    
    # Services
    PhoneService: Optional[str] = Field("No", description="Yes/No")
    MultipleLines: Optional[str] = Field("No", description="Yes/No")
    InternetService: str = Field(..., description="DSL, Fiber optic, No")
    OnlineSecurity: str = Field(..., description="Yes/No/No internet service")
    OnlineBackup: str = Field(..., description="Yes/No/No internet service")
    DeviceProtection: str = Field(..., description="Yes/No/No internet service")
    TechSupport: str = Field(..., description="Yes/No/No internet service")
    StreamingTV: str = Field(..., description="Yes/No/No internet service")
    StreamingMovies: str = Field(..., description="Yes/No/No internet service")

class ChurnPredictionResponse(BaseModel):
    churn_probability: float
    risk_level: str
