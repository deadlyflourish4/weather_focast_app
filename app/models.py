from pydantic import BaseModel, Field, field_validator, conint, confloat
from typing import List, Optional, Literal, Dict, Any
from datetime import datetime

class CurrentWeatherResponse(BaseModel):
    latitude: float
    longitude: float
    observed_time_utc: Optional[str] = Field(None, description="Time of observation in UTC")
    temperature_celsius: Optional[float] = Field(None, example=30.0)
    temperature_kelvin: Optional[float] = Field(None, example=303.15)
    humidity: Optional[float] = Field(None, example=85.0)
    pressure: Optional[float] = Field(None, example=1012.0, description="Pressure in mb/hPa")
    wind_speed_mps: Optional[float] = Field(None, example=2.5, description="Wind speed in m/s")
    wind_direction_degree: Optional[float] = Field(None, example=180.0)
    weather_condition: Optional[str] = Field(None, example="sunny", description="Aggregated weather condition (from API or predicted)")
    condition_source: str = Field(..., description="Indicates 'api' or 'predicted'")

class XGBoostTrainParams(BaseModel):
    learning_rate: Optional[confloat(gt=0)] = Field(0.1, description="Step size shrinkage used in update to prevents overfitting. range: (0,1]")
    booster: Optional[Literal['gbtree', 'gblinear', 'dart']] = Field('gbtree', description="Which booster to use.")
    gamma: Optional[confloat(ge=0)] = Field(0, description="Minimum loss reduction required to make a split. range: [0,∞]")
    max_depth: Optional[conint(ge=0)] = Field(8, description="Maximum depth of a tree. 0 indicates no limit. range: [0,∞]") # Default changed slightly
    min_child_weight: Optional[confloat(ge=0)] = Field(1, description="Minimum sum of instance weight needed in a child. range: [0,∞]")
    subsample: Optional[confloat(gt=0, le=1)] = Field(0.8, description="Subsample ratio of the training instance. range: (0,1]")
    sampling_method: Optional[Literal['uniform', 'gradient_based']] = Field('uniform', description="Sampling method.")
    reg_lambda: Optional[confloat(ge=0)] = Field(1, alias='lambda', description="L2 regularization term. range: [0,∞]")
    reg_alpha: Optional[confloat(ge=0)] = Field(0, alias='alpha', description="L1 regularization term. range: [0,∞]")
    tree_method: Optional[Literal['auto', 'exact', 'approx', 'hist']] = Field('auto', description="Tree construction algorithm.")
    n_estimators: Optional[conint(gt=0)] = Field(200, description="Number of boosting rounds (trees).") # Added n_estimators

    # Allow 'lambda' alias for reg_lambda in incoming JSON
    model_config = {
        "populate_by_name": True,
        "extra": "ignore" # Ignore extra fields in request
    }

class PredictRequestItem(BaseModel):
    humidity: float = Field(..., example=85.0)
    pressure: float = Field(..., example=1012.0)
    temperature: float = Field(..., example=300.15, description="Temperature in Kelvin")
    wind_direction: float = Field(..., example=180.0)
    wind_speed: float = Field(..., example=2.5, description="Wind speed in m/s")
    latitude: float = Field(..., example=10.76)
    longitude: float = Field(..., example=106.66)
    # Optional fields
    datetime: Optional[str] = Field(None, example="2024-01-01T12:00:00")
    province: Optional[str] = Field(None, example="Ho Chi Minh City, Vietnam")

class PredictRequest(BaseModel):
    instances: List[PredictRequestItem]

class PredictResponse(BaseModel):
    predictions: List[str]

class BackgroundTaskResponse(BaseModel):
    message: str

class TrainResponse(BaseModel):
    message: str
    model_path: str
    label_encoder_path: str
    scaler_path: str
    accuracy: Optional[float] = None
    
class RetrainTriggerResponse(BaseModel):
    message: str
    task_id: str
    status_endpoint: str = Field(..., description="URL to poll for retraining status and results")

class RetrainStatus(BaseModel):
    task_id: str
    status: Literal["pending", "running", "completed", "failed"]
    message: Optional[str] = Field(None, description="Status message or error details")

    # --- Metrics ---
    accuracy: Optional[float] = Field(None, description="Overall model accuracy on the test set")
    precision_macro: Optional[float] = Field(None, description="Macro-averaged precision across all classes")
    recall_macro: Optional[float] = Field(None, description="Macro-averaged recall across all classes")
    f1_score_macro: Optional[float] = Field(None, description="Macro-averaged F1-score across all classes")
    classification_report_dict: Optional[Dict[str, Dict[str, float]]] = Field(None, description="Detailed classification report as a dictionary") # Store the dict report

    # --- Metadata ---
    start_time: Optional[datetime] = Field(None, description="Time the retraining task started")
    end_time: Optional[datetime] = Field(None, description="Time the retraining task finished")
    params_used: Optional[Dict[str, Any]] = Field(None, description="Hyperparameters used for this training run")