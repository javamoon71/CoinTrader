# model_package.py
from dataclasses import dataclass
from datetime import datetime

@dataclass
class ModelPackage:
    model: any
    mse_close: float
    mse_rsi: float
    trained_at: datetime
    scaler: any = None
    features: list = None