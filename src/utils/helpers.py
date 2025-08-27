"""Utility helper functions for CNC ML Project."""

import logging
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import pickle
import joblib
from pathlib import Path


def setup_logging(log_level: str = 'INFO', log_file: Optional[str] = None) -> logging.Logger:
    """Set up logging configuration."""
    logger = logging.getLogger('cnc_ml')
    logger.setLevel(getattr(logging, log_level.upper()))
    
    if not logger.handlers:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File handler if specified
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
    
    return logger


def validate_datetime(dt_str: str) -> bool:
    """Validate if datetime string is valid and not 1969."""
    try:
        dt = pd.to_datetime(dt_str)
        return dt.year > 1970
    except:
        return False


def calculate_efficiency(running_time: float, job_duration: float) -> float:
    """Calculate efficiency ratio with bounds checking."""
    if job_duration <= 0:
        return 0.0
    efficiency = running_time / job_duration
    return min(efficiency, 1.0)  # Cap at 100%


def detect_outliers(data: pd.Series, method: str = 'iqr', threshold: float = 1.5) -> pd.Series:
    """Detect outliers using IQR or z-score method."""
    if method == 'iqr':
        q1 = data.quantile(0.25)
        q3 = data.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr
        return (data < lower_bound) | (data > upper_bound)
    
    elif method == 'zscore':
        z_scores = np.abs((data - data.mean()) / data.std())
        return z_scores > threshold
    
    else:
        raise ValueError("Method must be 'iqr' or 'zscore'")


def create_time_features(df: pd.DataFrame, datetime_col: str) -> pd.DataFrame:
    """Create time-based features from datetime column."""
    df = df.copy()
    dt = pd.to_datetime(df[datetime_col])
    
    df['hour'] = dt.dt.hour
    df['day_of_week'] = dt.dt.dayofweek
    df['day_of_month'] = dt.dt.day
    df['month'] = dt.dt.month
    df['quarter'] = dt.dt.quarter
    df['is_weekend'] = (dt.dt.dayofweek >= 5).astype(int)
    
    # Shift classification
    df['shift'] = df['hour'].apply(classify_shift)
    
    # Cyclical encoding for time features
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    
    return df


def classify_shift(hour: int) -> str:
    """Classify hour into shift (Day/Evening/Night)."""
    if 6 <= hour < 14:
        return 'Day'
    elif 14 <= hour < 22:
        return 'Evening'
    else:
        return 'Night'


def safe_divide(numerator: pd.Series, denominator: pd.Series, default_value: float = 0.0) -> pd.Series:
    """Safely divide two series, handling division by zero."""
    return np.where(denominator != 0, numerator / denominator, default_value)


def encode_categorical_features(df: pd.DataFrame, categorical_cols: List[str]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Encode categorical features and return encoders."""
    from sklearn.preprocessing import LabelEncoder
    
    df_encoded = df.copy()
    encoders = {}
    
    for col in categorical_cols:
        if col in df_encoded.columns:
            le = LabelEncoder()
            df_encoded[f'{col}_encoded'] = le.fit_transform(df_encoded[col].astype(str))
            encoders[col] = le
    
    return df_encoded, encoders


def save_model(model: Any, filepath: str) -> None:
    """Save model to file."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, filepath)


def load_model(filepath: str) -> Any:
    """Load model from file."""
    return joblib.load(filepath)


def calculate_performance_score(
    efficiency: float,
    quality_score: float = 1.0,
    consistency_score: float = 1.0,
    weights: Dict[str, float] = None
) -> float:
    """Calculate composite performance score."""
    if weights is None:
        weights = {'efficiency': 0.5, 'quality': 0.3, 'consistency': 0.2}
    
    score = (
        weights['efficiency'] * efficiency +
        weights['quality'] * quality_score +
        weights['consistency'] * consistency_score
    )
    
    return min(score, 1.0)


def get_downtime_categories() -> List[str]:
    """Get list of downtime categories from database schema."""
    return [
        'SetupTime', 'WaitingSetupTime', 'NotFeedingTime', 'AdjustmentTime',
        'DressingTime', 'ToolingTime', 'EngineeringTime', 'MaintenanceTime',
        'BuyInTime', 'BreakShiftChangeTime', 'IdleTime'
    ]


def seconds_to_human_readable(seconds: int) -> str:
    """Convert seconds to human readable format."""
    if seconds < 60:
        return f"{seconds}s"
    elif seconds < 3600:
        minutes = seconds // 60
        secs = seconds % 60
        return f"{minutes}m {secs}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        return f"{hours}h {minutes}m"


def validate_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """Perform comprehensive data quality validation."""
    quality_report = {
        'total_records': len(df),
        'missing_values': df.isnull().sum().to_dict(),
        'duplicate_records': df.duplicated().sum(),
        'data_types': df.dtypes.to_dict(),
        'negative_durations': {},
        'invalid_dates': 0,
        'outliers': {}
    }
    
    # Check for negative duration values
    duration_cols = get_downtime_categories() + ['JobDuration', 'RunningTime']
    for col in duration_cols:
        if col in df.columns:
            negative_count = (df[col] < 0).sum()
            if negative_count > 0:
                quality_report['negative_durations'][col] = negative_count
    
    # Check for invalid dates (1969 timestamps)
    if 'StartTime' in df.columns:
        invalid_dates = pd.to_datetime(df['StartTime'], errors='coerce').dt.year == 1969
        quality_report['invalid_dates'] = invalid_dates.sum()
    
    # Detect outliers in key metrics
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col in ['JobDuration', 'RunningTime', 'PartsProduced']:
            outliers = detect_outliers(df[col].dropna())
            quality_report['outliers'][col] = outliers.sum()
    
    return quality_report