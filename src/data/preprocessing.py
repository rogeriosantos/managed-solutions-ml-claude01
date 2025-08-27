"""Data preprocessing pipeline for CNC ML Project."""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer, KNNImputer
import logging

from ..utils.helpers import (
    setup_logging, validate_datetime, calculate_efficiency, 
    detect_outliers, create_time_features, safe_divide,
    get_downtime_categories, validate_data_quality
)
from ..utils.config import config

logger = setup_logging()


class DataPreprocessor:
    """Data preprocessing pipeline for CNC manufacturing data."""
    
    def __init__(self):
        """Initialize preprocessor with configuration."""
        self.scalers = {}
        self.encoders = {}
        self.imputers = {}
        self.feature_names = []
        self.is_fitted = False
        
        self.config = config.get_feature_config()
        self.downtime_cols = get_downtime_categories()
    
    def validate_raw_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean raw data from database."""
        logger.info(f"Starting data validation for {len(df)} records")
        
        # Generate quality report
        quality_report = validate_data_quality(df)
        logger.info(f"Data quality report: {quality_report}")
        
        # Clean datetime fields
        df = self._clean_datetime_fields(df)
        
        # Clean duration fields
        df = self._clean_duration_fields(df)
        
        # Handle missing values in critical fields
        df = self._handle_missing_values(df)
        
        # Remove extreme outliers
        df = self._remove_extreme_outliers(df)
        
        logger.info(f"Data validation complete. {len(df)} records remaining")
        return df
    
    def _clean_datetime_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean datetime fields and remove invalid timestamps."""
        df = df.copy()
        
        # Fix StartTime and EndTime
        for col in ['StartTime', 'EndTime']:
            if col in df.columns:
                # Remove 1969 timestamps
                df[col] = pd.to_datetime(df[col], errors='coerce')
                invalid_mask = (df[col].dt.year == 1969) | df[col].isnull()
                
                if invalid_mask.sum() > 0:
                    logger.warning(f"Removing {invalid_mask.sum()} records with invalid {col}")
                    df = df[~invalid_mask]
        
        # Calculate actual job duration if EndTime is available
        if 'StartTime' in df.columns and 'EndTime' in df.columns:
            df['ActualDuration'] = (df['EndTime'] - df['StartTime']).dt.total_seconds()
            
            # Compare with reported JobDuration
            if 'JobDuration' in df.columns:
                duration_diff = abs(df['ActualDuration'] - df['JobDuration'])
                significant_diff = duration_diff > 300  # 5 minutes
                
                if significant_diff.sum() > 0:
                    logger.info(f"Found {significant_diff.sum()} records with significant duration differences")
        
        return df
    
    def _clean_duration_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean duration fields and handle negative/extreme values."""
        df = df.copy()
        
        duration_fields = ['JobDuration', 'RunningTime'] + self.downtime_cols
        
        for col in duration_fields:
            if col in df.columns:
                # Handle negative values
                negative_mask = df[col] < 0
                if negative_mask.sum() > 0:
                    logger.warning(f"Setting {negative_mask.sum()} negative {col} values to 0")
                    df.loc[negative_mask, col] = 0
                
                # Handle extreme values (> 24 hours for single job)
                if col != 'JobDuration':  # JobDuration can be longer
                    extreme_mask = df[col] > 86400  # 24 hours
                    if extreme_mask.sum() > 0:
                        logger.warning(f"Capping {extreme_mask.sum()} extreme {col} values")
                        df.loc[extreme_mask, col] = 86400
        
        # Ensure JobDuration is within reasonable bounds
        if 'JobDuration' in df.columns:
            min_duration = self.config['min_job_duration']
            max_duration = self.config['max_job_duration']
            
            df = df[(df['JobDuration'] >= min_duration) & (df['JobDuration'] <= max_duration)]
            logger.info(f"Filtered jobs to duration range [{min_duration}, {max_duration}] seconds")
        
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in dataset."""
        df = df.copy()
        
        # Critical fields that cannot be null
        critical_fields = ['machine', 'StartTime', 'JobDuration', 'State']
        
        for field in critical_fields:
            if field in df.columns:
                null_mask = df[field].isnull()
                if null_mask.sum() > 0:
                    logger.warning(f"Removing {null_mask.sum()} records with null {field}")
                    df = df[~null_mask]
        
        # Fill missing operator names with 'Unknown'
        if 'OperatorName' in df.columns:
            df['OperatorName'] = df['OperatorName'].fillna('Unknown')
        
        # Fill missing part numbers
        if 'PartNumber' in df.columns:
            df['PartNumber'] = df['PartNumber'].fillna('Unknown')
        
        # Fill missing employee IDs
        if 'EmpID' in df.columns:
            df['EmpID'] = df['EmpID'].fillna('Unknown')
        
        # Fill missing duration fields with 0
        for col in self.downtime_cols:
            if col in df.columns:
                df[col] = df[col].fillna(0)
        
        # Fill missing PartsProduced with 0
        if 'PartsProduced' in df.columns:
            df['PartsProduced'] = df['PartsProduced'].fillna(0)
        
        return df
    
    def _remove_extreme_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove extreme outliers from the dataset."""
        df = df.copy()
        initial_count = len(df)
        
        # Remove extremely high idle times (> 7 days)
        if 'IdleTime' in df.columns:
            extreme_idle = df['IdleTime'] > 604800  # 7 days
            df = df[~extreme_idle]
            if extreme_idle.sum() > 0:
                logger.info(f"Removed {extreme_idle.sum()} records with extreme idle times")
        
        # Remove jobs with impossible parts counts
        if 'PartsProduced' in df.columns:
            extreme_parts = df['PartsProduced'] > 10000  # Very high for single job
            df = df[~extreme_parts]
            if extreme_parts.sum() > 0:
                logger.info(f"Removed {extreme_parts.sum()} records with extreme parts count")
        
        logger.info(f"Outlier removal: {initial_count} -> {len(df)} records")
        return df
    
    def create_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create engineered features from raw data."""
        df = df.copy()
        logger.info("Creating derived features...")
        
        # Efficiency metrics
        if 'RunningTime' in df.columns and 'JobDuration' in df.columns:
            df['efficiency'] = safe_divide(df['RunningTime'], df['JobDuration'])
            df['efficiency'] = df['efficiency'].clip(0, 1)  # Cap at 100%
        
        # Productivity metrics
        if 'PartsProduced' in df.columns and 'RunningTime' in df.columns:
            df['parts_per_hour'] = safe_divide(df['PartsProduced'] * 3600, df['RunningTime'])
        
        # Total downtime
        df['total_downtime'] = df[self.downtime_cols].sum(axis=1)
        
        # Downtime ratios
        for col in self.downtime_cols:
            if col in df.columns and 'JobDuration' in df.columns:
                df[f'{col}_ratio'] = safe_divide(df[col], df['JobDuration'])
        
        # Setup efficiency (RunningTime / (RunningTime + SetupTime))
        if all(col in df.columns for col in ['RunningTime', 'SetupTime']):
            total_active_time = df['RunningTime'] + df['SetupTime']
            df['setup_efficiency'] = safe_divide(df['RunningTime'], total_active_time)
        
        # Time-based features
        if 'StartTime' in df.columns:
            df = create_time_features(df, 'StartTime')
        
        # Categorical features
        df = self._create_categorical_features(df)
        
        # Rolling averages for operators (if enough data)
        df = self._create_rolling_features(df)
        
        logger.info(f"Feature engineering complete. Dataset shape: {df.shape}")
        return df
    
    def _create_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create categorical feature encodings."""
        df = df.copy()
        
        # Machine complexity category
        if 'machine' in df.columns:
            machine_complexity = df.groupby('machine')['JobDuration'].mean()
            df['machine_complexity'] = df['machine'].map(machine_complexity)
            df['machine_complexity_cat'] = pd.cut(
                df['machine_complexity'], 
                bins=3, 
                labels=['Simple', 'Medium', 'Complex']
            )
        
        # Part complexity category
        if 'PartNumber' in df.columns:
            part_complexity = df.groupby('PartNumber')['SetupTime'].mean()
            df['part_complexity'] = df['PartNumber'].map(part_complexity)
        
        # Operator experience level
        if 'OperatorName' in df.columns:
            operator_jobs = df.groupby('OperatorName').size()
            df['operator_experience'] = df['OperatorName'].map(operator_jobs)
            df['operator_experience_cat'] = pd.cut(
                df['operator_experience'], 
                bins=[0, 10, 50, 200, float('inf')], 
                labels=['Novice', 'Intermediate', 'Experienced', 'Expert']
            )
        
        return df
    
    def _create_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create rolling average features for performance tracking."""
        if len(df) < 10:  # Need minimum data for rolling features
            return df
        
        df = df.copy()
        df = df.sort_values('StartTime')
        
        # Rolling features by operator
        if 'OperatorName' in df.columns:
            for window in [5, 10]:
                for metric in ['efficiency', 'parts_per_hour', 'total_downtime']:
                    if metric in df.columns:
                        col_name = f'operator_{metric}_rolling_{window}'
                        df[col_name] = df.groupby('OperatorName')[metric].transform(
                            lambda x: x.rolling(window=window, min_periods=1).mean()
                        )
        
        # Rolling features by machine
        if 'machine' in df.columns:
            for window in [5, 10]:
                for metric in ['efficiency', 'JobDuration']:
                    if metric in df.columns:
                        col_name = f'machine_{metric}_rolling_{window}'
                        df[col_name] = df.groupby('machine')[metric].transform(
                            lambda x: x.rolling(window=window, min_periods=1).mean()
                        )
        
        return df
    
    def prepare_features_for_modeling(self, df: pd.DataFrame, target_cols: List[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Prepare features for machine learning models."""
        df = df.copy()
        
        # Select feature columns
        feature_cols = self._select_feature_columns(df)
        
        # Handle categorical variables
        df_encoded = self._encode_categorical_features(df[feature_cols])
        
        # Handle numerical scaling
        df_scaled = self._scale_numerical_features(df_encoded)
        
        # Prepare targets
        if target_cols:
            target_df = df[target_cols].copy()
        else:
            target_df = pd.DataFrame()
        
        self.feature_names = df_scaled.columns.tolist()
        self.is_fitted = True
        
        logger.info(f"Feature preparation complete. Features: {len(self.feature_names)}")
        return df_scaled, target_df
    
    def _select_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """Select relevant columns for modeling."""
        # Exclude target-like columns and identifiers
        exclude_cols = [
            'StartTime', 'EndTime', 'JobNumber', 'State', 'OpNumber', 'EmpID',
            'ActualDuration'  # Derived column that would cause data leakage
        ]
        
        # Include numerical and categorical features
        feature_cols = []
        for col in df.columns:
            if col not in exclude_cols:
                # Include if numerical or important categorical
                if df[col].dtype in ['int64', 'float64'] or col in ['machine', 'OperatorName', 'PartNumber']:
                    feature_cols.append(col)
        
        return feature_cols
    
    def _encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical features for modeling."""
        df_encoded = df.copy()
        
        categorical_cols = ['machine', 'OperatorName', 'PartNumber', 'shift', 
                          'machine_complexity_cat', 'operator_experience_cat']
        
        for col in categorical_cols:
            if col in df_encoded.columns:
                if col not in self.encoders:
                    self.encoders[col] = LabelEncoder()
                    df_encoded[f'{col}_encoded'] = self.encoders[col].fit_transform(
                        df_encoded[col].astype(str)
                    )
                else:
                    df_encoded[f'{col}_encoded'] = self.encoders[col].transform(
                        df_encoded[col].astype(str)
                    )
                
                # Drop original categorical column
                df_encoded = df_encoded.drop(columns=[col])
        
        return df_encoded
    
    def _scale_numerical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Scale numerical features."""
        df_scaled = df.copy()
        
        # Identify numerical columns
        numerical_cols = df_scaled.select_dtypes(include=[np.number]).columns.tolist()
        
        # Separate features that should use different scalers
        time_features = [col for col in numerical_cols if any(
            keyword in col.lower() for keyword in ['time', 'duration']
        )]
        
        ratio_features = [col for col in numerical_cols if 'ratio' in col.lower() or 'efficiency' in col.lower()]
        
        other_features = [col for col in numerical_cols if col not in time_features + ratio_features]
        
        # Scale time features
        if time_features:
            if 'time_scaler' not in self.scalers:
                self.scalers['time_scaler'] = StandardScaler()
                df_scaled[time_features] = self.scalers['time_scaler'].fit_transform(df_scaled[time_features])
            else:
                df_scaled[time_features] = self.scalers['time_scaler'].transform(df_scaled[time_features])
        
        # MinMax scale ratio features (already 0-1 range)
        if ratio_features:
            if 'ratio_scaler' not in self.scalers:
                self.scalers['ratio_scaler'] = MinMaxScaler()
                df_scaled[ratio_features] = self.scalers['ratio_scaler'].fit_transform(df_scaled[ratio_features])
            else:
                df_scaled[ratio_features] = self.scalers['ratio_scaler'].transform(df_scaled[ratio_features])
        
        # Standard scale other features
        if other_features:
            if 'standard_scaler' not in self.scalers:
                self.scalers['standard_scaler'] = StandardScaler()
                df_scaled[other_features] = self.scalers['standard_scaler'].fit_transform(df_scaled[other_features])
            else:
                df_scaled[other_features] = self.scalers['standard_scaler'].transform(df_scaled[other_features])
        
        return df_scaled
    
    def transform_new_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform new data using fitted preprocessors."""
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted first")
        
        # Apply same preprocessing steps
        df = self.validate_raw_data(df)
        df = self.create_derived_features(df)
        
        # Select same features
        feature_cols = [col for col in df.columns if col in self.feature_names or 
                       any(col.startswith(base) for base in ['machine', 'OperatorName', 'PartNumber', 'shift'])]
        
        df_features = df[feature_cols]
        
        # Apply encodings
        df_encoded = self._encode_categorical_features(df_features)
        
        # Apply scaling
        df_scaled = self._scale_numerical_features(df_encoded)
        
        # Ensure same feature order
        missing_cols = set(self.feature_names) - set(df_scaled.columns)
        if missing_cols:
            for col in missing_cols:
                df_scaled[col] = 0
        
        df_scaled = df_scaled[self.feature_names]
        
        return df_scaled
    
    def get_feature_importance_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate data for feature importance analysis."""
        feature_stats = {}
        
        for col in df.columns:
            if df[col].dtype in ['int64', 'float64']:
                feature_stats[col] = {
                    'mean': df[col].mean(),
                    'std': df[col].std(),
                    'min': df[col].min(),
                    'max': df[col].max(),
                    'missing_ratio': df[col].isnull().mean(),
                    'zeros_ratio': (df[col] == 0).mean()
                }
        
        return feature_stats