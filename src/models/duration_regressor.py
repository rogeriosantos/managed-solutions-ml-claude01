"""Duration regression models for predicting downtime and job durations."""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import logging
import joblib

from ..utils.config import config
from ..utils.helpers import setup_logging, get_downtime_categories

logger = setup_logging()


class DurationRegressor:
    """Regression models for predicting various duration types."""
    
    def __init__(self, target_type: str = 'job_duration', model_type: str = 'xgboost'):
        """
        Initialize duration regressor.
        
        Args:
            target_type: Type of duration to predict ('job_duration', 'setup_time', 'maintenance_time', etc.)
            model_type: ML model type ('random_forest', 'xgboost', 'gradient_boosting')
        """
        self.target_type = target_type
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.is_fitted = False
        
        self.config = config.get_model_config()
        self.downtime_categories = get_downtime_categories()
        
        # Initialize model
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the regression model based on type."""
        if self.model_type == 'random_forest':
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=self.config['random_state'],
                n_jobs=-1
            )
        
        elif self.model_type == 'xgboost':
            self.model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.config['random_state'],
                n_jobs=-1
            )
        
        elif self.model_type == 'gradient_boosting':
            self.model = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                random_state=self.config['random_state']
            )
        
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def prepare_duration_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for duration prediction."""
        logger.info(f"Preparing features for {self.target_type} prediction...")
        
        features_df = df.copy()
        
        # Historical duration patterns
        if self.target_type == 'job_duration':
            # Historical job duration by machine-part combination
            machine_part_avg = features_df.groupby(['machine', 'PartNumber'])['JobDuration'].transform('mean')
            features_df['machine_part_avg_duration'] = machine_part_avg
            
            # Historical by operator-machine combination
            operator_machine_avg = features_df.groupby(['OperatorName', 'machine'])['JobDuration'].transform('mean')
            features_df['operator_machine_avg_duration'] = operator_machine_avg
            
            # Historical by operator-part combination
            operator_part_avg = features_df.groupby(['OperatorName', 'PartNumber'])['JobDuration'].transform('mean')
            features_df['operator_part_avg_duration'] = operator_part_avg
        
        elif self.target_type in self.downtime_categories:
            # Historical downtime by category
            machine_downtime_avg = features_df.groupby('machine')[self.target_type].transform('mean')
            features_df[f'machine_avg_{self.target_type}'] = machine_downtime_avg
            
            operator_downtime_avg = features_df.groupby('OperatorName')[self.target_type].transform('mean')
            features_df[f'operator_avg_{self.target_type}'] = operator_downtime_avg
            
            part_downtime_avg = features_df.groupby('PartNumber')[self.target_type].transform('mean')
            features_df[f'part_avg_{self.target_type}'] = part_downtime_avg
        
        # Rolling averages
        features_df = self._add_rolling_features(features_df)
        
        # Time-based features
        features_df = self._add_temporal_features(features_df)
        
        # Complexity indicators
        features_df = self._add_complexity_features(features_df)
        
        # Recent performance indicators
        features_df = self._add_performance_features(features_df)
        
        logger.info(f"Feature preparation complete for {self.target_type}")
        return features_df
    
    def _add_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add rolling average features."""
        df = df.copy()
        df = df.sort_values(['machine', 'OperatorName', 'StartTime'])
        
        # Rolling features for target variable
        for window in [3, 5, 10]:
            # By machine
            col_name = f'machine_{self.target_type}_rolling_{window}'
            df[col_name] = df.groupby('machine')[self.target_type].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )
            
            # By operator
            col_name = f'operator_{self.target_type}_rolling_{window}'
            df[col_name] = df.groupby('OperatorName')[self.target_type].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )
        
        # Rolling features for related variables
        if self.target_type == 'job_duration':
            related_vars = ['RunningTime', 'SetupTime', 'total_downtime']
        elif self.target_type == 'SetupTime':
            related_vars = ['JobDuration', 'RunningTime']
        else:
            related_vars = ['JobDuration', 'SetupTime']
        
        for var in related_vars:
            if var in df.columns:
                for window in [3, 5]:
                    col_name = f'{var}_rolling_{window}'
                    df[col_name] = df.groupby('machine')[var].transform(
                        lambda x: x.rolling(window=window, min_periods=1).mean()
                    )
        
        return df
    
    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features that might affect duration."""
        df = df.copy()
        
        if 'StartTime' in df.columns:
            start_time = pd.to_datetime(df['StartTime'])
            
            # Time since start of shift
            df['hour_in_shift'] = (start_time.dt.hour - 6) % 24  # Assuming 6am shift start
            df['minutes_in_hour'] = start_time.dt.minute
            
            # Day of week effects
            df['is_monday'] = (start_time.dt.dayofweek == 0).astype(int)
            df['is_friday'] = (start_time.dt.dayofweek == 4).astype(int)
            
            # End of month effects (potential rush)
            df['days_until_month_end'] = start_time.dt.days_in_month - start_time.dt.day
            df['is_month_end'] = (df['days_until_month_end'] <= 3).astype(int)
        
        # Consecutive job number (job sequence effect)
        df['job_sequence'] = df.groupby(['machine', 'OperatorName']).cumcount() + 1
        
        # Time since last job of same part
        df = df.sort_values(['machine', 'PartNumber', 'StartTime'])
        df['time_since_same_part'] = df.groupby(['machine', 'PartNumber'])['StartTime'].diff().dt.total_seconds()
        df['time_since_same_part'] = df['time_since_same_part'].fillna(86400)  # 24 hours if first
        
        return df
    
    def _add_complexity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add features indicating job/part complexity."""
        df = df.copy()
        
        # Part complexity based on historical setup times
        if 'PartNumber' in df.columns and 'SetupTime' in df.columns:
            part_complexity = df.groupby('PartNumber')['SetupTime'].mean()
            df['part_setup_complexity'] = df['PartNumber'].map(part_complexity)
            
            # Normalize to 0-1 scale
            df['part_complexity_score'] = (
                df['part_setup_complexity'] / df['part_setup_complexity'].max()
            ).fillna(0)
        
        # Machine complexity based on average job duration
        if 'machine' in df.columns:
            machine_complexity = df.groupby('machine')['JobDuration'].mean()
            df['machine_complexity'] = df['machine'].map(machine_complexity)
        
        # Operator experience with this specific part
        if 'OperatorName' in df.columns and 'PartNumber' in df.columns:
            operator_part_experience = df.groupby(['OperatorName', 'PartNumber']).cumcount() + 1
            df['operator_part_experience'] = operator_part_experience
            
            # Operator experience with this machine
            operator_machine_experience = df.groupby(['OperatorName', 'machine']).cumcount() + 1
            df['operator_machine_experience'] = operator_machine_experience
        
        # Expected parts quantity (if available)
        if 'PartsProduced' in df.columns:
            # Use historical average for this part as expected quantity
            part_avg_quantity = df.groupby('PartNumber')['PartsProduced'].mean()
            df['expected_parts_quantity'] = df['PartNumber'].map(part_avg_quantity).fillna(1)
        
        return df
    
    def _add_performance_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add recent performance indicators."""
        df = df.copy()
        df = df.sort_values(['machine', 'OperatorName', 'StartTime'])
        
        # Recent efficiency trend
        if 'efficiency' in df.columns:
            df['recent_efficiency'] = df.groupby(['machine', 'OperatorName'])['efficiency'].transform(
                lambda x: x.rolling(window=3, min_periods=1).mean()
            )
            
            # Efficiency deviation from machine average
            machine_avg_efficiency = df.groupby('machine')['efficiency'].transform('mean')
            df['efficiency_vs_machine_avg'] = df['efficiency'] - machine_avg_efficiency
        
        # Recent downtime trend
        if 'total_downtime' in df.columns:
            df['recent_downtime_trend'] = df.groupby(['machine', 'OperatorName'])['total_downtime'].transform(
                lambda x: x.rolling(window=3, min_periods=1).mean()
            )
        
        # Performance consistency (standard deviation of recent efficiency)
        if 'efficiency' in df.columns:
            df['efficiency_consistency'] = df.groupby(['machine', 'OperatorName'])['efficiency'].transform(
                lambda x: x.rolling(window=5, min_periods=1).std()
            ).fillna(0)
        
        return df
    
    def train(self, df: pd.DataFrame, optimize_hyperparameters: bool = True) -> Dict[str, Any]:
        """Train the duration regression model."""
        logger.info(f"Training {self.model_type} model for {self.target_type} prediction...")
        
        # Prepare features
        features_df = self.prepare_duration_features(df)
        
        # Remove rows with missing target
        valid_data = features_df.dropna(subset=[self.target_type])
        
        # For job duration, only use jobs that actually completed
        if self.target_type == 'job_duration':
            valid_data = valid_data[valid_data['JobDuration'] > 0]
        
        logger.info(f"Valid training samples: {len(valid_data)}")
        
        # Select features
        feature_cols = self._select_features(valid_data)
        X = valid_data[feature_cols]
        y = valid_data[self.target_type]
        
        # Handle infinite or very large values
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median())
        
        # Log transform target if it's highly skewed
        if y.skew() > 2:
            y = np.log1p(y)  # log(1 + y) to handle zeros
            self.log_transformed = True
        else:
            self.log_transformed = False
        
        # Store feature names
        self.feature_names = feature_cols
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.config['test_size'],
            random_state=self.config['random_state']
        )
        
        logger.info(f"Training set: {len(X_train)}, Test set: {len(X_test)}")
        
        # Scale features for gradient boosting models
        if self.model_type in ['gradient_boosting']:
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
        else:
            X_train_scaled = X_train
            X_test_scaled = X_test
        
        # Hyperparameter optimization
        if optimize_hyperparameters:
            logger.info("Optimizing hyperparameters...")
            self.model = self._optimize_hyperparameters(X_train_scaled, y_train)
        
        # Train final model
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        train_pred = self.model.predict(X_train_scaled)
        test_pred = self.model.predict(X_test_scaled)
        
        # Transform back if log transformed
        if self.log_transformed:
            y_train_orig = np.expm1(y_train)
            y_test_orig = np.expm1(y_test)
            train_pred_orig = np.expm1(train_pred)
            test_pred_orig = np.expm1(test_pred)
        else:
            y_train_orig = y_train
            y_test_orig = y_test
            train_pred_orig = train_pred
            test_pred_orig = test_pred
        
        # Calculate metrics
        train_metrics = self._calculate_metrics(y_train_orig, train_pred_orig)
        test_metrics = self._calculate_metrics(y_test_orig, test_pred_orig)
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=5, scoring='neg_mean_squared_error')
        
        self.is_fitted = True
        
        results = {
            'target_type': self.target_type,
            'model_type': self.model_type,
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'cv_rmse_mean': np.sqrt(-cv_scores.mean()),
            'cv_rmse_std': np.sqrt(cv_scores.std()),
            'feature_importance': self.get_feature_importance(),
            'log_transformed': self.log_transformed,
            'n_features': len(self.feature_names),
            'n_samples': len(valid_data)
        }
        
        logger.info(f"Model training complete. Test RMSE: {test_metrics['rmse']:.2f}")
        
        return results
    
    def _select_features(self, df: pd.DataFrame) -> List[str]:
        """Select relevant features for the model."""
        # Exclude target and identifier columns
        exclude_cols = [
            self.target_type, 'StartTime', 'EndTime', 'JobNumber', 'State',
            'OpNumber', 'EmpID', 'ActualDuration'
        ]
        
        # For downtime prediction, exclude the specific downtime categories
        if self.target_type in self.downtime_categories:
            exclude_cols.extend(self.downtime_categories)
        
        # Include numerical and encoded categorical features
        feature_cols = []
        for col in df.columns:
            if col not in exclude_cols:
                if df[col].dtype in ['int64', 'float64'] or '_encoded' in col:
                    feature_cols.append(col)
        
        return feature_cols
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate regression metrics."""
        return {
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
            'mape': np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1))) * 100  # Handle zeros
        }
    
    def _optimize_hyperparameters(self, X_train: np.ndarray, y_train: np.ndarray):
        """Optimize model hyperparameters."""
        if self.model_type == 'random_forest':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        
        elif self.model_type == 'xgboost':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            }
        
        elif self.model_type == 'gradient_boosting':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0]
            }
        
        grid_search = GridSearchCV(
            self.model,
            param_grid,
            cv=3,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best CV score: {np.sqrt(-grid_search.best_score_):.2f}")
        
        return grid_search.best_estimator_
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict durations for new data."""
        if not self.is_fitted:
            raise ValueError("Model must be trained first")
        
        # Prepare features
        features_df = self.prepare_duration_features(X)
        X_features = features_df[self.feature_names]
        
        # Handle missing values
        X_features = X_features.fillna(X_features.median())
        X_features = X_features.replace([np.inf, -np.inf], np.nan)
        X_features = X_features.fillna(0)
        
        # Scale if needed
        if self.model_type in ['gradient_boosting']:
            X_scaled = self.scaler.transform(X_features)
        else:
            X_scaled = X_features
        
        # Make predictions
        predictions = self.model.predict(X_scaled)
        
        # Transform back if log transformed
        if self.log_transformed:
            predictions = np.expm1(predictions)
        
        return predictions
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores."""
        if not self.is_fitted:
            raise ValueError("Model must be trained first")
        
        if hasattr(self.model, 'feature_importances_'):
            importance_dict = dict(zip(self.feature_names, self.model.feature_importances_))
            return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
        else:
            logger.warning("Model doesn't support feature importance")
            return {}
    
    def predict_with_confidence(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Predict with confidence intervals (for tree-based models)."""
        if not self.is_fitted:
            raise ValueError("Model must be trained first")
        
        predictions = self.predict(X)
        
        # For tree-based models, use prediction variance
        if hasattr(self.model, 'estimators_'):
            # Get predictions from all trees
            all_predictions = []
            
            # Prepare features
            features_df = self.prepare_duration_features(X)
            X_features = features_df[self.feature_names]
            X_features = X_features.fillna(X_features.median())
            
            if self.model_type in ['gradient_boosting']:
                X_scaled = self.scaler.transform(X_features)
            else:
                X_scaled = X_features
            
            for estimator in self.model.estimators_.flat:
                tree_pred = estimator.predict(X_scaled)
                if self.log_transformed:
                    tree_pred = np.expm1(tree_pred)
                all_predictions.append(tree_pred)
            
            all_predictions = np.array(all_predictions)
            confidence = np.std(all_predictions, axis=0)
        else:
            # For non-tree models, use a simple heuristic
            confidence = predictions * 0.1  # 10% of prediction as confidence
        
        return predictions, confidence
    
    def save_model(self, filepath: str):
        """Save trained model to file."""
        if not self.is_fitted:
            raise ValueError("Model must be trained first")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'target_type': self.target_type,
            'model_type': self.model_type,
            'log_transformed': self.log_transformed
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained model from file."""
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.target_type = model_data['target_type']
        self.model_type = model_data['model_type']
        self.log_transformed = model_data['log_transformed']
        self.is_fitted = True
        
        logger.info(f"Model loaded from {filepath}")


class MultiTargetDurationRegressor:
    """Multi-target regressor for predicting multiple durations simultaneously."""
    
    def __init__(self, targets: List[str] = None, model_type: str = 'xgboost'):
        """Initialize multi-target duration regressor."""
        if targets is None:
            targets = ['JobDuration', 'SetupTime', 'MaintenanceTime', 'IdleTime']
        
        self.targets = targets
        self.model_type = model_type
        self.models = {}
        self.is_fitted = False
        
        # Initialize individual models for each target
        for target in targets:
            self.models[target] = DurationRegressor(target_type=target, model_type=model_type)
    
    def train(self, df: pd.DataFrame, optimize_hyperparameters: bool = True) -> Dict[str, Dict[str, Any]]:
        """Train all duration models."""
        logger.info(f"Training multi-target duration models for {self.targets}")
        
        results = {}
        
        for target in self.targets:
            if target in df.columns:
                logger.info(f"Training model for {target}...")
                target_results = self.models[target].train(df, optimize_hyperparameters)
                results[target] = target_results
            else:
                logger.warning(f"Target {target} not found in data")
        
        self.is_fitted = True
        return results
    
    def predict(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Predict all target durations."""
        if not self.is_fitted:
            raise ValueError("Models must be trained first")
        
        predictions = {}
        
        for target in self.targets:
            if self.models[target].is_fitted:
                predictions[target] = self.models[target].predict(X)
        
        return predictions
    
    def save_models(self, base_filepath: str):
        """Save all trained models."""
        for target in self.targets:
            if self.models[target].is_fitted:
                filepath = f"{base_filepath}_{target}.joblib"
                self.models[target].save_model(filepath)
    
    def load_models(self, base_filepath: str):
        """Load all trained models."""
        for target in self.targets:
            filepath = f"{base_filepath}_{target}.joblib"
            try:
                self.models[target].load_model(filepath)
            except FileNotFoundError:
                logger.warning(f"Model file not found: {filepath}")
        
        self.is_fitted = True