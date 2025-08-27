"""Downtime classification model for predicting downtime categories."""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import xgboost as xgb
import logging
import joblib

from ..utils.config import config
from ..utils.helpers import setup_logging, get_downtime_categories

logger = setup_logging()


class DowntimeClassifier:
    """Multi-class classifier for predicting next downtime category."""
    
    def __init__(self, model_type: str = 'xgboost'):
        """Initialize downtime classifier."""
        self.model_type = model_type
        self.model = None
        self.label_encoder = LabelEncoder()
        self.feature_names = []
        self.class_weights = None
        self.is_fitted = False
        
        self.config = config.get_model_config()
        self.downtime_categories = get_downtime_categories()
        
        # Initialize model based on type
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the ML model based on type."""
        if self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=self.config['random_state'],
                n_jobs=-1
            )
        
        elif self.model_type == 'xgboost':
            self.model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.config['random_state'],
                n_jobs=-1
            )
        
        elif self.model_type == 'gradient_boosting':
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                random_state=self.config['random_state']
            )
        
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def prepare_training_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare data for training downtime classification model."""
        logger.info("Preparing training data for downtime classification...")
        
        # Create target variable: dominant downtime category
        downtime_data = df[self.downtime_categories].copy()
        
        # Find the category with maximum time for each job
        df['dominant_downtime'] = downtime_data.idxmax(axis=1)
        
        # Only keep records where there's actual downtime
        df['total_downtime'] = downtime_data.sum(axis=1)
        df_with_downtime = df[df['total_downtime'] > 0].copy()
        
        # Handle case where all downtime is zero (no downtime)
        df.loc[df['total_downtime'] == 0, 'dominant_downtime'] = 'NoDowntime'
        
        logger.info(f"Found {len(df_with_downtime)} records with downtime")
        
        # Create features for next downtime prediction
        features_df = self._create_downtime_features(df)
        
        # Create target for "next" downtime (shift target by 1)
        features_df = features_df.sort_values(['machine', 'OperatorName', 'StartTime'])
        features_df['next_downtime'] = features_df.groupby(['machine', 'OperatorName'])['dominant_downtime'].shift(-1)
        
        # Remove rows without next downtime information
        training_data = features_df.dropna(subset=['next_downtime']).copy()
        
        logger.info(f"Training data prepared: {len(training_data)} samples")
        
        # Separate features and target
        feature_cols = [col for col in training_data.columns if col not in 
                       ['dominant_downtime', 'next_downtime', 'StartTime', 'EndTime'] + self.downtime_categories]
        
        X = training_data[feature_cols]
        y = training_data['next_downtime']
        
        return X, y
    
    def _create_downtime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features for downtime prediction."""
        features_df = df.copy()
        
        # Historical downtime patterns (rolling averages)
        for category in self.downtime_categories:
            if category in features_df.columns:
                # Rolling mean of downtime category by machine
                features_df[f'{category}_machine_avg'] = features_df.groupby('machine')[category].transform(
                    lambda x: x.rolling(window=5, min_periods=1).mean()
                )
                
                # Rolling mean by operator
                features_df[f'{category}_operator_avg'] = features_df.groupby('OperatorName')[category].transform(
                    lambda x: x.rolling(window=5, min_periods=1).mean()
                )
        
        # Time since last occurrence of each downtime type
        for category in self.downtime_categories:
            if category in features_df.columns:
                features_df[f'time_since_{category}'] = self._calculate_time_since_last_occurrence(
                    features_df, category
                )
        
        # Part complexity indicators
        if 'PartNumber' in features_df.columns:
            part_setup_times = features_df.groupby('PartNumber')['SetupTime'].mean()
            features_df['part_avg_setup_time'] = features_df['PartNumber'].map(part_setup_times)
        
        # Machine utilization features
        features_df['machine_utilization'] = features_df['RunningTime'] / (features_df['RunningTime'] + features_df['total_downtime'])
        
        # Operator experience with machine
        features_df['operator_machine_jobs'] = features_df.groupby(['OperatorName', 'machine']).cumcount() + 1
        
        # Recent performance trend
        features_df['recent_efficiency_trend'] = features_df.groupby(['machine', 'OperatorName'])['efficiency'].transform(
            lambda x: x.rolling(window=3, min_periods=1).mean()
        )
        
        return features_df
    
    def _calculate_time_since_last_occurrence(self, df: pd.DataFrame, category: str) -> pd.Series:
        """Calculate time since last occurrence of specific downtime category."""
        df_sorted = df.sort_values(['machine', 'OperatorName', 'StartTime'])
        
        # Find jobs where this downtime category occurred
        category_occurred = (df_sorted[category] > 0)
        
        # Calculate cumulative job count since last occurrence
        time_since = df_sorted.groupby(['machine', 'OperatorName']).apply(
            lambda group: self._jobs_since_last_occurrence(group[category_occurred.loc[group.index]])
        ).reset_index(level=[0, 1], drop=True)
        
        return time_since.reindex(df.index).fillna(999)  # Large number if never occurred
    
    def _jobs_since_last_occurrence(self, occurred_series: pd.Series) -> pd.Series:
        """Calculate jobs since last occurrence for a group."""
        result = pd.Series(index=occurred_series.index, dtype=float)
        last_occurrence_idx = None
        
        for idx in occurred_series.index:
            if occurred_series.loc[idx]:
                last_occurrence_idx = idx
                result.loc[idx] = 0
            else:
                if last_occurrence_idx is not None:
                    result.loc[idx] = idx - last_occurrence_idx
                else:
                    result.loc[idx] = 999  # Never occurred
        
        return result
    
    def train(self, X: pd.DataFrame, y: pd.Series, optimize_hyperparameters: bool = True) -> Dict[str, Any]:
        """Train the downtime classification model."""
        logger.info(f"Training {self.model_type} downtime classifier...")
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        # Encode target labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Calculate class weights to handle imbalanced data
        classes = np.unique(y_encoded)
        class_weights = compute_class_weight('balanced', classes=classes, y=y_encoded)
        self.class_weights = dict(zip(classes, class_weights))
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, 
            test_size=self.config['test_size'],
            random_state=self.config['random_state'],
            stratify=y_encoded
        )
        
        logger.info(f"Training set: {len(X_train)}, Test set: {len(X_test)}")
        logger.info(f"Class distribution: {np.bincount(y_encoded)}")
        
        # Set class weights in model
        if hasattr(self.model, 'class_weight'):
            self.model.class_weight = self.class_weights
        
        # Hyperparameter optimization
        if optimize_hyperparameters:
            logger.info("Optimizing hyperparameters...")
            self.model = self._optimize_hyperparameters(X_train, y_train)
        
        # Train final model
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)
        
        # Cross-validation score
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=5, scoring='accuracy')
        
        # Predictions for detailed evaluation
        y_pred = self.model.predict(X_test)
        
        # Classification report
        target_names = self.label_encoder.classes_
        class_report = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
        
        self.is_fitted = True
        
        results = {
            'train_accuracy': train_score,
            'test_accuracy': test_score,
            'cv_accuracy_mean': cv_scores.mean(),
            'cv_accuracy_std': cv_scores.std(),
            'classification_report': class_report,
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'feature_importance': self.get_feature_importance()
        }
        
        logger.info(f"Model training complete. Test accuracy: {test_score:.3f}")
        
        return results
    
    def _optimize_hyperparameters(self, X_train: pd.DataFrame, y_train: np.ndarray):
        """Optimize model hyperparameters using grid search."""
        if self.model_type == 'random_forest':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        
        elif self.model_type == 'xgboost':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0]
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
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best CV score: {grid_search.best_score_:.3f}")
        
        return grid_search.best_estimator_
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict downtime categories."""
        if not self.is_fitted:
            raise ValueError("Model must be trained first")
        
        # Ensure features match training
        X_features = X[self.feature_names]
        
        # Get predictions
        y_pred_encoded = self.model.predict(X_features)
        
        # Decode labels
        y_pred = self.label_encoder.inverse_transform(y_pred_encoded)
        
        return y_pred
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probability for each downtime category."""
        if not self.is_fitted:
            raise ValueError("Model must be trained first")
        
        X_features = X[self.feature_names]
        return self.model.predict_proba(X_features)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores."""
        if not self.is_fitted:
            raise ValueError("Model must be trained first")
        
        if hasattr(self.model, 'feature_importances_'):
            importance_dict = dict(zip(self.feature_names, self.model.feature_importances_))
            # Sort by importance
            return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
        else:
            logger.warning("Model doesn't support feature importance")
            return {}
    
    def get_prediction_confidence(self, X: pd.DataFrame) -> np.ndarray:
        """Get confidence scores for predictions."""
        probabilities = self.predict_proba(X)
        return np.max(probabilities, axis=1)
    
    def save_model(self, filepath: str):
        """Save trained model to file."""
        if not self.is_fitted:
            raise ValueError("Model must be trained first")
        
        model_data = {
            'model': self.model,
            'label_encoder': self.label_encoder,
            'feature_names': self.feature_names,
            'class_weights': self.class_weights,
            'model_type': self.model_type
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained model from file."""
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.label_encoder = model_data['label_encoder']
        self.feature_names = model_data['feature_names']
        self.class_weights = model_data['class_weights']
        self.model_type = model_data['model_type']
        self.is_fitted = True
        
        logger.info(f"Model loaded from {filepath}")
    
    def analyze_downtime_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze downtime patterns in the dataset."""
        logger.info("Analyzing downtime patterns...")
        
        analysis = {}
        
        # Overall downtime distribution
        downtime_totals = df[self.downtime_categories].sum()
        analysis['downtime_distribution'] = downtime_totals.sort_values(ascending=False).to_dict()
        
        # Downtime by machine
        machine_downtime = df.groupby('machine')[self.downtime_categories].mean()
        analysis['machine_downtime_patterns'] = machine_downtime.to_dict()
        
        # Downtime by operator
        operator_downtime = df[df['OperatorName'] != 'Unknown'].groupby('OperatorName')[self.downtime_categories].mean()
        analysis['operator_downtime_patterns'] = operator_downtime.to_dict()
        
        # Time-based patterns
        if 'hour' in df.columns:
            hourly_downtime = df.groupby('hour')[self.downtime_categories].mean()
            analysis['hourly_downtime_patterns'] = hourly_downtime.to_dict()
        
        # Most common downtime sequences
        df_sorted = df.sort_values(['machine', 'StartTime'])
        df_sorted['dominant_downtime'] = df_sorted[self.downtime_categories].idxmax(axis=1)
        
        # Find common sequences of 2
        downtime_sequences = []
        for machine in df_sorted['machine'].unique():
            machine_data = df_sorted[df_sorted['machine'] == machine]
            dominant_sequence = machine_data['dominant_downtime'].tolist()
            
            for i in range(len(dominant_sequence) - 1):
                downtime_sequences.append((dominant_sequence[i], dominant_sequence[i + 1]))
        
        from collections import Counter
        common_sequences = Counter(downtime_sequences).most_common(10)
        analysis['common_downtime_sequences'] = common_sequences
        
        logger.info("Downtime pattern analysis complete")
        return analysis