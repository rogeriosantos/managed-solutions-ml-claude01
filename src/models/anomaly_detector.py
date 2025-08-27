"""Anomaly detection for manufacturing operations."""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import logging

from ..utils.helpers import setup_logging

logger = setup_logging()


class ManufacturingAnomalyDetector:
    """Detect anomalies in manufacturing operations."""
    
    def __init__(self, method: str = 'isolation_forest'):
        """Initialize anomaly detector."""
        self.method = method
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.anomaly_threshold = 0.1  # 10% contamination rate
        self.is_fitted = False
        
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize anomaly detection model."""
        if self.method == 'isolation_forest':
            self.model = IsolationForest(
                contamination=self.anomaly_threshold,
                random_state=42,
                n_jobs=-1
            )
        
        elif self.method == 'one_class_svm':
            self.model = OneClassSVM(
                nu=self.anomaly_threshold,
                kernel='rbf',
                gamma='scale'
            )
        
        else:
            raise ValueError(f"Unsupported method: {self.method}")
    
    def prepare_anomaly_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for anomaly detection."""
        logger.info("Preparing features for anomaly detection...")
        
        features_df = df.copy()
        
        # Basic efficiency and time features
        if 'efficiency' not in features_df.columns:
            features_df['efficiency'] = features_df['RunningTime'] / features_df['JobDuration']
            features_df['efficiency'] = features_df['efficiency'].clip(0, 1)
        
        # Downtime ratios
        downtime_cols = ['SetupTime', 'WaitingSetupTime', 'NotFeedingTime', 'AdjustmentTime',
                        'DressingTime', 'ToolingTime', 'EngineeringTime', 'MaintenanceTime',
                        'BuyInTime', 'BreakShiftChangeTime', 'IdleTime']
        
        for col in downtime_cols:
            if col in features_df.columns:
                features_df[f'{col}_ratio'] = features_df[col] / features_df['JobDuration']
        
        # Productivity metrics
        if 'PartsProduced' in features_df.columns:
            features_df['parts_per_hour'] = features_df['PartsProduced'] * 3600 / features_df['RunningTime']
            features_df['parts_per_hour'] = features_df['parts_per_hour'].replace([np.inf, -np.inf], 0)
        
        # Deviation from machine averages
        features_df = self._add_deviation_features(features_df)
        
        # Deviation from operator averages
        features_df = self._add_operator_deviation_features(features_df)
        
        # Time-based anomaly indicators
        features_df = self._add_temporal_anomaly_features(features_df)
        
        logger.info("Anomaly feature preparation complete")
        return features_df
    
    def _add_deviation_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add features showing deviation from machine norms."""
        df = df.copy()
        
        # Machine baseline metrics
        machine_baselines = df.groupby('machine').agg({
            'efficiency': 'mean',
            'JobDuration': 'mean',
            'SetupTime': 'mean',
            'total_downtime': 'mean'
        })
        
        # Calculate deviations
        for metric in ['efficiency', 'JobDuration', 'SetupTime', 'total_downtime']:
            baseline_col = f'{metric}_machine_baseline'
            deviation_col = f'{metric}_machine_deviation'
            
            df[baseline_col] = df['machine'].map(machine_baselines[metric])
            df[deviation_col] = (df[metric] - df[baseline_col]) / df[baseline_col]
        
        return df
    
    def _add_operator_deviation_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add features showing deviation from operator norms."""
        df = df.copy()
        
        # Operator baseline metrics (excluding unknown operators)
        operator_data = df[df['OperatorName'] != 'Unknown']
        if len(operator_data) > 0:
            operator_baselines = operator_data.groupby('OperatorName').agg({
                'efficiency': 'mean',
                'SetupTime': 'mean',
                'total_downtime': 'mean'
            })
            
            # Calculate deviations for known operators
            for metric in ['efficiency', 'SetupTime', 'total_downtime']:
                baseline_col = f'{metric}_operator_baseline'
                deviation_col = f'{metric}_operator_deviation'
                
                df[baseline_col] = df['OperatorName'].map(operator_baselines[metric])
                df[deviation_col] = (df[metric] - df[baseline_col]) / df[baseline_col]
                df[deviation_col] = df[deviation_col].fillna(0)  # Fill for unknown operators
        
        return df
    
    def _add_temporal_anomaly_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based anomaly indicators."""
        df = df.copy()
        
        # Sort by machine and time
        df = df.sort_values(['machine', 'StartTime'])
        
        # Time gaps between consecutive jobs on same machine
        df['time_gap'] = df.groupby('machine')['StartTime'].diff().dt.total_seconds()
        df['time_gap'] = df['time_gap'].fillna(0)
        
        # Unusually long or short gaps
        df['unusual_long_gap'] = (df['time_gap'] > 86400).astype(int)  # > 24 hours
        df['unusual_short_gap'] = (df['time_gap'] < 60).astype(int)    # < 1 minute
        
        # Jobs starting at unusual times
        if 'hour' in df.columns:
            df['off_hours_job'] = ((df['hour'] < 6) | (df['hour'] > 22)).astype(int)
        elif 'StartTime' in df.columns:
            df['hour'] = pd.to_datetime(df['StartTime']).dt.hour
            df['off_hours_job'] = ((df['hour'] < 6) | (df['hour'] > 22)).astype(int)
        
        # Weekend work
        if 'day_of_week' in df.columns:
            df['weekend_work'] = (df['day_of_week'] >= 5).astype(int)
        elif 'StartTime' in df.columns:
            df['day_of_week'] = pd.to_datetime(df['StartTime']).dt.dayofweek
            df['weekend_work'] = (df['day_of_week'] >= 5).astype(int)
        
        return df
    
    def fit(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Train the anomaly detection model."""
        logger.info(f"Training {self.method} anomaly detector...")
        
        # Prepare features
        features_df = self.prepare_anomaly_features(df)
        
        # Select features for anomaly detection
        feature_cols = self._select_anomaly_features(features_df)
        X = features_df[feature_cols]
        
        # Handle missing values and infinities
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median())
        
        # Store feature names
        self.feature_names = feature_cols
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model.fit(X_scaled)
        
        # Predict anomalies on training data for analysis
        anomaly_scores = self.model.decision_function(X_scaled)
        predictions = self.model.predict(X_scaled)
        
        # Analyze results
        n_anomalies = (predictions == -1).sum()
        anomaly_rate = n_anomalies / len(predictions)
        
        # Get anomaly details
        anomaly_indices = np.where(predictions == -1)[0]
        anomaly_details = features_df.iloc[anomaly_indices][['machine', 'OperatorName', 'StartTime', 'efficiency', 'total_downtime']].copy()
        anomaly_details['anomaly_score'] = anomaly_scores[anomaly_indices]
        
        self.is_fitted = True
        
        results = {
            'method': self.method,
            'n_samples': len(X),
            'n_features': len(feature_cols),
            'n_anomalies': n_anomalies,
            'anomaly_rate': anomaly_rate,
            'anomaly_details': anomaly_details.to_dict('records'),
            'feature_importance': self._get_feature_importance(X, predictions)
        }
        
        logger.info(f"Anomaly detection training complete. Found {n_anomalies} anomalies ({anomaly_rate:.1%})")
        
        return results
    
    def _select_anomaly_features(self, df: pd.DataFrame) -> List[str]:
        """Select features relevant for anomaly detection."""
        # Focus on performance and deviation features
        feature_patterns = [
            'efficiency', 'parts_per_hour', 'total_downtime',
            '_ratio', '_deviation', '_gap', '_job', '_work'
        ]
        
        # Exclude identifier and time columns
        exclude_patterns = [
            'StartTime', 'EndTime', 'JobNumber', 'State', 'OpNumber',
            'EmpID', 'OperatorName', 'machine', 'PartNumber'
        ]
        
        feature_cols = []
        for col in df.columns:
            # Include if matches feature patterns and doesn't match exclude patterns
            if any(pattern in col for pattern in feature_patterns):
                if not any(exclude in col for exclude in exclude_patterns):
                    if df[col].dtype in ['int64', 'float64']:
                        feature_cols.append(col)
        
        return feature_cols
    
    def predict(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Predict anomalies in new data."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        # Prepare features
        features_df = self.prepare_anomaly_features(df)
        X = features_df[self.feature_names]
        
        # Handle missing values
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median())
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Make predictions
        predictions = self.model.predict(X_scaled)
        scores = self.model.decision_function(X_scaled)
        
        return predictions, scores
    
    def detect_specific_anomalies(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect specific types of anomalies."""
        logger.info("Detecting specific anomaly types...")
        
        anomalies = {}
        
        # Efficiency anomalies
        anomalies['efficiency_anomalies'] = self._detect_efficiency_anomalies(df)
        
        # Duration anomalies
        anomalies['duration_anomalies'] = self._detect_duration_anomalies(df)
        
        # Downtime anomalies
        anomalies['downtime_anomalies'] = self._detect_downtime_anomalies(df)
        
        # Operator behavior anomalies
        anomalies['operator_anomalies'] = self._detect_operator_anomalies(df)
        
        # Machine performance anomalies
        anomalies['machine_anomalies'] = self._detect_machine_anomalies(df)
        
        # Temporal anomalies
        anomalies['temporal_anomalies'] = self._detect_temporal_anomalies(df)
        
        return anomalies
    
    def _detect_efficiency_anomalies(self, df: pd.DataFrame) -> List[Dict]:
        """Detect efficiency-related anomalies."""
        anomalies = []
        
        # Extremely low efficiency
        low_efficiency = df[df['efficiency'] < 0.1]  # Less than 10%
        for _, row in low_efficiency.iterrows():
            anomalies.append({
                'type': 'extremely_low_efficiency',
                'machine': row['machine'],
                'operator': row['OperatorName'],
                'start_time': row['StartTime'],
                'efficiency': row['efficiency'],
                'severity': 'high'
            })
        
        # Impossible efficiency (>100%)
        high_efficiency = df[df['efficiency'] > 1.1]  # More than 110%
        for _, row in high_efficiency.iterrows():
            anomalies.append({
                'type': 'impossible_efficiency',
                'machine': row['machine'],
                'operator': row['OperatorName'],
                'start_time': row['StartTime'],
                'efficiency': row['efficiency'],
                'severity': 'medium'
            })
        
        return anomalies
    
    def _detect_duration_anomalies(self, df: pd.DataFrame) -> List[Dict]:
        """Detect duration-related anomalies."""
        anomalies = []
        
        # Extremely long jobs
        job_duration_q99 = df['JobDuration'].quantile(0.99)
        long_jobs = df[df['JobDuration'] > job_duration_q99 * 2]  # More than 2x 99th percentile
        
        for _, row in long_jobs.iterrows():
            anomalies.append({
                'type': 'extremely_long_job',
                'machine': row['machine'],
                'operator': row['OperatorName'],
                'start_time': row['StartTime'],
                'duration_hours': row['JobDuration'] / 3600,
                'severity': 'medium'
            })
        
        # Extremely short jobs
        short_jobs = df[df['JobDuration'] < 60]  # Less than 1 minute
        
        for _, row in short_jobs.iterrows():
            anomalies.append({
                'type': 'extremely_short_job',
                'machine': row['machine'],
                'operator': row['OperatorName'],
                'start_time': row['StartTime'],
                'duration_seconds': row['JobDuration'],
                'severity': 'low'
            })
        
        return anomalies
    
    def _detect_downtime_anomalies(self, df: pd.DataFrame) -> List[Dict]:
        """Detect downtime-related anomalies."""
        anomalies = []
        
        # Extremely high idle time
        high_idle = df[df['IdleTime'] > 86400]  # More than 24 hours
        
        for _, row in high_idle.iterrows():
            anomalies.append({
                'type': 'extreme_idle_time',
                'machine': row['machine'],
                'operator': row['OperatorName'],
                'start_time': row['StartTime'],
                'idle_hours': row['IdleTime'] / 3600,
                'severity': 'high'
            })
        
        # Multiple high downtime categories in single job
        downtime_cols = ['SetupTime', 'MaintenanceTime', 'AdjustmentTime', 'ToolingTime']
        for _, row in df.iterrows():
            high_downtimes = sum(1 for col in downtime_cols if col in df.columns and row[col] > 3600)  # > 1 hour
            
            if high_downtimes >= 3:
                anomalies.append({
                    'type': 'multiple_high_downtimes',
                    'machine': row['machine'],
                    'operator': row['OperatorName'],
                    'start_time': row['StartTime'],
                    'high_downtime_count': high_downtimes,
                    'severity': 'medium'
                })
        
        return anomalies
    
    def _detect_operator_anomalies(self, df: pd.DataFrame) -> List[Dict]:
        """Detect operator behavior anomalies."""
        anomalies = []
        
        if 'OperatorName' in df.columns:
            # Operator performing significantly worse than usual
            operator_performance = df[df['OperatorName'] != 'Unknown'].groupby('OperatorName')['efficiency'].agg(['mean', 'std'])
            
            for _, row in df.iterrows():
                if row['OperatorName'] != 'Unknown':
                    operator_stats = operator_performance.loc[row['OperatorName']]
                    if row['efficiency'] < operator_stats['mean'] - 3 * operator_stats['std']:
                        anomalies.append({
                            'type': 'operator_performance_drop',
                            'machine': row['machine'],
                            'operator': row['OperatorName'],
                            'start_time': row['StartTime'],
                            'efficiency': row['efficiency'],
                            'expected_efficiency': operator_stats['mean'],
                            'severity': 'medium'
                        })
        
        return anomalies
    
    def _detect_machine_anomalies(self, df: pd.DataFrame) -> List[Dict]:
        """Detect machine performance anomalies."""
        anomalies = []
        
        # Machine performing significantly worse than usual
        machine_performance = df.groupby('machine')['efficiency'].agg(['mean', 'std'])
        
        for _, row in df.iterrows():
            machine_stats = machine_performance.loc[row['machine']]
            if row['efficiency'] < machine_stats['mean'] - 3 * machine_stats['std']:
                anomalies.append({
                    'type': 'machine_performance_drop',
                    'machine': row['machine'],
                    'operator': row['OperatorName'],
                    'start_time': row['StartTime'],
                    'efficiency': row['efficiency'],
                    'expected_efficiency': machine_stats['mean'],
                    'severity': 'high'
                })
        
        return anomalies
    
    def _detect_temporal_anomalies(self, df: pd.DataFrame) -> List[Dict]:
        """Detect time-based anomalies."""
        anomalies = []
        
        # Jobs running at unusual times
        if 'StartTime' in df.columns:
            df_time = df.copy()
            df_time['hour'] = pd.to_datetime(df_time['StartTime']).dt.hour
            df_time['day_of_week'] = pd.to_datetime(df_time['StartTime']).dt.dayofweek
            
            # Very early morning jobs (2-5 AM)
            early_jobs = df_time[(df_time['hour'] >= 2) & (df_time['hour'] <= 5)]
            for _, row in early_jobs.iterrows():
                anomalies.append({
                    'type': 'unusual_time_job',
                    'machine': row['machine'],
                    'operator': row['OperatorName'],
                    'start_time': row['StartTime'],
                    'hour': row['hour'],
                    'severity': 'low'
                })
            
            # Weekend jobs
            weekend_jobs = df_time[df_time['day_of_week'] >= 5]
            for _, row in weekend_jobs.iterrows():
                anomalies.append({
                    'type': 'weekend_job',
                    'machine': row['machine'],
                    'operator': row['OperatorName'],
                    'start_time': row['StartTime'],
                    'day_of_week': row['day_of_week'],
                    'severity': 'low'
                })
        
        return anomalies
    
    def _get_feature_importance(self, X: pd.DataFrame, predictions: np.ndarray) -> Dict[str, float]:
        """Calculate feature importance for anomaly detection."""
        if self.method == 'isolation_forest':
            # For isolation forest, we can't directly get feature importance
            # Use a simple approach based on feature variance in anomalies vs normal
            normal_mask = predictions == 1
            anomaly_mask = predictions == -1
            
            if anomaly_mask.sum() > 0 and normal_mask.sum() > 0:
                importance = {}
                for i, feature in enumerate(self.feature_names):
                    normal_var = X.iloc[normal_mask, i].var()
                    anomaly_var = X.iloc[anomaly_mask, i].var()
                    
                    # Features with higher variance in anomalies are more important
                    if normal_var > 0:
                        importance[feature] = anomaly_var / normal_var
                    else:
                        importance[feature] = 0
                
                # Normalize
                total_importance = sum(importance.values())
                if total_importance > 0:
                    importance = {k: v/total_importance for k, v in importance.items()}
                
                return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
        
        return {}
    
    def generate_anomaly_report(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive anomaly detection report."""
        logger.info("Generating anomaly detection report...")
        
        if not self.is_fitted:
            fit_results = self.fit(df)
        else:
            predictions, scores = self.predict(df)
            fit_results = {'predictions': predictions, 'scores': scores}
        
        # Detect specific anomalies
        specific_anomalies = self.detect_specific_anomalies(df)
        
        # Summarize anomaly types
        anomaly_summary = {}
        for anomaly_type, anomaly_list in specific_anomalies.items():
            anomaly_summary[anomaly_type] = {
                'count': len(anomaly_list),
                'severity_breakdown': {}
            }
            
            # Count by severity
            for anomaly in anomaly_list:
                severity = anomaly.get('severity', 'unknown')
                anomaly_summary[anomaly_type]['severity_breakdown'][severity] = \
                    anomaly_summary[anomaly_type]['severity_breakdown'].get(severity, 0) + 1
        
        report = {
            'model_results': fit_results,
            'specific_anomalies': specific_anomalies,
            'anomaly_summary': anomaly_summary,
            'total_anomalies': sum(len(anomalies) for anomalies in specific_anomalies.values()),
            'recommendations': self._generate_recommendations(specific_anomalies)
        }
        
        return report
    
    def _generate_recommendations(self, anomalies: Dict[str, List]) -> List[str]:
        """Generate recommendations based on detected anomalies."""
        recommendations = []
        
        # Check for efficiency anomalies
        if anomalies.get('efficiency_anomalies'):
            recommendations.append("Investigate jobs with extremely low efficiency for potential process issues")
            recommendations.append("Review data quality for jobs showing impossible efficiency values")
        
        # Check for duration anomalies
        if anomalies.get('duration_anomalies'):
            recommendations.append("Review extremely long jobs for potential equipment or process issues")
            recommendations.append("Verify data collection for very short jobs")
        
        # Check for downtime anomalies
        if anomalies.get('downtime_anomalies'):
            recommendations.append("Investigate machines with extreme idle times for scheduling optimization")
            recommendations.append("Review jobs with multiple high downtimes for process inefficiencies")
        
        # Check for operator anomalies
        if anomalies.get('operator_anomalies'):
            recommendations.append("Provide additional training or support for operators showing performance drops")
        
        # Check for machine anomalies
        if anomalies.get('machine_anomalies'):
            recommendations.append("Schedule maintenance for machines showing performance degradation")
        
        # Check for temporal anomalies
        if anomalies.get('temporal_anomalies'):
            recommendations.append("Review scheduling policies for jobs running at unusual times")
        
        return recommendations