"""Machine and operator efficiency prediction models."""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import logging
import joblib

from ..utils.config import config
from ..utils.helpers import setup_logging, safe_divide

logger = setup_logging()


class EfficiencyPredictor:
    """Model for predicting machine and operator efficiency."""

    def __init__(self, model_type: str = "xgboost"):
        """Initialize efficiency predictor."""
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.is_fitted = False

        self.config = config.get_model_config()
        self.feature_config = config.get_feature_config()

        self._initialize_model()

    def _initialize_model(self):
        """Initialize the ML model."""
        if self.model_type == "random_forest":
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=self.config["random_state"],
                n_jobs=-1,
            )

        elif self.model_type == "xgboost":
            self.model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.config["random_state"],
                n_jobs=-1,
            )

        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def prepare_efficiency_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for efficiency prediction."""
        logger.info("Preparing efficiency prediction features...")

        features_df = df.copy()

        # Calculate total_downtime from individual downtime columns if not present
        if "total_downtime" not in features_df.columns:
            downtime_cols = [
                "SetupTime",
                "WaitingSetupTime",
                "NotFeedingTime",
                "AdjustmentTime",
                "DressingTime",
                "ToolingTime",
                "EngineeringTime",
                "MaintenanceTime",
                "BuyInTime",
                "BreakShiftChangeTime",
                "IdleTime",
            ]
            # Only use columns that exist in the dataframe
            available_downtime_cols = [
                col for col in downtime_cols if col in features_df.columns
            ]
            if available_downtime_cols:
                features_df["total_downtime"] = features_df[
                    available_downtime_cols
                ].sum(axis=1)
            else:
                # Fallback: use JobDuration - RunningTime
                features_df["total_downtime"] = (
                    features_df["JobDuration"] - features_df["RunningTime"]
                )
                features_df["total_downtime"] = features_df["total_downtime"].clip(
                    lower=0
                )

        # Basic efficiency calculation if not present
        if "efficiency" not in features_df.columns:
            features_df["efficiency"] = safe_divide(
                features_df["RunningTime"], features_df["JobDuration"]
            )
            features_df["efficiency"] = features_df["efficiency"].clip(0, 1)

        # Historical efficiency patterns
        features_df = self._add_historical_efficiency_features(features_df)

        # Machine characteristics
        features_df = self._add_machine_features(features_df)

        # Operator characteristics
        features_df = self._add_operator_features(features_df)

        # Part characteristics
        features_df = self._add_part_features(features_df)

        # Contextual features
        features_df = self._add_contextual_features(features_df)

        # Workload and complexity features
        features_df = self._add_workload_features(features_df)

        logger.info("Efficiency features preparation complete")
        return features_df

    def _add_historical_efficiency_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add historical efficiency patterns."""
        df = df.copy()
        df = df.sort_values(["machine", "OperatorName", "StartTime"])

        # Machine historical efficiency
        for window in [5, 10, 20]:
            df[f"machine_efficiency_rolling_{window}"] = df.groupby("machine")[
                "efficiency"
            ].transform(lambda x: x.rolling(window=window, min_periods=1).mean())

        # Operator historical efficiency
        for window in [5, 10, 20]:
            df[f"operator_efficiency_rolling_{window}"] = df.groupby("OperatorName")[
                "efficiency"
            ].transform(lambda x: x.rolling(window=window, min_periods=1).mean())

        # Machine-Operator combination efficiency
        for window in [3, 5]:
            df[f"machine_operator_efficiency_rolling_{window}"] = df.groupby(
                ["machine", "OperatorName"]
            )["efficiency"].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )

        # Efficiency trend (slope of recent efficiency)
        df["efficiency_trend"] = df.groupby(["machine", "OperatorName"])[
            "efficiency"
        ].transform(
            lambda x: x.rolling(window=5, min_periods=2).apply(
                lambda y: np.polyfit(range(len(y)), y, 1)[0] if len(y) > 1 else 0
            )
        )

        # Efficiency volatility
        df["efficiency_volatility"] = (
            df.groupby(["machine", "OperatorName"])["efficiency"]
            .transform(lambda x: x.rolling(window=10, min_periods=2).std())
            .fillna(0)
        )

        return df

    def _add_machine_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add machine-specific features."""
        df = df.copy()

        # Machine baseline efficiency
        machine_baseline = df.groupby("machine")["efficiency"].mean()
        df["machine_baseline_efficiency"] = df["machine"].map(machine_baseline)

        # Machine complexity (based on average job duration)
        machine_complexity = df.groupby("machine")["JobDuration"].mean()
        df["machine_complexity"] = df["machine"].map(machine_complexity)

        # Machine utilization rate
        def calculate_utilization(group):
            return group["RunningTime"].sum() / (
                group["RunningTime"].sum() + group["total_downtime"].sum()
            )

        # Use agg instead of apply for better performance and to avoid deprecation warnings
        machine_utilization = df.groupby("machine").agg(
            total_running_time=("RunningTime", "sum"),
            total_downtime=("total_downtime", "sum"),
        )
        machine_utilization = machine_utilization["total_running_time"] / (
            machine_utilization["total_running_time"]
            + machine_utilization["total_downtime"]
        )
        df["machine_utilization_rate"] = df["machine"].map(machine_utilization)

        # Machine age proxy (based on maintenance frequency)
        machine_maintenance = df.groupby("machine")["MaintenanceTime"].mean()
        df["machine_maintenance_frequency"] = df["machine"].map(machine_maintenance)

        # Machine specialization (how many different parts it typically runs)
        machine_part_diversity = df.groupby("machine")["PartNumber"].nunique()
        df["machine_part_diversity"] = df["machine"].map(machine_part_diversity)

        return df

    def _add_operator_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add operator-specific features."""
        df = df.copy()

        # Operator baseline efficiency
        operator_baseline = (
            df[df["OperatorName"] != "Unknown"]
            .groupby("OperatorName")["efficiency"]
            .mean()
        )
        df["operator_baseline_efficiency"] = (
            df["OperatorName"].map(operator_baseline).fillna(df["efficiency"].mean())
        )

        # Operator experience (total jobs run)
        operator_experience = df.groupby("OperatorName").cumcount() + 1
        df["operator_total_experience"] = operator_experience

        # Operator experience with this machine
        operator_machine_exp = df.groupby(["OperatorName", "machine"]).cumcount() + 1
        df["operator_machine_experience"] = operator_machine_exp

        # Operator versatility (number of different machines operated)
        operator_versatility = df.groupby("OperatorName")["machine"].nunique()
        df["operator_versatility"] = (
            df["OperatorName"].map(operator_versatility).fillna(1)
        )

        # Operator consistency (std dev of efficiency)
        operator_consistency = (
            df[df["OperatorName"] != "Unknown"]
            .groupby("OperatorName")["efficiency"]
            .std()
        )
        df["operator_consistency"] = (
            df["OperatorName"].map(operator_consistency).fillna(df["efficiency"].std())
        )
        df["operator_consistency"] = 1 / (
            1 + df["operator_consistency"]
        )  # Invert so higher is better

        # Operator learning rate (improvement over time)
        df = df.sort_values(["OperatorName", "machine", "StartTime"])
        df["operator_learning_rate"] = (
            df.groupby(["OperatorName", "machine"])["efficiency"]
            .transform(lambda x: x.diff().rolling(window=5, min_periods=2).mean())
            .fillna(0)
        )

        # Operator setup expertise (average setup time)
        operator_setup_skill = (
            df[df["OperatorName"] != "Unknown"]
            .groupby("OperatorName")["SetupTime"]
            .mean()
        )
        df["operator_setup_expertise"] = (
            df["OperatorName"].map(operator_setup_skill).fillna(df["SetupTime"].mean())
        )
        # Invert so lower setup time = higher expertise
        df["operator_setup_expertise"] = 1 / (1 + df["operator_setup_expertise"] / 1000)

        return df

    def _add_part_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add part-specific features."""
        df = df.copy()

        # Part complexity (based on typical setup time)
        part_complexity = (
            df[df["PartNumber"] != "Unknown"].groupby("PartNumber")["SetupTime"].mean()
        )
        df["part_complexity"] = (
            df["PartNumber"].map(part_complexity).fillna(df["SetupTime"].mean())
        )

        # Part production volume (how often it's made)
        part_frequency = df.groupby("PartNumber").size()
        df["part_production_frequency"] = df["PartNumber"].map(part_frequency).fillna(1)

        # Part typical batch size
        part_batch_size = (
            df[df["PartNumber"] != "Unknown"]
            .groupby("PartNumber")["PartsProduced"]
            .mean()
        )
        df["part_typical_batch_size"] = (
            df["PartNumber"].map(part_batch_size).fillna(df["PartsProduced"].mean())
        )

        # Operator familiarity with part
        operator_part_exp = df.groupby(["OperatorName", "PartNumber"]).cumcount() + 1
        df["operator_part_familiarity"] = operator_part_exp

        # Machine suitability for part (based on historical efficiency)
        machine_part_efficiency = df.groupby(["machine", "PartNumber"])[
            "efficiency"
        ].mean()
        df["machine_part_suitability"] = (
            df.set_index(["machine", "PartNumber"])
            .index.map(machine_part_efficiency)
            .fillna(df["efficiency"].mean())
        )

        return df

    def _add_contextual_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add contextual features that might affect efficiency."""
        df = df.copy()

        # Time-based context
        if "hour" in df.columns:
            # Efficiency by hour of day
            hourly_efficiency = df.groupby("hour")["efficiency"].mean()
            df["expected_hourly_efficiency"] = df["hour"].map(hourly_efficiency)

            # Early morning factor (potential fatigue)
            df["early_morning_factor"] = (df["hour"] < 6).astype(int)

            # Late shift factor
            df["late_shift_factor"] = (df["hour"] > 20).astype(int)

        if "day_of_week" in df.columns:
            # Efficiency by day of week
            daily_efficiency = df.groupby("day_of_week")["efficiency"].mean()
            df["expected_daily_efficiency"] = df["day_of_week"].map(daily_efficiency)

            # Monday blues and Friday effects
            df["monday_effect"] = (df["day_of_week"] == 0).astype(int)
            df["friday_effect"] = (df["day_of_week"] == 4).astype(int)

        # Job sequence effects
        df["jobs_today"] = (
            df.groupby(["machine", "OperatorName", df["StartTime"].dt.date]).cumcount()
            + 1
        )
        df["first_job_of_day"] = (df["jobs_today"] == 1).astype(int)

        # Time since last job (potential warm-up effects)
        df = df.sort_values(["machine", "OperatorName", "StartTime"])
        df["time_since_last_job"] = (
            df.groupby(["machine", "OperatorName"])["StartTime"]
            .diff()
            .dt.total_seconds()
            .fillna(0)
        )
        df["long_break_before"] = (df["time_since_last_job"] > 3600).astype(
            int
        )  # > 1 hour

        return df

    def _add_workload_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add workload and pressure features."""
        df = df.copy()

        # Daily workload
        daily_jobs = df.groupby(["machine", df["StartTime"].dt.date]).size()
        df["daily_job_count"] = (
            df.set_index(["machine", df["StartTime"].dt.date])
            .index.map(daily_jobs)
            .fillna(1)
        )

        # Expected vs actual parts produced (pressure indicator)
        if "PartsProduced" in df.columns:
            expected_parts = df.groupby("PartNumber")["PartsProduced"].mean()
            df["expected_parts"] = df["PartNumber"].map(expected_parts).fillna(1)
            df["parts_pressure"] = df["PartsProduced"] / df["expected_parts"]

        # Rush job indicator (short time since order)
        # This would typically come from order data, but we'll use time patterns
        if "StartTime" in df.columns:
            df["hour_of_start"] = pd.to_datetime(df["StartTime"]).dt.hour
            # Jobs started very early or very late might be rush jobs
            df["potential_rush_job"] = (
                (df["hour_of_start"] < 6) | (df["hour_of_start"] > 22)
            ).astype(int)

        # Machine queue pressure (how many different operators using same machine)
        machine_operator_count = df.groupby(["machine", df["StartTime"].dt.date])[
            "OperatorName"
        ].nunique()
        df["machine_operator_pressure"] = (
            df.set_index(["machine", df["StartTime"].dt.date])
            .index.map(machine_operator_count)
            .fillna(1)
        )

        return df

    def train(
        self, df: pd.DataFrame, optimize_hyperparameters: bool = True
    ) -> Dict[str, Any]:
        """Train the efficiency prediction model."""
        logger.info(f"Training {self.model_type} efficiency predictor...")

        # Prepare features
        features_df = self.prepare_efficiency_features(df)

        # Remove rows with invalid efficiency values
        valid_data = features_df[
            (features_df["efficiency"] >= 0)
            & (features_df["efficiency"] <= 1)
            & (features_df["JobDuration"] > 0)
        ].copy()

        logger.info(f"Valid training samples: {len(valid_data)}")

        # Select features
        feature_cols = self._select_features(valid_data)
        X = valid_data[feature_cols]
        y = valid_data["efficiency"]

        # Handle missing values
        X = X.fillna(X.median())
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(0)

        # Store feature names
        self.feature_names = feature_cols

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.config["test_size"],
            random_state=self.config["random_state"],
        )

        logger.info(f"Training set: {len(X_train)}, Test set: {len(X_test)}")

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Hyperparameter optimization
        if optimize_hyperparameters:
            logger.info("Optimizing hyperparameters...")
            self.model = self._optimize_hyperparameters(X_train_scaled, y_train)

        # Train final model
        self.model.fit(X_train_scaled, y_train)

        # Evaluate model
        train_pred = self.model.predict(X_train_scaled)
        test_pred = self.model.predict(X_test_scaled)

        # Clip predictions to valid efficiency range
        train_pred = np.clip(train_pred, 0, 1)
        test_pred = np.clip(test_pred, 0, 1)

        # Calculate metrics
        train_metrics = self._calculate_metrics(y_train, train_pred)
        test_metrics = self._calculate_metrics(y_test, test_pred)

        self.is_fitted = True

        results = {
            "model_type": self.model_type,
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
            "feature_importance": self.get_feature_importance(),
            "n_features": len(self.feature_names),
            "n_samples": len(valid_data),
            "efficiency_distribution": {
                "mean": y.mean(),
                "std": y.std(),
                "min": y.min(),
                "max": y.max(),
            },
        }

        logger.info(
            f"Efficiency model training complete. Test R²: {test_metrics['r2']:.3f}"
        )

        return results

    def _select_features(self, df: pd.DataFrame) -> List[str]:
        """Select relevant features for efficiency prediction."""
        exclude_cols = [
            "efficiency",
            "StartTime",
            "EndTime",
            "JobNumber",
            "State",
            "OpNumber",
            "EmpID",
            "ActualDuration",
            "RunningTime",
            "JobDuration",
        ]

        feature_cols = []
        for col in df.columns:
            if col not in exclude_cols:
                if df[col].dtype in ["int64", "float64"] or "_encoded" in col:
                    # Exclude highly correlated features
                    if not any(
                        similar in col
                        for similar in ["efficiency_rolling", "efficiency_trend"]
                        if col != similar
                    ):
                        feature_cols.append(col)

        return feature_cols

    def _calculate_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> Dict[str, float]:
        """Calculate regression metrics for efficiency prediction."""
        return {
            "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
            "mae": mean_absolute_error(y_true, y_pred),
            "r2": r2_score(y_true, y_pred),
            "mape": np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 0.01))) * 100,
        }

    def _optimize_hyperparameters(self, X_train: np.ndarray, y_train: np.ndarray):
        """Optimize model hyperparameters."""
        if self.model_type == "random_forest":
            param_grid = {
                "n_estimators": [100, 200, 300],
                "max_depth": [10, 15, 20],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
            }

        elif self.model_type == "xgboost":
            param_grid = {
                "n_estimators": [100, 200, 300],
                "max_depth": [4, 6, 8],
                "learning_rate": [0.05, 0.1, 0.15],
                "subsample": [0.8, 0.9, 1.0],
                "colsample_bytree": [0.8, 0.9, 1.0],
            }

        from sklearn.model_selection import GridSearchCV

        grid_search = GridSearchCV(
            self.model,
            param_grid,
            cv=3,
            scoring="neg_mean_squared_error",
            n_jobs=-1,
            verbose=1,
        )

        grid_search.fit(X_train, y_train)

        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best CV score: {np.sqrt(-grid_search.best_score_):.3f}")

        return grid_search.best_estimator_

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict efficiency for new data."""
        if not self.is_fitted:
            raise ValueError("Model must be trained first")

        # Prepare features
        features_df = self.prepare_efficiency_features(X)
        X_features = features_df[self.feature_names]

        # Handle missing values
        X_features = X_features.fillna(X_features.median())
        X_features = X_features.replace([np.inf, -np.inf], np.nan)
        X_features = X_features.fillna(0)

        # Scale features
        X_scaled = self.scaler.transform(X_features)

        # Make predictions and clip to valid range
        predictions = self.model.predict(X_scaled)
        predictions = np.clip(predictions, 0, 1)

        return predictions

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores."""
        if not self.is_fitted:
            raise ValueError("Model must be trained first")

        if hasattr(self.model, "feature_importances_"):
            importance_dict = dict(
                zip(self.feature_names, self.model.feature_importances_)
            )
            return dict(
                sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
            )
        else:
            return {}

    def analyze_efficiency_factors(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze factors affecting efficiency."""
        logger.info("Analyzing efficiency factors...")

        analysis = {}

        # Efficiency by machine
        machine_efficiency = (
            df.groupby("machine")["efficiency"].agg(["mean", "std", "count"]).round(3)
        )
        analysis["machine_efficiency"] = machine_efficiency.to_dict()

        # Efficiency by operator
        operator_efficiency = (
            df[df["OperatorName"] != "Unknown"]
            .groupby("OperatorName")["efficiency"]
            .agg(["mean", "std", "count"])
            .round(3)
        )
        analysis["operator_efficiency"] = operator_efficiency.to_dict()

        # Efficiency by hour of day
        if "hour" in df.columns:
            hourly_efficiency = df.groupby("hour")["efficiency"].mean().round(3)
            analysis["hourly_efficiency"] = hourly_efficiency.to_dict()

        # Efficiency by day of week
        if "day_of_week" in df.columns:
            daily_efficiency = df.groupby("day_of_week")["efficiency"].mean().round(3)
            analysis["daily_efficiency"] = daily_efficiency.to_dict()

        # Top machine-operator combinations
        combo_efficiency = (
            df.groupby(["machine", "OperatorName"])["efficiency"]
            .agg(["mean", "count"])
            .reset_index()
        )
        combo_efficiency = combo_efficiency[
            combo_efficiency["count"] >= 3
        ]  # At least 3 jobs
        top_combos = combo_efficiency.nlargest(10, "mean")[
            ["machine", "OperatorName", "mean", "count"]
        ]
        analysis["top_machine_operator_combos"] = top_combos.to_dict("records")

        # Efficiency correlation with other factors
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        efficiency_correlations = (
            df[numeric_cols].corr()["efficiency"].abs().sort_values(ascending=False)
        )
        analysis["efficiency_correlations"] = efficiency_correlations.head(10).to_dict()

        logger.info("Efficiency factor analysis complete")
        return analysis

    def save_model(self, filepath: str):
        """Save trained model to file."""
        if not self.is_fitted:
            raise ValueError("Model must be trained first")

        model_data = {
            "model": self.model,
            "scaler": self.scaler,
            "feature_names": self.feature_names,
            "model_type": self.model_type,
        }

        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath: str):
        """Load trained model from file."""
        model_data = joblib.load(filepath)

        self.model = model_data["model"]
        self.scaler = model_data["scaler"]
        self.feature_names = model_data["feature_names"]
        self.model_type = model_data["model_type"]
        self.is_fitted = True

        logger.info(f"Model loaded from {filepath}")
