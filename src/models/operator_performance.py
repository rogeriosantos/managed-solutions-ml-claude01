"""Operator performance analysis and modeling."""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import logging

from ..utils.helpers import setup_logging, calculate_performance_score, safe_divide

logger = setup_logging()


class OperatorPerformanceAnalyzer:
    """Comprehensive operator performance analysis and clustering."""

    def __init__(self):
        """Initialize operator performance analyzer."""
        self.scaler = StandardScaler()
        self.performance_features = []
        self.operator_profiles = {}
        self.performance_clusters = {}
        self.is_fitted = False

    def analyze_operator_performance(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Comprehensive operator performance analysis."""
        logger.info("Starting comprehensive operator performance analysis...")

        # Filter out unknown operators
        operator_data = df[df["OperatorName"] != "Unknown"].copy()

        if len(operator_data) == 0:
            logger.warning("No operator data available")
            return {}

        analysis = {}

        # Basic performance metrics
        analysis["basic_metrics"] = self._calculate_basic_metrics(operator_data)

        # Skill specialization analysis
        analysis["skill_specialization"] = self._analyze_skill_specialization(
            operator_data
        )

        # Learning curve analysis
        analysis["learning_curves"] = self._analyze_learning_curves(operator_data)

        # Consistency analysis
        analysis["consistency_metrics"] = self._analyze_consistency(operator_data)

        # Versatility analysis
        analysis["versatility_metrics"] = self._analyze_versatility(operator_data)

        # Performance clustering
        analysis["performance_clusters"] = self._cluster_operator_performance(
            operator_data
        )

        # Comparative analysis
        analysis["comparative_rankings"] = self._create_operator_rankings(operator_data)

        # Time-based performance patterns
        analysis["time_patterns"] = self._analyze_time_patterns(operator_data)

        logger.info("Operator performance analysis complete")
        return analysis

    def _calculate_basic_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate basic performance metrics for each operator."""
        logger.info("Calculating basic performance metrics...")

        metrics = (
            df.groupby("OperatorName")
            .agg(
                {
                    "efficiency": ["mean", "std", "min", "max", "count"],
                    "JobDuration": ["mean", "sum"],
                    "RunningTime": ["mean", "sum"],
                    "PartsProduced": ["sum", "mean"],
                    "SetupTime": ["mean", "sum"],
                    "total_downtime": ["mean", "sum"],
                    "machine": "nunique",
                    "PartNumber": "nunique",
                    "StartTime": ["min", "max"],
                }
            )
            .round(2)
        )

        # Flatten column names
        metrics.columns = ["_".join(col).strip() for col in metrics.columns]

        # Calculate derived metrics
        metrics["total_jobs"] = metrics["efficiency_count"]
        metrics["avg_efficiency"] = metrics["efficiency_mean"]
        metrics["efficiency_consistency"] = 1 / (
            1 + metrics["efficiency_std"]
        )  # Higher is more consistent
        metrics["machines_operated"] = metrics["machine_nunique"]
        metrics["unique_parts"] = metrics["PartNumber_nunique"]

        # Calculate productivity metrics
        metrics["parts_per_hour"] = safe_divide(
            metrics["PartsProduced_sum"] * 3600, metrics["RunningTime_sum"]
        )

        # Calculate experience span
        metrics["experience_days"] = (
            pd.to_datetime(metrics["StartTime_max"])
            - pd.to_datetime(metrics["StartTime_min"])
        ).dt.days

        metrics["jobs_per_day"] = safe_divide(
            metrics["total_jobs"], metrics["experience_days"] + 1
        )

        return metrics.to_dict("index")

    def _analyze_skill_specialization(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze operator skill specialization by machine and part."""
        logger.info("Analyzing skill specialization...")

        specialization = {}

        # Machine specialization
        machine_perf = (
            df.groupby(["OperatorName", "machine"])["efficiency"]
            .agg(["mean", "count"])
            .reset_index()
        )
        machine_perf = machine_perf[machine_perf["count"] >= 3]  # At least 3 jobs

        machine_specialization = {}
        for operator in machine_perf["OperatorName"].unique():
            operator_machines = machine_perf[machine_perf["OperatorName"] == operator]
            if len(operator_machines) > 0:
                best_machine = operator_machines.loc[operator_machines["mean"].idxmax()]
                machine_specialization[operator] = {
                    "best_machine": best_machine["machine"],
                    "best_efficiency": best_machine["mean"],
                    "job_count": best_machine["count"],
                }

        specialization["machine_specialization"] = machine_specialization

        # Part specialization
        part_perf = (
            df.groupby(["OperatorName", "PartNumber"])["efficiency"]
            .agg(["mean", "count"])
            .reset_index()
        )
        part_perf = part_perf[part_perf["count"] >= 2]  # At least 2 jobs

        part_specialization = {}
        for operator in part_perf["OperatorName"].unique():
            operator_parts = part_perf[part_perf["OperatorName"] == operator]
            if len(operator_parts) > 0:
                best_parts = operator_parts.nlargest(3, "mean")  # Top 3 parts
                part_specialization[operator] = best_parts[
                    ["PartNumber", "mean", "count"]
                ].to_dict("records")

        specialization["part_specialization"] = part_specialization

        # Calculate specialization scores
        specialization_scores = {}
        for operator in df["OperatorName"].unique():
            operator_data = df[df["OperatorName"] == operator]

            # Machine specialization score (how much better on best machine)
            machine_efficiencies = operator_data.groupby("machine")["efficiency"].mean()
            if len(machine_efficiencies) > 1:
                spec_score = (
                    machine_efficiencies.max() - machine_efficiencies.mean()
                ) / machine_efficiencies.std()
            else:
                spec_score = 0

            specialization_scores[operator] = {
                "machine_specialization_score": spec_score,
                "versatility_score": len(machine_efficiencies),
                "total_experience": len(operator_data),
            }

        specialization["specialization_scores"] = specialization_scores

        return specialization

    def _analyze_learning_curves(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze operator learning curves and skill development."""
        logger.info("Analyzing learning curves...")

        learning_analysis = {}

        # Sort by operator and time
        df_sorted = df.sort_values(["OperatorName", "StartTime"])

        # Calculate job sequence number for each operator
        df_sorted["job_sequence"] = df_sorted.groupby("OperatorName").cumcount() + 1

        learning_curves = {}
        improvement_rates = {}

        for operator in df_sorted["OperatorName"].unique():
            operator_data = df_sorted[df_sorted["OperatorName"] == operator].copy()

            if len(operator_data) >= 10:  # Need sufficient data for learning curve
                # Calculate moving average efficiency
                window_size = min(5, len(operator_data) // 3)
                operator_data["efficiency_ma"] = (
                    operator_data["efficiency"]
                    .rolling(window=window_size, min_periods=1)
                    .mean()
                )

                # Calculate learning rate (slope of efficiency over time)
                x = operator_data["job_sequence"].values
                y = operator_data["efficiency_ma"].values

                if len(x) > 1:
                    slope, intercept = np.polyfit(x, y, 1)
                    improvement_rate = (
                        slope * 100
                    )  # Convert to percentage improvement per job
                else:
                    improvement_rate = 0

                learning_curves[operator] = {
                    "initial_efficiency": operator_data["efficiency"].iloc[:5].mean(),
                    "recent_efficiency": operator_data["efficiency"].iloc[-5:].mean(),
                    "improvement_rate": improvement_rate,
                    "total_improvement": y[-1] - y[0] if len(y) > 1 else 0,
                    "job_count": len(operator_data),
                    "experience_span_days": (
                        operator_data["StartTime"].max()
                        - operator_data["StartTime"].min()
                    ).days,
                }

        learning_analysis["learning_curves"] = learning_curves

        # Identify fast learners
        fast_learners = {
            k: v
            for k, v in learning_curves.items()
            if v["improvement_rate"] > 0.01
            and v["job_count"] >= 10  # >1% improvement per job
        }

        learning_analysis["fast_learners"] = dict(
            sorted(
                fast_learners.items(),
                key=lambda x: x[1]["improvement_rate"],
                reverse=True,
            )[:5]
        )

        return learning_analysis

    def _analyze_consistency(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze operator performance consistency."""
        logger.info("Analyzing performance consistency...")

        consistency = {}

        # Overall consistency metrics
        operator_consistency = df.groupby("OperatorName").agg(
            {
                "efficiency": [
                    "std",
                    "mean",
                    lambda x: x.quantile(0.25),
                    lambda x: x.quantile(0.75),
                ]
            }
        )

        operator_consistency.columns = [
            "efficiency_std",
            "efficiency_mean",
            "efficiency_q25",
            "efficiency_q75",
        ]

        # Calculate consistency scores
        operator_consistency["consistency_score"] = 1 / (
            1 + operator_consistency["efficiency_std"]
        )

        operator_consistency["interquartile_range"] = (
            operator_consistency["efficiency_q75"]
            - operator_consistency["efficiency_q25"]
        )

        consistency["overall_consistency"] = operator_consistency.to_dict("index")

        # Machine-specific consistency
        machine_consistency = (
            df.groupby(["OperatorName", "machine"])["efficiency"]
            .agg(["std", "count"])
            .reset_index()
        )
        machine_consistency = machine_consistency[machine_consistency["count"] >= 3]

        operator_machine_consistency = {}
        for operator in machine_consistency["OperatorName"].unique():
            operator_machines = machine_consistency[
                machine_consistency["OperatorName"] == operator
            ]
            most_consistent = operator_machines.loc[operator_machines["std"].idxmin()]
            least_consistent = operator_machines.loc[operator_machines["std"].idxmax()]

            operator_machine_consistency[operator] = {
                "most_consistent_machine": {
                    "machine": most_consistent["machine"],
                    "std_dev": most_consistent["std"],
                    "job_count": most_consistent["count"],
                },
                "least_consistent_machine": {
                    "machine": least_consistent["machine"],
                    "std_dev": least_consistent["std"],
                    "job_count": least_consistent["count"],
                },
            }

        consistency["machine_consistency"] = operator_machine_consistency

        return consistency

    def _analyze_versatility(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze operator versatility across machines and parts."""
        logger.info("Analyzing operator versatility...")

        versatility = {}

        # Calculate versatility metrics
        operator_versatility = df.groupby("OperatorName").agg(
            {"machine": "nunique", "PartNumber": "nunique", "efficiency": "mean"}
        )

        # Calculate versatility scores
        max_machines = operator_versatility["machine"].max()
        max_parts = operator_versatility["PartNumber"].max()

        operator_versatility["machine_versatility_score"] = (
            operator_versatility["machine"] / max_machines
        )
        operator_versatility["part_versatility_score"] = (
            operator_versatility["PartNumber"] / max_parts
        )
        operator_versatility["overall_versatility_score"] = (
            operator_versatility["machine_versatility_score"] * 0.6
            + operator_versatility["part_versatility_score"] * 0.4
        )

        versatility["versatility_scores"] = operator_versatility.to_dict("index")

        # Cross-training potential analysis
        cross_training_potential = {}

        for operator in df["OperatorName"].unique():
            operator_data = df[df["OperatorName"] == operator]

            # Machines they've worked on
            current_machines = set(operator_data["machine"].unique())
            all_machines = set(df["machine"].unique())
            potential_machines = all_machines - current_machines

            # Suggest machines based on similarity to current ones
            if potential_machines:
                # Simple recommendation based on similar setup times
                current_avg_setup = (
                    operator_data.groupby("machine")["SetupTime"].mean().mean()
                )

                machine_similarities = {}
                for machine in potential_machines:
                    machine_data = df[df["machine"] == machine]
                    machine_avg_setup = machine_data["SetupTime"].mean()
                    similarity = 1 / (
                        1 + abs(current_avg_setup - machine_avg_setup) / 1000
                    )
                    machine_similarities[machine] = similarity

                # Recommend top 3 similar machines
                recommended = dict(
                    sorted(
                        machine_similarities.items(), key=lambda x: x[1], reverse=True
                    )[:3]
                )

                cross_training_potential[operator] = {
                    "current_machines": list(current_machines),
                    "recommended_machines": recommended,
                    "versatility_potential": len(potential_machines)
                    / len(all_machines),
                }

        versatility["cross_training_potential"] = cross_training_potential

        return versatility

    def _cluster_operator_performance(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Cluster operators based on performance characteristics."""
        logger.info("Clustering operator performance...")

        # Prepare features for clustering
        operator_features = df.groupby("OperatorName").agg(
            {
                "efficiency": ["mean", "std"],
                "SetupTime": "mean",
                "total_downtime": "mean",
                "PartsProduced": "mean",
                "JobDuration": "mean",
                "machine": "nunique",
                "PartNumber": "nunique",
            }
        )

        # Flatten columns
        operator_features.columns = [
            "_".join(col).strip() for col in operator_features.columns
        ]

        # Add derived features
        operator_features["parts_per_hour"] = safe_divide(
            operator_features["PartsProduced_mean"] * 3600,
            operator_features["JobDuration_mean"],
        )

        operator_features["consistency_score"] = 1 / (
            1 + operator_features["efficiency_std"]
        )
        operator_features["versatility_score"] = (
            operator_features["machine_nunique"]
            * operator_features["PartNumber_nunique"]
        )

        # Select features for clustering
        clustering_features = [
            "efficiency_mean",
            "consistency_score",
            "SetupTime_mean",
            "parts_per_hour",
            "versatility_score",
            "total_downtime_mean",
        ]

        X = operator_features[clustering_features].fillna(
            operator_features[clustering_features].median()
        )

        # Standardize features
        X_scaled = self.scaler.fit_transform(X)

        # Determine optimal number of clusters
        silhouette_scores = []
        K_range = range(2, min(8, len(X) // 2))

        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X_scaled)
            silhouette_scores.append(silhouette_score(X_scaled, labels))

        if silhouette_scores:
            optimal_k = K_range[np.argmax(silhouette_scores)]
        else:
            optimal_k = 3  # Default

        # Perform final clustering
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X_scaled)

        # Add cluster labels to operator features
        operator_features["cluster"] = cluster_labels

        # Analyze cluster characteristics
        cluster_analysis = {}
        for cluster_id in range(optimal_k):
            cluster_operators = operator_features[
                operator_features["cluster"] == cluster_id
            ]

            cluster_analysis[f"cluster_{cluster_id}"] = {
                "size": len(cluster_operators),
                "operators": cluster_operators.index.tolist(),
                "characteristics": {
                    "avg_efficiency": cluster_operators["efficiency_mean"].mean(),
                    "avg_consistency": cluster_operators["consistency_score"].mean(),
                    "avg_versatility": cluster_operators["versatility_score"].mean(),
                    "avg_setup_time": cluster_operators["SetupTime_mean"].mean(),
                    "avg_parts_per_hour": cluster_operators["parts_per_hour"].mean(),
                },
            }

        # Label clusters based on characteristics
        cluster_names = self._label_clusters(cluster_analysis)

        clustering_results = {
            "optimal_clusters": optimal_k,
            "silhouette_score": (
                silhouette_scores[optimal_k - 2] if silhouette_scores else 0
            ),
            "cluster_analysis": cluster_analysis,
            "cluster_names": cluster_names,
            "operator_clusters": operator_features[["cluster"]].to_dict("index"),
        }

        self.performance_clusters = clustering_results
        return clustering_results

    def _label_clusters(self, cluster_analysis: Dict[str, Any]) -> Dict[str, str]:
        """Label clusters based on their characteristics."""
        cluster_names = {}

        for cluster_id, analysis in cluster_analysis.items():
            chars = analysis["characteristics"]

            # Determine cluster type based on characteristics
            if chars["avg_efficiency"] > 0.8 and chars["avg_consistency"] > 0.8:
                cluster_names[cluster_id] = "High Performers"
            elif chars["avg_versatility"] > 5 and chars["avg_efficiency"] > 0.7:
                cluster_names[cluster_id] = "Versatile Operators"
            elif chars["avg_consistency"] > 0.8:
                cluster_names[cluster_id] = "Consistent Workers"
            elif chars["avg_setup_time"] < 300:  # Less than 5 minutes
                cluster_names[cluster_id] = "Setup Specialists"
            elif chars["avg_parts_per_hour"] > chars.get(
                "overall_avg_parts_per_hour", 10
            ):
                cluster_names[cluster_id] = "High Productivity"
            else:
                cluster_names[cluster_id] = "Developing Operators"

        return cluster_names

    def _create_operator_rankings(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Create operator rankings across different metrics."""
        logger.info("Creating operator rankings...")

        # Calculate ranking metrics
        operator_metrics = (
            df.groupby("OperatorName")
            .agg(
                {
                    "efficiency": "mean",
                    "SetupTime": "mean",
                    "parts_per_hour": "mean",
                    "total_downtime": "mean",
                    "machine": "nunique",
                    "JobDuration": "count",
                }
            )
            .round(3)
        )

        # Calculate composite performance score
        operator_metrics["performance_score"] = operator_metrics.apply(
            lambda row: calculate_performance_score(
                efficiency=row["efficiency"],
                quality_score=min(row["parts_per_hour"] / 10, 1.0),  # Normalize
                consistency_score=1
                / (1 + row["total_downtime"] / 1000),  # Lower downtime is better
            ),
            axis=1,
        )

        rankings = {}

        # Individual metric rankings
        rankings["efficiency_ranking"] = operator_metrics.nlargest(10, "efficiency")[
            ["efficiency"]
        ].to_dict("index")
        rankings["setup_speed_ranking"] = operator_metrics.nsmallest(10, "SetupTime")[
            ["SetupTime"]
        ].to_dict("index")
        rankings["productivity_ranking"] = operator_metrics.nlargest(
            10, "parts_per_hour"
        )[["parts_per_hour"]].to_dict("index")
        rankings["versatility_ranking"] = operator_metrics.nlargest(10, "machine")[
            ["machine"]
        ].to_dict("index")
        rankings["overall_performance_ranking"] = operator_metrics.nlargest(
            10, "performance_score"
        )[["performance_score"]].to_dict("index")

        # Experience-based rankings (minimum job count filter)
        experienced_operators = operator_metrics[
            operator_metrics["JobDuration"] >= 10
        ]  # At least 10 jobs
        if len(experienced_operators) > 0:
            rankings["experienced_top_performers"] = experienced_operators.nlargest(
                5, "performance_score"
            )[["performance_score", "JobDuration"]].to_dict("index")

        return rankings

    def _analyze_time_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze time-based performance patterns."""
        logger.info("Analyzing time-based performance patterns...")

        time_patterns = {}

        # Performance by hour of day
        if "hour" in df.columns:
            hourly_performance = (
                df.groupby(["OperatorName", "hour"])["efficiency"].mean().reset_index()
            )

            # Find best and worst hours for each operator
            operator_hourly = {}
            for operator in hourly_performance["OperatorName"].unique():
                operator_hours = hourly_performance[
                    hourly_performance["OperatorName"] == operator
                ]
                if len(operator_hours) > 1:
                    best_hour = operator_hours.loc[
                        operator_hours["efficiency"].idxmax()
                    ]
                    worst_hour = operator_hours.loc[
                        operator_hours["efficiency"].idxmin()
                    ]

                    operator_hourly[operator] = {
                        "best_hour": {
                            "hour": best_hour["hour"],
                            "efficiency": best_hour["efficiency"],
                        },
                        "worst_hour": {
                            "hour": worst_hour["hour"],
                            "efficiency": worst_hour["efficiency"],
                        },
                        "hour_variation": operator_hours["efficiency"].std(),
                    }

            time_patterns["hourly_patterns"] = operator_hourly

        # Performance by day of week
        if "day_of_week" in df.columns:
            daily_performance = (
                df.groupby(["OperatorName", "day_of_week"])["efficiency"]
                .mean()
                .reset_index()
            )

            operator_daily = {}
            for operator in daily_performance["OperatorName"].unique():
                operator_days = daily_performance[
                    daily_performance["OperatorName"] == operator
                ]
                if len(operator_days) > 1:
                    operator_daily[operator] = {
                        "monday_efficiency": (
                            operator_days[operator_days["day_of_week"] == 0][
                                "efficiency"
                            ].iloc[0]
                            if len(operator_days[operator_days["day_of_week"] == 0]) > 0
                            else None
                        ),
                        "friday_efficiency": (
                            operator_days[operator_days["day_of_week"] == 4][
                                "efficiency"
                            ].iloc[0]
                            if len(operator_days[operator_days["day_of_week"] == 4]) > 0
                            else None
                        ),
                        "daily_variation": operator_days["efficiency"].std(),
                    }

            time_patterns["daily_patterns"] = operator_daily

        return time_patterns

    def get_operator_profile(
        self, operator_name: str, df: pd.DataFrame
    ) -> Dict[str, Any]:
        """Get comprehensive profile for a specific operator."""
        operator_data = df[df["OperatorName"] == operator_name]

        if len(operator_data) == 0:
            return {"error": f"No data found for operator {operator_name}"}

        profile = {
            "basic_stats": self._calculate_basic_metrics(operator_data).get(
                operator_name, {}
            ),
            "machines_operated": operator_data["machine"].unique().tolist(),
            "parts_produced": operator_data["PartNumber"].unique().tolist(),
            "total_jobs": len(operator_data),
            "efficiency_distribution": {
                "mean": operator_data["efficiency"].mean(),
                "median": operator_data["efficiency"].median(),
                "std": operator_data["efficiency"].std(),
                "min": operator_data["efficiency"].min(),
                "max": operator_data["efficiency"].max(),
            },
        }

        # Add cluster information if available
        if hasattr(self, "performance_clusters") and self.performance_clusters:
            operator_cluster_info = self.performance_clusters.get(
                "operator_clusters", {}
            ).get(operator_name, {})
            if "cluster" in operator_cluster_info:
                cluster_id = f"cluster_{operator_cluster_info['cluster']}"
                cluster_name = self.performance_clusters.get("cluster_names", {}).get(
                    cluster_id, "Unknown"
                )
                profile["performance_cluster"] = {
                    "cluster_id": cluster_id,
                    "cluster_name": cluster_name,
                }

        return profile
