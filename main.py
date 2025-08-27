#!/usr/bin/env python3
"""
Main entry point for CNC ML Project
Demonstrates core functionality and provides example usage.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

# Import project modules
from src.data.database import DatabaseManager
from src.data.preprocessing import DataPreprocessor
from src.data.auxiliary_tables import AuxiliaryTableManager
from src.models.efficiency_predictor import EfficiencyPredictor
from src.models.downtime_classifier import DowntimeClassifier
from src.models.operator_performance import OperatorPerformanceAnalyzer
from src.analytics.performance_matrix import PerformanceMatrixAnalyzer
from src.analytics.recommendations import AssignmentRecommendationEngine
from src.analytics.visualizations import CNCAnalyticsDashboard
from src.utils.helpers import setup_logging


def main():
    """Main application entry point."""
    # Set up logging
    logger = setup_logging("INFO")
    logger.info("Starting CNC ML Application")

    try:
        # Step 1: Connect to Database
        logger.info("Connecting to database...")
        db = DatabaseManager()

        if not db.test_connection():
            logger.error(
                "Database connection failed. Please check your credentials in .env file."
            )
            return

        logger.info("Database connection successful!")

        # Step 2: Load and explore data
        logger.info("Loading sample data...")
        df_raw = db.get_all_data(limit=10000)  # Start with 1k records for demo

        if len(df_raw) == 0:
            logger.error("No data found in database. Please check your database setup.")
            return

        logger.info(f"Loaded {len(df_raw)} records")
        logger.info(
            f"Date range: {df_raw['StartTime'].min()} to {df_raw['StartTime'].max()}"
        )

        # Step 3: Data Quality Summary
        logger.info("Generating data quality summary...")
        summary = db.get_summary_stats()

        print("\n" + "=" * 60)
        print("DATA SUMMARY")
        print("=" * 60)
        print(
            f"Total Records: {summary.get('total_records', {}).get('count', 'N/A'):,}"
        )
        print(
            f"Unique Machines: {summary.get('unique_machines', {}).get('count', 'N/A')}"
        )
        print(
            f"Unique Operators: {summary.get('unique_operators', {}).get('count', 'N/A')}"
        )
        print(f"Unique Parts: {summary.get('unique_parts', {}).get('count', 'N/A')}")

        date_range = summary.get("date_range", {})
        if date_range:
            print(
                f"Date Range: {date_range.get('min_date')} to {date_range.get('max_date')}"
            )

        # Step 4: Data Preprocessing
        logger.info("Preprocessing data...")
        preprocessor = DataPreprocessor()

        # Clean and validate data
        df_clean = preprocessor.validate_raw_data(df_raw)
        logger.info(f"After cleaning: {len(df_clean)} records")

        # Create engineered features
        df_features = preprocessor.create_derived_features(df_clean)
        logger.info(f"Features created. Shape: {df_features.shape}")

        # Step 5: Basic Analytics
        logger.info("Running basic performance analytics...")

        print("\n" + "=" * 60)
        print("PERFORMANCE METRICS")
        print("=" * 60)

        if "efficiency" in df_features.columns:
            print(f"Average Efficiency: {df_features['efficiency'].mean():.3f}")
            print(f"Efficiency Std Dev: {df_features['efficiency'].std():.3f}")
            high_eff_jobs = (df_features["efficiency"] > 0.8).sum()
            print(
                f"High Efficiency Jobs (>80%): {high_eff_jobs} ({high_eff_jobs/len(df_features)*100:.1f}%)"
            )

        # Machine performance
        machine_perf = (
            df_features.groupby("machine")["efficiency"].agg(["mean", "count"]).round(3)
        )
        machine_perf = machine_perf.sort_values("mean", ascending=False)

        print(f"\nTop 5 Machines by Efficiency:")
        for machine, row in machine_perf.head().iterrows():
            print(f"  {machine}: {row['mean']:.3f} efficiency ({row['count']} jobs)")

        # Operator performance
        if "OperatorName" in df_features.columns:
            operator_data = df_features[df_features["OperatorName"] != "Unknown"]
            if len(operator_data) > 0:
                operator_perf = (
                    operator_data.groupby("OperatorName")["efficiency"]
                    .agg(["mean", "count"])
                    .round(3)
                )
                operator_perf = operator_perf[
                    operator_perf["count"] >= 3
                ]  # At least 3 jobs
                operator_perf = operator_perf.sort_values("mean", ascending=False)

                print(f"\nTop 5 Operators by Efficiency (min 3 jobs):")
                for operator, row in operator_perf.head().iterrows():
                    print(
                        f"  {operator}: {row['mean']:.3f} efficiency ({row['count']} jobs)"
                    )

        # Step 6: Advanced Analytics (Optional)
        if len(df_features) >= 50:  # Need sufficient data
            logger.info("Running advanced analytics...")

            # Operator performance analysis
            analyzer = OperatorPerformanceAnalyzer()
            operator_analysis = analyzer.analyze_operator_performance(df_features)

            if "basic_metrics" in operator_analysis:
                print("\n" + "=" * 60)
                print("OPERATOR INSIGHTS")
                print("=" * 60)

                basic_metrics = operator_analysis["basic_metrics"]
                if basic_metrics:
                    top_operator = max(
                        basic_metrics.items(),
                        key=lambda x: x[1].get("avg_efficiency", 0),
                    )
                    print(
                        f"Top Performer: {top_operator[0]} ({top_operator[1].get('avg_efficiency', 0):.3f} efficiency)"
                    )

                    total_operators = len(basic_metrics)
                    print(f"Total Active Operators: {total_operators}")

        # Step 7: Sample Predictions (if enough data)
        if len(df_features) >= 100:
            logger.info("Training sample efficiency prediction model...")

            try:
                efficiency_predictor = EfficiencyPredictor()
                results = efficiency_predictor.train(
                    df_features, optimize_hyperparameters=False
                )

                print("\n" + "=" * 60)
                print("MODEL PERFORMANCE")
                print("=" * 60)
                print(f"Test R² Score: {results['test_metrics']['r2']:.3f}")
                print(f"Test RMSE: {results['test_metrics']['rmse']:.3f}")
                print(f"Training Samples: {results['n_samples']:,}")
                print(f"Features Used: {results['n_features']}")

            except Exception as e:
                logger.warning(f"Model training skipped: {e}")

        # # Step 8: Recommendations
        # print("\n" + "=" * 60)
        # print("RECOMMENDATIONS")
        # print("=" * 60)
        # print("1. Use the Jupyter notebooks for detailed analysis:")
        # print("   - notebooks/01_data_exploration.ipynb")
        # print("   - notebooks/02_feature_engineering.ipynb")
        # print()
        # print("2. Train full models with more data:")
        # print("   - Load complete dataset (remove limit=1000)")
        # print("   - Enable hyperparameter optimization")
        # print()
        # print("3. Set up production pipeline:")
        # print("   - Configure database credentials")
        # print("   - Schedule regular model updates")
        # print("   - Deploy prediction API")

        logger.info("Application completed successfully!")

    except Exception as e:
        logger.error(f"Application error: {e}")
        raise


if __name__ == "__main__":
    main()
