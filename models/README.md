# Models Directory

This directory contains trained machine learning models for the CNC ML project.

## Model Files

- `efficiency_predictor.joblib` - XGBoost model for efficiency prediction
- `downtime_classifier.joblib` - Classification model for downtime prediction
- `operator_performance.joblib` - Operator performance analysis model

## Usage

Models are automatically loaded by the API when needed. If model files don't exist, they will be trained on-demand using available data.
