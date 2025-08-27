# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CNC Machine Learning Application for downtime prediction and operator performance optimization using data from CIMCO MDC Software. Built with Python, FastAPI, and scikit-learn for production ML deployment.

## Architecture

### Core Structure
- **src/data/**: Database connectivity and data preprocessing
  - `database.py`: MySQL connection via Railway hosting
  - `preprocessing.py`: Data cleaning and feature engineering
  - `auxiliary_tables.py`: Performance summary tables
- **src/models/**: ML models for different prediction tasks
  - `efficiency_predictor.py`: Job efficiency forecasting
  - `downtime_classifier.py`: Downtime category prediction
  - `duration_regressor.py`: Job duration estimation
  - `operator_performance.py`: Operator analytics
  - `anomaly_detector.py`: Outlier detection
- **src/analytics/**: Business intelligence and recommendations
  - `performance_matrix.py`: 3D analysis (Operator × Machine × Part)
  - `recommendations.py`: Optimal job assignment engine
  - `visualizations.py`: Interactive dashboards
- **api/**: Production FastAPI server with Redis caching
- **notebooks/**: Jupyter analysis workflows

### Data Flow
1. Raw data from MySQL (`joblog_ob` table)
2. Data cleaning and validation (`DataPreprocessor`)
3. Feature engineering (efficiency, time features, categorical encoding)
4. Model training with hyperparameter optimization
5. Predictions via CLI, API, or notebooks

## Common Development Commands

### Environment Setup
```bash
pip install -r requirements.txt
cp .env.example .env  # Configure database credentials
```

### CLI Interface (Primary development tool)
```bash
# System status and diagnostics
python cli.py status

# Data exploration with samples
python cli.py explore --limit 1000 --save

# Model training
python cli.py train --model efficiency --optimize --save-model
python cli.py train --model downtime

# Interactive dashboards
python cli.py dashboard --output results

# Operator recommendations
python cli.py recommend MACHINE_001 PART_123 --top 5
```

### Direct Scripts
```bash
# Complete analysis pipeline
python main.py

# Start production API server
python start_api.py
```

### Development Workflows
```bash
# Jupyter analysis (recommended for exploration)
jupyter notebook
# Open: notebooks/01_data_exploration.ipynb
# Then: notebooks/02_feature_engineering.ipynb

# Docker deployment
docker-compose up -d
```

### Testing and Code Quality
```bash
# Run tests (if test files exist in tests/)
pytest tests/

# Code formatting and linting
black src/ api/ *.py
flake8 src/ api/ *.py
mypy src/ api/
```

## Configuration

### Database Setup (.env file)
```
DB_HOST=your-railway-host.railway.app
DB_PORT=3306
DB_NAME=railway
DB_USER=root
DB_PASSWORD=your-password
DB_POOL_SIZE=10
```

### Model Configuration
- Random state: 42 (reproducible results)
- Test/validation split: 80/20 with 10% validation
- Efficiency threshold: 0.8 for high performance classification
- Feature engineering includes time-based features, categorical encoding

## Key Implementation Details

### Data Requirements
- Minimum 50 records for basic analysis
- 100+ records for model training
- Works with existing CIMCO MDC `joblog_ob` table structure
- Handles missing operators, malformed dates, and data quality issues automatically

### Model Architecture
- **Efficiency Predictor**: XGBoost regression with 127+ engineered features
- **Downtime Classifier**: Multi-class classification for downtime categories
- **Operator Performance**: Statistical analysis with skill profiling
- All models support hyperparameter optimization via optuna

### Production API Features
- FastAPI with async endpoints at `/predict/`, `/recommend/`, `/analytics/`
- Redis caching for predictions (1-4 hour TTL)
- Background task processing with Celery
- Comprehensive health checks and monitoring
- Docker deployment with nginx reverse proxy

### Development Patterns
- All models follow consistent interface: `train()`, `predict()`, `save_model()`, `load_model()`
- Configuration centralized in `src/utils/config.py`
- Logging setup via `src/utils/helpers.setup_logging()`
- Database connections use connection pooling
- Feature engineering is reproducible and cached

## Performance Notes

- Start with `--limit 1000` for initial testing
- Full dataset analysis requires 10GB+ RAM
- Model training with optimization can take 30+ minutes
- API responses cached for performance
- Use `optimize=True` only for final production models