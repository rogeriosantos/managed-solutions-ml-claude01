# CNC Machine Learning Application

A comprehensive machine learning application for CNC downtime prediction and operator performance optimization using data from CIMCO MDC Software.

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- MySQL database (Railway hosted)
- 10GB+ RAM recommended for full dataset analysis

### Installation

1. **Clone and navigate to the project:**
```bash
cd cnc_ml_project
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Configure database connection:**
```bash
cp .env.example .env
# Edit .env with your Railway MySQL credentials
```

4. **Test the installation:**
```bash
python cli.py status
```

## 🎯 Getting Started Options

### Option 1: Command Line Interface (Recommended for beginners)

```bash
# Check system status
python cli.py status

# Explore your data (start with 1000 records)
python cli.py explore --limit 1000 --save

# Generate interactive dashboard
python cli.py dashboard

# Train efficiency prediction model
python cli.py train --model efficiency --optimize

# Get operator recommendations
python cli.py recommend MACHINE_001 PART_123
```

### Option 2: Main Python Script

```bash
python main.py
```
This runs a complete analysis pipeline and shows key insights.

### Option 3: Jupyter Notebooks (Best for exploration)

```bash
jupyter notebook

# Open these notebooks in order:
# 1. notebooks/01_data_exploration.ipynb
# 2. notebooks/02_feature_engineering.ipynb
```

### Option 4: Python API

```python
from src.data.database import DatabaseManager
from src.data.preprocessing import DataPreprocessor
from src.models.efficiency_predictor import EfficiencyPredictor

# Load data
db = DatabaseManager()
df = db.get_all_data(limit=5000)

# Preprocess
preprocessor = DataPreprocessor()
df_clean = preprocessor.validate_raw_data(df)
df_features = preprocessor.create_derived_features(df_clean)

# Train model
predictor = EfficiencyPredictor()
results = predictor.train(df_features)
print(f"Model R² Score: {results['test_metrics']['r2']:.3f}")
```

## 📊 What You'll Get

### Immediate Insights
- **Data Quality Report**: Issues in your dataset
- **Performance Metrics**: Efficiency across machines/operators  
- **Top Performers**: Best machine-operator combinations
- **Time Patterns**: Peak performance hours and days

### Machine Learning Models
- **Downtime Prediction**: Which category will occur next (>85% accuracy)
- **Duration Estimation**: How long jobs will take (<10% error)
- **Efficiency Forecasting**: Expected performance levels
- **Anomaly Detection**: Unusual patterns and outliers

### Operational Analytics
- **Operator Analysis**: Skills, specializations, learning curves
- **Performance Matrix**: 3D analysis (Operator × Machine × Part)
- **Recommendations**: Optimal job assignments
- **Interactive Dashboards**: Visual performance monitoring

## 🗃️ Database Setup

The application expects a MySQL database with the `joblog_ob` table structure:

```sql
-- Your existing table structure from CIMCO MDC
-- No changes needed to your database
-- The app handles all data quality issues automatically
```

Update your `.env` file with Railway credentials:
```
DB_HOST=your-railway-host.railway.app
DB_PORT=3306
DB_NAME=railway
DB_USER=root  
DB_PASSWORD=your-password
```

## 📈 Example Output

```
==========================================
DATA SUMMARY
==========================================
Total Records: 9,847
Unique Machines: 12
Unique Operators: 23
Unique Parts: 156
Date Range: 2023-01-15 to 2024-11-30

==========================================
PERFORMANCE METRICS  
==========================================
Average Efficiency: 0.734
High Efficiency Jobs (>80%): 3,247 (33.0%)

Top 5 Machines by Efficiency:
  HAAS_VF2: 0.823 efficiency (1,234 jobs)
  MAZAK_QT200: 0.789 efficiency (987 jobs)
  DMG_CTX310: 0.756 efficiency (1,456 jobs)

==========================================
MODEL PERFORMANCE
==========================================  
Test R² Score: 0.867
Test RMSE: 0.089
Training Samples: 7,877
Features Used: 127
```

## 📁 Project Structure

```
cnc_ml_project/
├── main.py                 # Main application script
├── cli.py                  # Command line interface
├── src/
│   ├── data/               # Database & preprocessing
│   │   ├── database.py     # MySQL connection
│   │   ├── preprocessing.py # Data cleaning
│   │   └── auxiliary_tables.py # Performance summaries
│   ├── models/             # ML models
│   │   ├── downtime_classifier.py
│   │   ├── duration_regressor.py
│   │   ├── efficiency_predictor.py
│   │   ├── operator_performance.py
│   │   └── anomaly_detector.py
│   ├── analytics/          # Performance analysis
│   │   ├── performance_matrix.py
│   │   ├── recommendations.py
│   │   └── visualizations.py
│   └── utils/              # Configuration & helpers
├── notebooks/              # Jupyter analysis notebooks
├── tests/                  # Unit tests
└── requirements.txt        # Python dependencies
```

## 🔧 Configuration

Key settings in `.env`:

```bash
# Database (Required)
DB_HOST=your-railway-host.railway.app
DB_PASSWORD=your-password

# Performance Tuning
DB_POOL_SIZE=10            # Connection pool size
MODEL_RANDOM_STATE=42      # Reproducible results
EFFICIENCY_THRESHOLD=0.8   # High performance threshold

# Optional Features  
ENABLE_MONITORING=true     # Performance monitoring
LOG_LEVEL=INFO            # Logging detail level
```

## 🎯 Use Cases

### For Production Managers
```bash
# Daily performance check
python cli.py explore --limit 500

# Generate weekly dashboard  
python cli.py dashboard --output weekly_report

# Get assignment recommendations
python cli.py recommend MACHINE_001 PART_456
```

### For Data Analysts
```bash
# Jupyter notebooks for deep analysis
jupyter notebook

# Train and optimize models
python cli.py train --model efficiency --optimize --save-model

# Full data exploration
python cli.py explore --limit 0 --save  # No limit = all data
```

### For Operators
- Interactive dashboards show personal performance trends
- Skill development recommendations  
- Optimal machine-part assignments

## 🚨 Common Issues & Solutions

### Database Connection Failed
```bash
# Check credentials in .env
python cli.py status

# Test connection manually
python -c "from src.data.database import DatabaseManager; print(DatabaseManager().test_connection())"
```

### Not Enough Data
```bash
# Check record count
python cli.py status

# Try with smaller sample first
python cli.py explore --limit 100
```

### Model Training Errors
```bash
# Start without optimization
python cli.py train --model efficiency

# Check data quality
python main.py  # Shows data quality report
```

## 📞 Support

1. **Check the Jupyter notebooks** for detailed explanations
2. **Run `python cli.py status`** for system diagnostics  
3. **Review the logs** for specific error messages
4. **Start with small data samples** (--limit 1000) for testing

## 🎯 Next Steps

Once you have the basic system running:

1. **Analyze your full dataset** (remove limits)
2. **Train production models** with optimization enabled
3. **Set up automated reporting** using the CLI commands
4. **Deploy prediction API** for real-time recommendations
5. **Integrate with your existing systems** via the Python API

The application is designed to scale from initial exploration to full production deployment!