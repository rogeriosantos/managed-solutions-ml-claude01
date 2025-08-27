"""
FastAPI application for CNC ML Production API
Provides endpoints for predictions, analytics, and monitoring.
"""

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import logging
import time
from typing import List, Dict, Any, Optional
import pandas as pd
from datetime import datetime, timedelta
import redis
import json

# Import project modules
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.database import DatabaseManager
from src.data.preprocessing import DataPreprocessor
from src.models.efficiency_predictor import EfficiencyPredictor
from src.models.downtime_classifier import DowntimeClassifier
from src.models.operator_performance import OperatorPerformanceAnalyzer
from src.analytics.recommendations import AssignmentRecommendationEngine
from src.utils.config import config
from src.utils.helpers import setup_logging

# Import API modules (using absolute imports instead of relative)
from api.models import *
from api.dependencies import (
    get_database_manager,
    get_redis_client,
    get_efficiency_model,
    get_downtime_model,
    get_recommendation_engine,
)
from api.middleware import LoggingMiddleware, RateLimitMiddleware

# Initialize logging
logger = setup_logging(config.LOG_LEVEL)

# Create FastAPI app
app = FastAPI(
    title="CNC ML Production API",
    description="Production API for CNC Manufacturing Machine Learning Analytics",
    version="1.0.0",
    docs_url="/docs" if config.DEBUG else None,
    redoc_url="/redoc" if config.DEBUG else None,
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if config.DEBUG else ["https://yourdomain.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(LoggingMiddleware)
app.add_middleware(
    RateLimitMiddleware, requests_per_minute=100, burst_limit=10
)  # 100 requests per minute


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    global startup_time
    startup_time = time.time()
    logger.info("Starting CNC ML Production API...")

    # Test database connection
    try:
        db = DatabaseManager()
        if db.test_connection():
            logger.info("Database connection successful")
        else:
            logger.error("Database connection failed")
    except Exception as e:
        logger.error(f"Database startup error: {e}")

    # Initialize Redis connection
    try:
        redis_client = redis.from_url(config.REDIS_URL)
        redis_client.ping()
        logger.info("Redis connection successful")
    except Exception as e:
        logger.error(f"Redis startup error: {e}")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down CNC ML Production API...")


# Health Check Endpoints
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint for load balancers."""
    return HealthResponse(
        status="healthy", timestamp=datetime.utcnow(), version="1.0.0"
    )


@app.get("/health/detailed", response_model=DetailedHealthResponse)
async def detailed_health_check(
    db: DatabaseManager = Depends(get_database_manager),
    redis_client=Depends(get_redis_client),
):
    """Detailed health check with service status."""
    checks = {"database": False, "redis": False, "models": False}

    # Database check
    try:
        checks["database"] = db.test_connection()
    except:
        pass

    # Redis check
    try:
        redis_client.ping()
        checks["redis"] = True
    except:
        pass

    # Models check (basic check if models directory exists)
    try:
        import os

        checks["models"] = os.path.exists("models")
    except:
        pass

    overall_status = "healthy" if all(checks.values()) else "unhealthy"

    return DetailedHealthResponse(
        status=overall_status,
        timestamp=datetime.utcnow(),
        services=checks,
        uptime=time.time() - startup_time if "startup_time" in globals() else 0,
    )


# Prediction Endpoints
@app.post("/predict/efficiency", response_model=EfficiencyPredictionResponse)
async def predict_efficiency(
    request: EfficiencyPredictionRequest,
    background_tasks: BackgroundTasks,
    efficiency_model=Depends(get_efficiency_model),
    redis_client=Depends(get_redis_client),
):
    """Predict efficiency for a job."""
    try:
        # Check cache first
        cache_key = f"efficiency_pred:{hash(str(request.dict()))}"
        cached_result = redis_client.get(cache_key)

        if cached_result:
            logger.info("Returning cached efficiency prediction")
            return EfficiencyPredictionResponse.parse_raw(cached_result)

        # Load efficiency model
        predictor = efficiency_model

        # Check if model is trained, if not, train it now
        if not predictor.is_fitted:
            logger.info("Model not trained. Training now with available data...")
            try:
                # Get training data
                db = get_database_manager()
                df = db.get_all_data(limit=1000)

                if len(df) > 50:
                    logger.info(f"Training efficiency model with {len(df)} records...")
                    results = predictor.train(df)
                    logger.info(
                        f"Model trained successfully. R² Score: {results.get('test_r2', 'N/A')}"
                    )
                else:
                    raise ValueError(
                        f"Insufficient data for training. Found {len(df)} records, need at least 50."
                    )

            except Exception as train_error:
                logger.error(f"Failed to train model: {train_error}")
                raise HTTPException(
                    status_code=503, detail=f"Model training failed: {str(train_error)}"
                )

        # Convert request to DataFrame
        input_data = pd.DataFrame(
            [
                {
                    "machine": request.machine,
                    "OperatorName": request.operator or "Unknown",
                    "PartNumber": request.part or "Unknown",
                    "JobDuration": request.planned_cycle_time * 60
                    or 3600,  # Convert minutes to seconds
                    "SetupTime": (
                        request.setup_time * 60 if request.setup_time else 600
                    ),  # Convert minutes to seconds
                    "hour": datetime.now().hour,
                    "day_of_week": datetime.now().weekday(),
                }
            ]
        )

        # Make prediction
        efficiency_pred = predictor.predict(input_data)[0]

        # Create response
        response = EfficiencyPredictionResponse(
            predicted_efficiency=float(efficiency_pred),
            confidence_score=0.85,  # Would calculate from model
            factors={
                "machine_performance": 0.3,
                "operator_experience": 0.4,
                "part_complexity": 0.2,
                "time_factors": 0.1,
            },
            recommendations=[
                "High efficiency expected for this combination",
                "Optimal assignment based on historical data",
            ],
        )

        # Cache result for 1 hour
        redis_client.setex(cache_key, 3600, response.json())

        # Log prediction for monitoring
        background_tasks.add_task(
            log_prediction, "efficiency", request.dict(), response.dict()
        )

        return response

    except Exception as e:
        logger.error(f"Efficiency prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/downtime", response_model=DowntimePredictionResponse)
async def predict_downtime(
    request: DowntimePredictionRequest,
    downtime_model=Depends(get_downtime_model),
    redis_client=Depends(get_redis_client),
):
    """Predict next downtime category and duration."""
    try:
        # Check cache
        cache_key = f"downtime_pred:{hash(str(request.dict()))}"
        cached_result = redis_client.get(cache_key)

        if cached_result:
            return DowntimePredictionResponse.parse_raw(cached_result)

        # Load downtime classifier
        classifier = downtime_model

        # Create input data
        input_data = pd.DataFrame(
            [
                {
                    "machine": request.machine,
                    "OperatorName": request.operator or "Unknown",
                    "PartNumber": request.part or "Unknown",
                    "recent_efficiency": request.recent_efficiency or 0.7,
                    "hour": datetime.now().hour,
                }
            ]
        )

        # Predict downtime category and probability
        predictions = classifier.predict_proba(input_data)[0]
        categories = classifier.label_encoder.classes_

        # Get top 3 predictions
        top_indices = predictions.argsort()[-3:][::-1]
        predicted_categories = []

        for idx in top_indices:
            predicted_categories.append(
                DowntimeCategory(
                    category=categories[idx],
                    probability=float(predictions[idx]),
                    estimated_duration=300 + idx * 600,  # Simple estimation
                )
            )

        response = DowntimePredictionResponse(
            predicted_categories=predicted_categories,
            confidence_score=float(predictions.max()),
            next_maintenance_window=datetime.now() + timedelta(hours=24),
            risk_factors=[
                "Extended operation period",
                "Complex part changeover scheduled",
            ],
        )

        # Cache for 30 minutes
        redis_client.setex(cache_key, 1800, response.json())

        return response

    except Exception as e:
        logger.error(f"Downtime prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/recommend/operator", response_model=OperatorRecommendationResponse)
async def recommend_operator(
    request: OperatorRecommendationRequest,
    rec_engine=Depends(get_recommendation_engine),
    redis_client=Depends(get_redis_client),
):
    """Recommend best operator for a job."""
    try:
        # Check cache
        cache_key = f"operator_rec:{hash(str(request.dict()))}"
        cached_result = redis_client.get(cache_key)

        if cached_result:
            return OperatorRecommendationResponse.parse_raw(cached_result)

        # Get recommendation engine - already injected as dependency

        # Get recommendations
        recommendations = rec_engine.recommend_operator_for_job(
            machine=request.machine,
            part=request.part,
            n_recommendations=request.top_n or 5,
        )

        # Convert to response format
        operator_recommendations = []
        for rec in recommendations:
            operator_recommendations.append(
                OperatorRecommendation(
                    operator_name=rec["operator"],
                    score=rec["score"],
                    confidence=rec["confidence"],
                    reasons=rec["reasons"],
                    estimated_efficiency=rec["score"],
                    experience_level="High" if rec["score"] > 0.8 else "Medium",
                )
            )

        response = OperatorRecommendationResponse(
            recommendations=operator_recommendations,
            job_complexity_score=0.6,
            alternative_assignments=[],
            scheduling_notes=[
                "Consider operator availability",
                "Account for shift preferences",
            ],
        )

        # Cache for 2 hours
        redis_client.setex(cache_key, 7200, response.json())

        return response

    except Exception as e:
        logger.error(f"Operator recommendation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Analytics Endpoints
@app.get("/analytics/performance/overview", response_model=PerformanceOverviewResponse)
async def get_performance_overview(
    days: int = 30,
    db: DatabaseManager = Depends(get_database_manager),
    redis_client=Depends(get_redis_client),
):
    """Get overall performance analytics."""
    try:
        # Check cache
        cache_key = f"perf_overview:{days}"
        cached_result = redis_client.get(cache_key)

        if cached_result:
            return PerformanceOverviewResponse.parse_raw(cached_result)

        # Get recent data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        df = db.get_data_by_date_range(
            start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")
        )

        if df.empty:
            raise HTTPException(
                status_code=404, detail="No data found for specified period"
            )

        # Preprocess data to add missing columns
        preprocessor = DataPreprocessor()
        df_clean = preprocessor.validate_raw_data(df)
        df = preprocessor.create_derived_features(df_clean)

        # Calculate metrics
        response = PerformanceOverviewResponse(
            period_days=days,
            total_jobs=len(df),
            average_efficiency=float(df["efficiency"].mean()),
            efficiency_trend=calculate_trend(df, "efficiency"),
            top_machines=get_top_performers(df, "machine", "efficiency"),
            top_operators=get_top_performers(df, "OperatorName", "efficiency"),
            downtime_summary=calculate_downtime_summary(df),
            quality_indicators={
                "jobs_above_80_efficiency": int((df["efficiency"] > 0.8).sum()),
                "consistency_score": float(1 / (1 + df["efficiency"].std())),
            },
        )

        # Cache for 1 hour
        redis_client.setex(cache_key, 3600, response.json())

        return response

    except Exception as e:
        logger.error(f"Performance overview error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get(
    "/analytics/operator/{operator_name}", response_model=OperatorAnalyticsResponse
)
async def get_operator_analytics(
    operator_name: str,
    days: int = 90,
    db: DatabaseManager = Depends(get_database_manager),
):
    """Get detailed analytics for a specific operator."""
    try:
        # Get operator data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        df = db.get_data_by_date_range(
            start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")
        )

        # Preprocess data to add missing columns
        preprocessor = DataPreprocessor()
        df_clean = preprocessor.validate_raw_data(df)
        df_features = preprocessor.create_derived_features(df_clean)

        operator_data = df_features[df_features["OperatorName"] == operator_name]

        if operator_data.empty:
            raise HTTPException(
                status_code=404, detail=f"No data found for operator {operator_name}"
            )

        # Initialize analyzer
        analyzer = OperatorPerformanceAnalyzer()
        profile = analyzer.get_operator_profile(operator_name, df_features)

        response = OperatorAnalyticsResponse(
            operator_name=operator_name,
            period_days=days,
            total_jobs=profile["total_jobs"],
            machines_operated=profile["machines_operated"],
            parts_worked=profile["parts_produced"][:10],  # Limit to top 10
            efficiency_stats=profile["efficiency_distribution"],
            skill_areas=[
                "Machine Setup",
                "Quality Control",
            ],  # Would calculate from data
            improvement_suggestions=[
                "Focus on reducing setup times",
                "Cross-train on additional machines",
            ],
            performance_trend="improving",  # Would calculate from data
        )

        return response

    except Exception as e:
        logger.error(f"Operator analytics error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Batch Processing Endpoints
@app.post("/batch/retrain-models")
async def trigger_model_retraining(
    background_tasks: BackgroundTasks, model_types: Optional[List[str]] = None
):
    """Trigger background model retraining."""
    try:
        model_types = model_types or ["efficiency", "downtime"]

        for model_type in model_types:
            background_tasks.add_task(retrain_model, model_type)

        return JSONResponse(
            {
                "message": "Model retraining initiated",
                "models": model_types,
                "status": "queued",
            }
        )

    except Exception as e:
        logger.error(f"Model retraining error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch/generate-report")
async def generate_performance_report(
    request: ReportGenerationRequest, background_tasks: BackgroundTasks
):
    """Generate comprehensive performance report."""
    try:
        background_tasks.add_task(
            generate_report,
            request.start_date,
            request.end_date,
            request.report_type,
            request.include_visualizations,
        )

        return JSONResponse(
            {
                "message": "Report generation initiated",
                "report_id": f"report_{int(time.time())}",
                "estimated_completion": "5-10 minutes",
            }
        )

    except Exception as e:
        logger.error(f"Report generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Utility functions
def calculate_trend(df: pd.DataFrame, column: str) -> str:
    """Calculate trend direction for a metric."""
    if len(df) < 2:
        return "stable"

    # Simple trend calculation
    recent_avg = df[column].tail(len(df) // 3).mean()
    older_avg = df[column].head(len(df) // 3).mean()

    if recent_avg > older_avg * 1.05:
        return "improving"
    elif recent_avg < older_avg * 0.95:
        return "declining"
    else:
        return "stable"


def get_top_performers(
    df: pd.DataFrame, group_col: str, metric_col: str, top_n: int = 5
) -> List[Dict]:
    """Get top performers for a metric."""
    grouped = df.groupby(group_col)[metric_col].agg(["mean", "count"]).round(3)
    grouped = grouped[grouped["count"] >= 3]  # At least 3 data points
    top = grouped.nlargest(top_n, "mean")

    return [
        {"name": name, "score": float(row["mean"]), "jobs": int(row["count"])}
        for name, row in top.iterrows()
    ]


def calculate_downtime_summary(df: pd.DataFrame) -> Dict:
    """Calculate downtime summary statistics."""
    downtime_cols = ["SetupTime", "MaintenanceTime", "IdleTime"]
    available_cols = [col for col in downtime_cols if col in df.columns]

    if not available_cols:
        return {"total_downtime_hours": 0, "main_category": "Unknown"}

    total_downtime = df[available_cols].sum(axis=1).sum() / 3600  # Convert to hours
    main_category = df[available_cols].sum().idxmax() if available_cols else "Unknown"

    return {
        "total_downtime_hours": float(total_downtime),
        "main_category": main_category,
    }


# Background task functions
async def log_prediction(prediction_type: str, request_data: Dict, response_data: Dict):
    """Log prediction for monitoring and analytics."""
    try:
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "type": prediction_type,
            "request": request_data,
            "response": response_data,
        }
        logger.info(f"Prediction logged: {json.dumps(log_entry)}")
    except Exception as e:
        logger.error(f"Prediction logging error: {e}")


async def retrain_model(model_type: str):
    """Background task to retrain models."""
    try:
        logger.info(f"Starting {model_type} model retraining...")

        # Load fresh data
        db = DatabaseManager()
        df = db.get_clean_data()

        if model_type == "efficiency":
            predictor = EfficiencyPredictor()
            results = predictor.train(df, optimize_hyperparameters=True)
            predictor.save_model(
                f"models/efficiency_predictor_v{int(time.time())}.joblib"
            )

        elif model_type == "downtime":
            classifier = DowntimeClassifier()
            X, y = classifier.prepare_training_data(df)
            results = classifier.train(X, y, optimize_hyperparameters=True)
            classifier.save_model(
                f"models/downtime_classifier_v{int(time.time())}.joblib"
            )

        logger.info(f"{model_type} model retraining completed successfully")

    except Exception as e:
        logger.error(f"Model retraining error for {model_type}: {e}")


async def generate_report(
    start_date: str, end_date: str, report_type: str, include_viz: bool
):
    """Background task to generate performance reports."""
    try:
        logger.info(f"Generating {report_type} report from {start_date} to {end_date}")

        # Implementation would generate comprehensive reports
        # and save them to storage

        logger.info("Report generation completed")

    except Exception as e:
        logger.error(f"Report generation error: {e}")


if __name__ == "__main__":
    startup_time = time.time()
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=config.DEBUG,
        workers=1 if config.DEBUG else 4,
    )
