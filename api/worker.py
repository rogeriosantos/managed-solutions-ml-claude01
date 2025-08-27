"""
Celery worker for background tasks and batch processing.
"""
import os
import asyncio
import pickle
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from celery import Celery, Task
from celery.schedules import crontab
import pandas as pd
import logging

from ..config import config
from .dependencies import (
    get_database_manager,
    get_efficiency_model, 
    get_downtime_model,
    get_recommendation_engine,
    get_redis_client,
    refresh_models_cache,
    cleanup_old_cache_entries
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create Celery app
celery_app = Celery(
    "cnc_ml_worker",
    broker=os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/1"),
    backend=os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/2"),
    include=["api.worker"]
)

# Celery configuration
celery_app.conf.update(
    task_serializer="pickle",
    accept_content=["pickle"],
    result_serializer="pickle",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=30 * 60,  # 30 minutes
    task_soft_time_limit=25 * 60,  # 25 minutes
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    worker_max_tasks_per_child=1000,
)

# Periodic tasks schedule
celery_app.conf.beat_schedule = {
    "refresh-models-cache": {
        "task": "api.worker.refresh_models_task",
        "schedule": crontab(minute=0, hour="*/2"),  # Every 2 hours
    },
    "cleanup-old-cache": {
        "task": "api.worker.cleanup_cache_task",
        "schedule": crontab(minute=30, hour="*/6"),  # Every 6 hours at :30
    },
    "retrain-models": {
        "task": "api.worker.retrain_models_task",
        "schedule": crontab(minute=0, hour=2, day_of_week=1),  # Weekly on Monday at 2 AM
    },
    "health-check": {
        "task": "api.worker.health_check_task",
        "schedule": crontab(minute="*/15"),  # Every 15 minutes
    },
    "generate-daily-report": {
        "task": "api.worker.generate_daily_report_task",
        "schedule": crontab(minute=0, hour=6),  # Daily at 6 AM
    },
}


class CallbackTask(Task):
    """Custom task class with error handling and logging."""
    
    def on_success(self, retval, task_id, args, kwargs):
        logger.info(f"Task {self.name} [{task_id}] succeeded")
    
    def on_failure(self, exc, task_id, args, kwargs, einfo):
        logger.error(f"Task {self.name} [{task_id}] failed: {exc}")
    
    def on_retry(self, exc, task_id, args, kwargs, einfo):
        logger.warning(f"Task {self.name} [{task_id}] retry: {exc}")


@celery_app.task(bind=True, base=CallbackTask, name="api.worker.batch_predictions_task")
def batch_predictions_task(self, jobs_data: List[Dict[str, Any]], prediction_type: str = "both") -> Dict[str, Any]:
    """Process batch predictions for multiple jobs."""
    try:
        start_time = datetime.utcnow()
        results = []
        
        # Load models
        efficiency_model = get_efficiency_model() if prediction_type in ["efficiency", "both"] else None
        downtime_model = get_downtime_model() if prediction_type in ["downtime", "both"] else None
        
        for i, job_data in enumerate(jobs_data):
            job_start = datetime.utcnow()
            job_result = {
                "job_index": i,
                "efficiency_prediction": None,
                "downtime_prediction": None,
                "processing_time_ms": 0,
                "error": None
            }
            
            try:
                # Convert to DataFrame for model input
                df = pd.DataFrame([job_data])
                
                # Efficiency prediction
                if efficiency_model and prediction_type in ["efficiency", "both"]:
                    eff_result = efficiency_model.predict(df)
                    if not eff_result.empty:
                        job_result["efficiency_prediction"] = {
                            "predicted_efficiency": float(eff_result.iloc[0]["predicted_efficiency"]),
                            "confidence_score": float(eff_result.iloc[0].get("confidence", 0.8)),
                            "expected_completion_time": float(job_data.get("planned_cycle_time", 0) / eff_result.iloc[0]["predicted_efficiency"]),
                            "risk_factors": [],
                            "recommendations": []
                        }
                
                # Downtime prediction
                if downtime_model and prediction_type in ["downtime", "both"]:
                    down_result = downtime_model.predict(df)
                    if not down_result.empty:
                        downtime_prob = float(down_result.iloc[0]["predicted_downtime_probability"])
                        risk_level = "HIGH" if downtime_prob > 0.7 else "MEDIUM" if downtime_prob > 0.3 else "LOW"
                        
                        job_result["downtime_prediction"] = {
                            "downtime_probability": downtime_prob,
                            "risk_level": risk_level,
                            "predicted_downtime_duration": float(down_result.iloc[0].get("predicted_duration", 0)),
                            "maintenance_recommended": downtime_prob > 0.5,
                            "risk_factors": [],
                            "prevention_actions": []
                        }
                
                job_result["processing_time_ms"] = (datetime.utcnow() - job_start).total_seconds() * 1000
                
            except Exception as e:
                job_result["error"] = str(e)
                job_result["processing_time_ms"] = (datetime.utcnow() - job_start).total_seconds() * 1000
            
            results.append(job_result)
            
            # Update task progress
            self.update_state(
                state="PROGRESS",
                meta={"current": i + 1, "total": len(jobs_data), "status": f"Processing job {i + 1}"}
            )
        
        # Calculate summary
        successful = len([r for r in results if r["error"] is None])
        total_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        summary = {
            "total_jobs": len(jobs_data),
            "successful": successful,
            "failed": len(jobs_data) - successful,
            "success_rate": successful / len(jobs_data) if jobs_data else 0,
            "total_processing_time_ms": total_time,
            "avg_processing_time_ms": total_time / len(jobs_data) if jobs_data else 0
        }
        
        if prediction_type in ["efficiency", "both"]:
            eff_predictions = [r["efficiency_prediction"]["predicted_efficiency"] 
                             for r in results 
                             if r["efficiency_prediction"] and r["error"] is None]
            if eff_predictions:
                summary["avg_efficiency"] = sum(eff_predictions) / len(eff_predictions)
        
        if prediction_type in ["downtime", "both"]:
            high_risk_count = len([r for r in results 
                                 if r["downtime_prediction"] and 
                                 r["downtime_prediction"]["risk_level"] == "HIGH" and 
                                 r["error"] is None])
            summary["high_downtime_risk"] = high_risk_count
        
        return {
            "job_results": results,
            "summary": summary,
            "total_processing_time_ms": total_time,
            "success_rate": summary["success_rate"]
        }
        
    except Exception as e:
        logger.error(f"Batch predictions task failed: {e}")
        raise


@celery_app.task(bind=True, base=CallbackTask, name="api.worker.refresh_models_task")
def refresh_models_task(self):
    """Refresh models with latest data."""
    try:
        logger.info("Starting models refresh task")
        
        # Use asyncio for async operations
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(refresh_models_cache())
        loop.close()
        
        logger.info("Models refresh task completed successfully")
        return {"status": "success", "timestamp": datetime.utcnow().isoformat()}
        
    except Exception as e:
        logger.error(f"Models refresh task failed: {e}")
        raise


@celery_app.task(bind=True, base=CallbackTask, name="api.worker.cleanup_cache_task")
def cleanup_cache_task(self):
    """Clean up old cache entries."""
    try:
        logger.info("Starting cache cleanup task")
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(cleanup_old_cache_entries())
        loop.close()
        
        logger.info("Cache cleanup task completed successfully")
        return {"status": "success", "timestamp": datetime.utcnow().isoformat()}
        
    except Exception as e:
        logger.error(f"Cache cleanup task failed: {e}")
        raise


@celery_app.task(bind=True, base=CallbackTask, name="api.worker.retrain_models_task")
def retrain_models_task(self):
    """Retrain models with latest data."""
    try:
        logger.info("Starting model retraining task")
        
        # Get latest data
        db_manager = get_database_manager()
        df = db_manager.get_data(limit=50000)  # Get more data for training
        
        if df.empty or len(df) < config.MODEL_MIN_DATA_POINTS:
            logger.warning(f"Insufficient data for retraining: {len(df)} rows")
            return {"status": "skipped", "reason": "insufficient_data", "rows": len(df)}
        
        # Retrain efficiency model
        efficiency_model = get_efficiency_model()
        eff_results = efficiency_model.train(df, optimize_hyperparameters=True)
        
        # Save efficiency model
        eff_model_path = os.path.join(config.MODEL_DIR, "efficiency_model.pkl")
        os.makedirs(os.path.dirname(eff_model_path), exist_ok=True)
        with open(eff_model_path, 'wb') as f:
            pickle.dump(efficiency_model, f)
        
        # Retrain downtime model
        downtime_model = get_downtime_model()
        down_results = downtime_model.train(df, optimize_hyperparameters=True)
        
        # Save downtime model
        down_model_path = os.path.join(config.MODEL_DIR, "downtime_model.pkl")
        with open(down_model_path, 'wb') as f:
            pickle.dump(downtime_model, f)
        
        # Update recommendation engine
        recommendation_engine = get_recommendation_engine()
        recommendation_engine.fit(df)
        
        result = {
            "status": "success",
            "timestamp": datetime.utcnow().isoformat(),
            "data_rows": len(df),
            "efficiency_model": eff_results,
            "downtime_model": down_results
        }
        
        logger.info(f"Model retraining completed successfully: {result}")
        return result
        
    except Exception as e:
        logger.error(f"Model retraining task failed: {e}")
        raise


@celery_app.task(bind=True, base=CallbackTask, name="api.worker.health_check_task")
def health_check_task(self):
    """Periodic health check task."""
    try:
        health_status = {"timestamp": datetime.utcnow().isoformat()}
        
        # Check database
        try:
            db_manager = get_database_manager()
            df = db_manager.get_data(limit=1)
            health_status["database"] = "healthy"
        except Exception as e:
            health_status["database"] = f"error: {str(e)}"
        
        # Check Redis
        try:
            redis_client = get_redis_client()
            redis_client.ping()
            health_status["redis"] = "healthy"
        except Exception as e:
            health_status["redis"] = f"error: {str(e)}"
        
        # Check models
        try:
            get_efficiency_model()
            get_downtime_model()
            health_status["models"] = "loaded"
        except Exception as e:
            health_status["models"] = f"error: {str(e)}"
        
        # Overall status
        errors = [k for k, v in health_status.items() if isinstance(v, str) and v.startswith("error")]
        health_status["overall"] = "unhealthy" if errors else "healthy"
        
        if errors:
            logger.warning(f"Health check found issues: {errors}")
        
        return health_status
        
    except Exception as e:
        logger.error(f"Health check task failed: {e}")
        raise


@celery_app.task(bind=True, base=CallbackTask, name="api.worker.generate_daily_report_task")
def generate_daily_report_task(self):
    """Generate daily performance report."""
    try:
        logger.info("Starting daily report generation")
        
        # Get yesterday's data
        end_date = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        start_date = end_date - timedelta(days=1)
        
        db_manager = get_database_manager()
        df = db_manager.get_data_by_date_range(start_date, end_date)
        
        if df.empty:
            logger.warning("No data available for daily report")
            return {"status": "no_data", "date": start_date.isoformat()}
        
        # Calculate key metrics
        total_jobs = len(df)
        avg_efficiency = df["efficiency"].mean() if "efficiency" in df.columns else 0
        total_downtime = df["downtime_duration"].sum() if "downtime_duration" in df.columns else 0
        
        # Top performers
        if "operator" in df.columns and "efficiency" in df.columns:
            top_operators = df.groupby("operator")["efficiency"].mean().sort_values(ascending=False).head(5)
            top_operators_list = [{"operator": op, "efficiency": eff} for op, eff in top_operators.items()]
        else:
            top_operators_list = []
        
        # Machine performance
        if "machine" in df.columns and "efficiency" in df.columns:
            machine_performance = df.groupby("machine")["efficiency"].mean().sort_values(ascending=False)
            machine_performance_list = [{"machine": machine, "efficiency": eff} 
                                      for machine, eff in machine_performance.items()]
        else:
            machine_performance_list = []
        
        report = {
            "date": start_date.isoformat(),
            "total_jobs": total_jobs,
            "average_efficiency": round(avg_efficiency, 3),
            "total_downtime_minutes": round(total_downtime, 2),
            "top_operators": top_operators_list,
            "machine_performance": machine_performance_list,
            "generated_at": datetime.utcnow().isoformat()
        }
        
        # Store report in Redis for quick access
        try:
            redis_client = get_redis_client()
            report_key = f"daily_report:{start_date.strftime('%Y-%m-%d')}"
            redis_client.setex(report_key, 86400 * 7, pickle.dumps(report).decode('latin1'))  # Keep for 7 days
        except Exception as e:
            logger.error(f"Failed to store daily report in Redis: {e}")
        
        logger.info(f"Daily report generated successfully for {start_date.strftime('%Y-%m-%d')}")
        return report
        
    except Exception as e:
        logger.error(f"Daily report generation failed: {e}")
        raise


@celery_app.task(bind=True, base=CallbackTask, name="api.worker.export_data_task")
def export_data_task(self, filters: Dict[str, Any], format: str = "csv") -> Dict[str, Any]:
    """Export data based on filters."""
    try:
        logger.info(f"Starting data export task with format: {format}")
        
        # Get data based on filters
        db_manager = get_database_manager()
        
        if "start_date" in filters and "end_date" in filters:
            start_date = datetime.fromisoformat(filters["start_date"])
            end_date = datetime.fromisoformat(filters["end_date"])
            df = db_manager.get_data_by_date_range(start_date, end_date)
        else:
            df = db_manager.get_data(limit=filters.get("limit", 10000))
        
        # Apply additional filters
        for column, value in filters.items():
            if column in df.columns and column not in ["start_date", "end_date", "limit"]:
                if isinstance(value, list):
                    df = df[df[column].isin(value)]
                else:
                    df = df[df[column] == value]
        
        if df.empty:
            return {"status": "no_data", "message": "No data matches the specified filters"}
        
        # Generate filename
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"cnc_export_{timestamp}.{format}"
        filepath = os.path.join(config.DATA_DIR, "exports", filename)
        
        # Create export directory
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Export data
        if format == "csv":
            df.to_csv(filepath, index=False)
        elif format == "parquet":
            df.to_parquet(filepath, index=False)
        elif format == "json":
            df.to_json(filepath, orient="records", indent=2)
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        file_size = os.path.getsize(filepath)
        
        result = {
            "status": "success",
            "filename": filename,
            "filepath": filepath,
            "format": format,
            "rows_exported": len(df),
            "file_size_bytes": file_size,
            "generated_at": datetime.utcnow().isoformat()
        }
        
        logger.info(f"Data export completed: {result}")
        return result
        
    except Exception as e:
        logger.error(f"Data export task failed: {e}")
        raise


# Task status checking utilities
def get_task_status(task_id: str) -> Dict[str, Any]:
    """Get status of a Celery task."""
    try:
        task = celery_app.AsyncResult(task_id)
        
        if task.state == "PENDING":
            response = {"status": "PENDING", "message": "Task is waiting to be processed"}
        elif task.state == "PROGRESS":
            response = {
                "status": "PROGRESS",
                "current": task.info.get("current", 0),
                "total": task.info.get("total", 1),
                "message": task.info.get("status", "Processing...")
            }
        elif task.state == "SUCCESS":
            response = {
                "status": "SUCCESS",
                "result": task.result,
                "message": "Task completed successfully"
            }
        else:  # FAILURE
            response = {
                "status": "FAILURE",
                "error": str(task.info),
                "message": "Task failed"
            }
        
        return response
        
    except Exception as e:
        return {"status": "ERROR", "error": str(e), "message": "Failed to get task status"}


if __name__ == "__main__":
    celery_app.start()