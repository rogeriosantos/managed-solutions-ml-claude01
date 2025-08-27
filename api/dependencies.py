"""
FastAPI dependency injection for database, Redis, and model management.
"""

import os
import asyncio
import pickle
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Generator
from functools import lru_cache
import redis
import pandas as pd
from fastapi import Depends, HTTPException, status
from sqlalchemy.orm import Session
import logging

from src.data.database import DatabaseManager
from src.models.efficiency_predictor import EfficiencyPredictor
from src.models.downtime_classifier import DowntimeClassifier
from src.analytics.recommendations import AssignmentRecommendationEngine
from src.utils.config import config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global instances
_db_manager: Optional[DatabaseManager] = None
_redis_client: Optional[redis.Redis] = None
_efficiency_model: Optional[EfficiencyPredictor] = None
_downtime_model: Optional[DowntimeClassifier] = None
_recommendation_engine: Optional[AssignmentRecommendationEngine] = None
_model_cache: Dict[str, Any] = {}
_app_start_time = datetime.utcnow()


@lru_cache()
def get_settings():
    """Get application settings with caching."""
    return config


def get_database_manager() -> DatabaseManager:
    """Get database manager singleton."""
    global _db_manager
    if _db_manager is None:
        try:
            _db_manager = DatabaseManager()
            logger.info("Database manager initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize database manager: {e}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Database connection unavailable",
            )
    return _db_manager


def get_database_session() -> Generator[Session, None, None]:
    """Get database session with proper cleanup."""
    db_manager = get_database_manager()
    session = None
    try:
        session = db_manager.get_session()
        yield session
    except Exception as e:
        if session:
            session.rollback()
        logger.error(f"Database session error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database operation failed",
        )
    finally:
        if session:
            session.close()


def get_redis_client() -> redis.Redis:
    """Get Redis client singleton."""
    global _redis_client
    if _redis_client is None:
        try:
            redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
            _redis_client = redis.from_url(
                redis_url,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True,
                health_check_interval=30,
            )
            # Test connection
            _redis_client.ping()
            logger.info("Redis client initialized successfully")
        except Exception as e:
            logger.warning(
                f"Redis connection failed: {e}. Using mock Redis client for development."
            )
            # Create a mock Redis client for development
            _redis_client = MockRedisClient()
    return _redis_client


class MockRedisClient:
    """Mock Redis client for development when Redis server is not available."""

    def __init__(self):
        self._data = {}

    def ping(self):
        return True

    def get(self, key):
        return self._data.get(key)

    def set(self, key, value):
        self._data[key] = value
        return True

    def setex(self, key, time, value):
        # For simplicity, ignore expiration in mock
        self._data[key] = value
        return True

    def delete(self, key):
        if key in self._data:
            del self._data[key]
        return True


def load_model_from_cache_or_file(model_name: str, model_class, model_path: str):
    """Load model from memory cache or file with fallback."""
    global _model_cache

    if model_name in _model_cache:
        return _model_cache[model_name]

    try:
        # Try to load from file
        if os.path.exists(model_path):
            with open(model_path, "rb") as f:
                model = pickle.load(f)
                _model_cache[model_name] = model
                logger.info(f"Loaded {model_name} from file: {model_path}")
                return model
        else:
            # Create new model instance and train it with available data
            logger.warning(
                f"Model file not found: {model_path}. Creating and training new instance."
            )
            model = model_class()

            # Auto-train the model with available data
            try:
                db = DatabaseManager()
                df = db.get_all_data(limit=1000)  # Use sample data for quick training

                if len(df) > 50:  # Ensure we have enough data
                    logger.info(f"Training {model_name} with {len(df)} records...")

                    if model_name == "efficiency_model":
                        # Train efficiency predictor
                        results = model.train(df)
                        logger.info(
                            f"Efficiency model trained successfully. R² Score: {results.get('test_r2', 'N/A')}"
                        )
                    elif model_name == "downtime_model":
                        # Train downtime classifier
                        X, y = model.prepare_training_data(df)
                        if len(X) > 10:
                            results = model.train(X, y)
                            logger.info(
                                f"Downtime model trained successfully. Accuracy: {results.get('test_accuracy', 'N/A')}"
                            )

                    # Save the trained model
                    os.makedirs(config.MODEL_DIR, exist_ok=True)
                    with open(model_path, "wb") as f:
                        pickle.dump(model, f)
                    logger.info(f"Saved trained {model_name} to {model_path}")
                else:
                    logger.warning(
                        f"Insufficient data ({len(df)} records) to train {model_name}. Using untrained model."
                    )

            except Exception as train_error:
                logger.error(
                    f"Failed to auto-train {model_name}: {train_error}. Using untrained model."
                )

            _model_cache[model_name] = model
            return model

    except Exception as e:
        logger.error(f"Failed to load {model_name}: {e}")
        # Return new instance as fallback
        model = model_class()
        _model_cache[model_name] = model
        return model


def get_efficiency_model() -> EfficiencyPredictor:
    """Get efficiency prediction model."""
    global _efficiency_model
    if _efficiency_model is None:
        model_path = os.path.join(config.MODEL_DIR, "efficiency_model.pkl")
        _efficiency_model = load_model_from_cache_or_file(
            "efficiency_model", EfficiencyPredictor, model_path
        )
    return _efficiency_model


def get_downtime_model() -> DowntimeClassifier:
    """Get downtime prediction model."""
    global _downtime_model
    if _downtime_model is None:
        model_path = os.path.join(config.MODEL_DIR, "downtime_model.pkl")
        _downtime_model = load_model_from_cache_or_file(
            "downtime_model", DowntimeClassifier, model_path
        )
    return _downtime_model


def get_recommendation_engine() -> AssignmentRecommendationEngine:
    """Get recommendation engine."""
    global _recommendation_engine
    if _recommendation_engine is None:
        try:
            _recommendation_engine = AssignmentRecommendationEngine()
            # Try to initialize with database data
            db_manager = get_database_manager()
            df = db_manager.get_data(
                limit=10000
            )  # Load recent data for recommendations
            if not df.empty:
                _recommendation_engine.fit(df)
                logger.info("Recommendation engine initialized with database data")
            else:
                logger.warning("No data available for recommendation engine")
        except Exception as e:
            logger.error(f"Failed to initialize recommendation engine: {e}")
            _recommendation_engine = AssignmentRecommendationEngine()
    return _recommendation_engine


def get_cached_data(key: str, ttl_seconds: int = 300) -> Optional[Any]:
    """Get data from Redis cache with TTL."""
    try:
        redis_client = get_redis_client()
        cached_data = redis_client.get(key)
        if cached_data:
            return pickle.loads(cached_data.encode("latin1"))
        return None
    except Exception as e:
        logger.error(f"Cache read error for key {key}: {e}")
        return None


def set_cached_data(key: str, data: Any, ttl_seconds: int = 300) -> bool:
    """Set data in Redis cache with TTL."""
    try:
        redis_client = get_redis_client()
        serialized_data = pickle.dumps(data).decode("latin1")
        redis_client.setex(key, ttl_seconds, serialized_data)
        return True
    except Exception as e:
        logger.error(f"Cache write error for key {key}: {e}")
        return False


def get_database_data(
    limit: Optional[int] = None,
    filters: Optional[Dict[str, Any]] = None,
    use_cache: bool = True,
    cache_ttl: int = 600,
) -> pd.DataFrame:
    """Get database data with optional caching."""
    # Create cache key
    cache_key = f"db_data:{limit}:{hash(str(filters))}"

    if use_cache:
        cached_df = get_cached_data(cache_key, cache_ttl)
        if cached_df is not None:
            logger.info(f"Returning cached data for key: {cache_key}")
            return cached_df

    try:
        db_manager = get_database_manager()
        df = db_manager.get_data(limit=limit)

        # Apply filters if provided
        if filters and not df.empty:
            for column, value in filters.items():
                if column in df.columns:
                    if isinstance(value, list):
                        df = df[df[column].isin(value)]
                    else:
                        df = df[df[column] == value]

        # Cache the result
        if use_cache and not df.empty:
            set_cached_data(cache_key, df, cache_ttl)

        return df

    except Exception as e:
        logger.error(f"Failed to get database data: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve data from database",
        )


def check_model_health(model, model_name: str) -> Dict[str, Any]:
    """Check if model is loaded and ready."""
    try:
        if model is None:
            return {"status": "not_loaded", "error": f"{model_name} not initialized"}

        # Basic model validation
        if hasattr(model, "model") and model.model is not None:
            return {"status": "loaded", "type": type(model.model).__name__}
        elif hasattr(model, "models") and model.models:
            return {"status": "loaded", "models_count": len(model.models)}
        else:
            return {"status": "initialized", "note": "Model created but not trained"}

    except Exception as e:
        return {"status": "error", "error": str(e)}


def get_system_health() -> Dict[str, Any]:
    """Get comprehensive system health status."""
    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow(),
        "uptime_seconds": (datetime.utcnow() - _app_start_time).total_seconds(),
    }

    # Database health
    try:
        db_manager = get_database_manager()
        connection_info = db_manager.get_connection_info()
        health_status["database"] = {
            "status": "connected",
            "pool_size": connection_info.get("pool_size", "unknown"),
            "active_connections": connection_info.get("active_connections", "unknown"),
        }
    except Exception as e:
        health_status["database"] = {"status": "error", "error": str(e)}
        health_status["status"] = "degraded"

    # Redis health
    try:
        redis_client = get_redis_client()
        redis_info = redis_client.info()
        health_status["redis"] = {
            "status": "connected",
            "used_memory": redis_info.get("used_memory_human", "unknown"),
            "connected_clients": redis_info.get("connected_clients", "unknown"),
        }
    except Exception as e:
        health_status["redis"] = {"status": "error", "error": str(e)}
        health_status["status"] = "degraded"

    # Models health
    try:
        efficiency_model = get_efficiency_model()
        downtime_model = get_downtime_model()
        recommendation_engine = get_recommendation_engine()

        health_status["models"] = {
            "efficiency_model": check_model_health(
                efficiency_model, "efficiency_model"
            ),
            "downtime_model": check_model_health(downtime_model, "downtime_model"),
            "recommendation_engine": check_model_health(
                recommendation_engine, "recommendation_engine"
            ),
        }

        # Check if any model has issues
        for model_status in health_status["models"].values():
            if model_status["status"] in ["error", "not_loaded"]:
                health_status["status"] = "degraded"

    except Exception as e:
        health_status["models"] = {"status": "error", "error": str(e)}
        health_status["status"] = "unhealthy"

    return health_status


# Dependency functions for FastAPI
def get_db_manager() -> DatabaseManager:
    """FastAPI dependency for database manager."""
    return get_database_manager()


def get_redis() -> redis.Redis:
    """FastAPI dependency for Redis client."""
    return get_redis_client()


def get_efficiency_predictor() -> EfficiencyPredictor:
    """FastAPI dependency for efficiency model."""
    return get_efficiency_model()


def get_downtime_predictor() -> DowntimeClassifier:
    """FastAPI dependency for downtime model."""
    return get_downtime_model()


def get_recommender() -> AssignmentRecommendationEngine:
    """FastAPI dependency for recommendation engine."""
    return get_recommendation_engine()


# Background task utilities
async def refresh_models_cache():
    """Background task to refresh models cache periodically."""
    try:
        logger.info("Refreshing models cache...")

        # Reload data for recommendation engine
        db_manager = get_database_manager()
        df = db_manager.get_data(limit=10000)

        if not df.empty:
            recommendation_engine = get_recommendation_engine()
            recommendation_engine.fit(df)
            logger.info("Recommendation engine refreshed with latest data")

        # Clear old cache entries
        redis_client = get_redis_client()
        keys = redis_client.keys("db_data:*")
        if keys:
            redis_client.delete(*keys)
            logger.info(f"Cleared {len(keys)} cached database entries")

    except Exception as e:
        logger.error(f"Error refreshing models cache: {e}")


async def cleanup_old_cache_entries():
    """Background task to cleanup old cache entries."""
    try:
        redis_client = get_redis_client()

        # Get all keys with TTL info
        keys = redis_client.keys("*")
        expired_keys = []

        for key in keys:
            ttl = redis_client.ttl(key)
            if ttl == -1:  # No expiration set
                # Set default expiration for keys without TTL
                redis_client.expire(key, 3600)  # 1 hour
            elif ttl == -2:  # Key doesn't exist (race condition)
                expired_keys.append(key)

        if expired_keys:
            logger.info(f"Found {len(expired_keys)} expired keys to cleanup")

    except Exception as e:
        logger.error(f"Error during cache cleanup: {e}")
