"""Configuration settings for CNC ML Project."""

import os
from typing import Dict, Any
from dotenv import load_dotenv

load_dotenv()


class Config:
    """Application configuration class."""

    # Environment Configuration
    DEBUG = os.getenv("DEBUG", "True").lower() == "true"
    ENVIRONMENT = os.getenv("ENVIRONMENT", "development")

    # Model Storage Configuration
    MODEL_DIR = os.getenv("MODEL_DIR", "models")

    # Redis Configuration
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

    # Database Configuration
    DB_HOST = os.getenv("DB_HOST", "localhost")
    DB_PORT = int(os.getenv("DB_PORT", "3306"))
    DB_NAME = os.getenv("DB_NAME", "cnc_database")
    DB_USER = os.getenv("DB_USER", "root")
    DB_PASSWORD = os.getenv("DB_PASSWORD", "")

    # Connection Pool Settings
    DB_POOL_SIZE = int(os.getenv("DB_POOL_SIZE", "10"))
    DB_MAX_OVERFLOW = int(os.getenv("DB_MAX_OVERFLOW", "20"))
    DB_POOL_TIMEOUT = int(os.getenv("DB_POOL_TIMEOUT", "30"))

    # Model Configuration
    MODEL_RANDOM_STATE = int(os.getenv("MODEL_RANDOM_STATE", "42"))
    TEST_SIZE = float(os.getenv("TEST_SIZE", "0.2"))
    VALIDATION_SIZE = float(os.getenv("VALIDATION_SIZE", "0.1"))

    # Feature Engineering
    EFFICIENCY_THRESHOLD = float(os.getenv("EFFICIENCY_THRESHOLD", "0.8"))
    MIN_JOB_DURATION = int(os.getenv("MIN_JOB_DURATION", "60"))  # seconds
    MAX_JOB_DURATION = int(os.getenv("MAX_JOB_DURATION", "86400"))  # 24 hours

    # Performance Thresholds
    HIGH_PERFORMANCE_THRESHOLD = float(os.getenv("HIGH_PERFORMANCE_THRESHOLD", "0.85"))
    LOW_PERFORMANCE_THRESHOLD = float(os.getenv("LOW_PERFORMANCE_THRESHOLD", "0.60"))

    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT = os.getenv(
        "LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Cache Settings
    CACHE_TTL = int(os.getenv("CACHE_TTL", "3600"))  # 1 hour

    @classmethod
    def get_database_url(cls) -> str:
        """Generate database connection URL."""
        return f"mysql+mysqlconnector://{cls.DB_USER}:{cls.DB_PASSWORD}@{cls.DB_HOST}:{cls.DB_PORT}/{cls.DB_NAME}"

    @classmethod
    def get_model_config(cls) -> Dict[str, Any]:
        """Get model configuration dictionary."""
        return {
            "random_state": cls.MODEL_RANDOM_STATE,
            "test_size": cls.TEST_SIZE,
            "validation_size": cls.VALIDATION_SIZE,
        }

    @classmethod
    def get_feature_config(cls) -> Dict[str, Any]:
        """Get feature engineering configuration."""
        return {
            "efficiency_threshold": cls.EFFICIENCY_THRESHOLD,
            "min_job_duration": cls.MIN_JOB_DURATION,
            "max_job_duration": cls.MAX_JOB_DURATION,
        }


class DevelopmentConfig(Config):
    """Development environment configuration."""

    DEBUG = True
    LOG_LEVEL = "DEBUG"


class ProductionConfig(Config):
    """Production environment configuration."""

    DEBUG = False
    LOG_LEVEL = "WARNING"


config = Config()
