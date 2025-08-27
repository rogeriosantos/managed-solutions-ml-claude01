"""
Pydantic models for API request/response schemas.
"""

from datetime import datetime
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field, validator
from enum import Enum


class JobPriority(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class MachineType(str, Enum):
    MILL = "mill"
    LATHE = "lathe"
    DRILL = "drill"
    GRINDER = "grinder"
    OTHER = "other"


# Base models
class BaseJobData(BaseModel):
    machine: str = Field(..., description="Machine identifier")
    part: str = Field(..., description="Part number")
    operator: str = Field(..., description="Operator ID")
    planned_cycle_time: float = Field(
        ..., gt=0, description="Planned cycle time in minutes"
    )
    actual_cycle_time: Optional[float] = Field(
        None, gt=0, description="Actual cycle time in minutes"
    )
    quantity: int = Field(..., gt=0, description="Quantity of parts")

    @validator("machine")
    def machine_must_not_be_empty(cls, v):
        if not v.strip():
            raise ValueError("Machine must not be empty")
        return v.strip().upper()

    @validator("part")
    def part_must_not_be_empty(cls, v):
        if not v.strip():
            raise ValueError("Part must not be empty")
        return v.strip().upper()


# Efficiency Prediction
class EfficiencyPredictionRequest(BaseJobData):
    job_priority: JobPriority = JobPriority.NORMAL
    machine_type: Optional[MachineType] = None
    setup_time: Optional[float] = Field(None, ge=0, description="Setup time in minutes")
    material_type: Optional[str] = Field(None, description="Material type")
    complexity_score: Optional[float] = Field(
        None, ge=1, le=10, description="Job complexity (1-10)"
    )
    shift: Optional[str] = Field(None, description="Work shift")

    class Config:
        json_schema_extra = {
            "example": {
                "machine": "MILL_001",
                "part": "PART_12345",
                "operator": "OP_001",
                "planned_cycle_time": 45.5,
                "quantity": 100,
                "job_priority": "normal",
                "machine_type": "mill",
                "setup_time": 15.0,
                "material_type": "ALUMINUM",
                "complexity_score": 6.5,
                "shift": "DAY",
            }
        }


class EfficiencyPredictionResponse(BaseModel):
    predicted_efficiency: float = Field(
        ..., ge=0, le=2, description="Predicted efficiency ratio"
    )
    confidence_score: float = Field(
        ..., ge=0, le=1, description="Prediction confidence"
    )
    expected_completion_time: float = Field(
        ..., gt=0, description="Expected completion time in minutes"
    )
    risk_factors: List[str] = Field(
        default_factory=list, description="Identified risk factors"
    )
    recommendations: List[str] = Field(
        default_factory=list, description="Performance recommendations"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "predicted_efficiency": 0.85,
                "confidence_score": 0.92,
                "expected_completion_time": 53.5,
                "risk_factors": ["High complexity job", "New operator on this machine"],
                "recommendations": [
                    "Consider additional setup time",
                    "Monitor first few cycles closely",
                ],
            }
        }


# Downtime Prediction
class DowntimePredictionRequest(BaseJobData):
    machine_age_months: Optional[int] = Field(
        None, ge=0, description="Machine age in months"
    )
    last_maintenance_days: Optional[int] = Field(
        None, ge=0, description="Days since last maintenance"
    )
    avg_daily_runtime: Optional[float] = Field(
        None, ge=0, description="Average daily runtime hours"
    )
    recent_error_count: Optional[int] = Field(
        None, ge=0, description="Recent error count"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "machine": "LATHE_002",
                "part": "PART_67890",
                "operator": "OP_002",
                "planned_cycle_time": 30.0,
                "quantity": 50,
                "machine_age_months": 36,
                "last_maintenance_days": 14,
                "avg_daily_runtime": 16.5,
                "recent_error_count": 2,
            }
        }


class DowntimePredictionResponse(BaseModel):
    downtime_probability: float = Field(
        ..., ge=0, le=1, description="Probability of downtime"
    )
    risk_level: str = Field(..., description="Risk level: LOW, MEDIUM, HIGH")
    predicted_downtime_duration: Optional[float] = Field(
        None, description="Expected downtime duration in minutes"
    )
    maintenance_recommended: bool = Field(
        ..., description="Whether maintenance is recommended"
    )
    risk_factors: List[str] = Field(default_factory=list)
    prevention_actions: List[str] = Field(default_factory=list)

    class Config:
        json_schema_extra = {
            "example": {
                "downtime_probability": 0.15,
                "risk_level": "MEDIUM",
                "predicted_downtime_duration": 45.0,
                "maintenance_recommended": True,
                "risk_factors": [
                    "Machine due for maintenance",
                    "Recent error activity",
                ],
                "prevention_actions": [
                    "Schedule preventive maintenance",
                    "Replace worn components",
                ],
            }
        }


# Operator Recommendations
class OperatorRecommendationRequest(BaseModel):
    machine: str = Field(..., description="Machine identifier")
    part: Optional[str] = Field(None, description="Part number")
    job_priority: JobPriority = JobPriority.NORMAL
    required_skills: Optional[List[str]] = Field(
        default_factory=list, description="Required skills"
    )
    shift: Optional[str] = Field(None, description="Work shift")

    class Config:
        json_schema_extra = {
            "example": {
                "machine": "MILL_001",
                "part": "PART_12345",
                "job_priority": "high",
                "required_skills": ["CNC_PROGRAMMING", "PRECISION_MEASUREMENT"],
                "shift": "DAY",
            }
        }


class OperatorRecommendation(BaseModel):
    operator_id: str = Field(..., description="Operator identifier")
    efficiency_score: float = Field(
        ..., ge=0, le=1, description="Operator efficiency on this machine/part"
    )
    experience_level: str = Field(..., description="Experience level")
    availability_score: float = Field(
        ..., ge=0, le=1, description="Operator availability"
    )
    skill_match_score: float = Field(..., ge=0, le=1, description="Skills match score")
    overall_score: float = Field(
        ..., ge=0, le=1, description="Overall recommendation score"
    )
    rationale: List[str] = Field(
        default_factory=list, description="Reasoning for recommendation"
    )


class OperatorRecommendationResponse(BaseModel):
    recommendations: List[OperatorRecommendation] = Field(
        ..., description="Ranked operator recommendations"
    )
    total_available: int = Field(..., description="Total available operators")

    class Config:
        json_schema_extra = {
            "example": {
                "recommendations": [
                    {
                        "operator_id": "OP_001",
                        "efficiency_score": 0.92,
                        "experience_level": "EXPERT",
                        "availability_score": 0.85,
                        "skill_match_score": 0.95,
                        "overall_score": 0.91,
                        "rationale": [
                            "High efficiency on this machine",
                            "Perfect skill match",
                            "Available for assignment",
                        ],
                    }
                ],
                "total_available": 8,
            }
        }


# Performance Analytics
class PerformanceAnalyticsRequest(BaseModel):
    start_date: datetime = Field(..., description="Analysis start date")
    end_date: datetime = Field(..., description="Analysis end date")
    machines: Optional[List[str]] = Field(
        None, description="Specific machines to analyze"
    )
    operators: Optional[List[str]] = Field(
        None, description="Specific operators to analyze"
    )
    parts: Optional[List[str]] = Field(None, description="Specific parts to analyze")
    aggregation_level: str = Field(
        "daily",
        pattern="^(hourly|daily|weekly|monthly)$",
        description="Data aggregation level",
    )

    @validator("end_date")
    def end_date_must_be_after_start_date(cls, v, values):
        if "start_date" in values and v <= values["start_date"]:
            raise ValueError("end_date must be after start_date")
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "start_date": "2024-01-01T00:00:00",
                "end_date": "2024-01-31T23:59:59",
                "machines": ["MILL_001", "LATHE_002"],
                "aggregation_level": "daily",
            }
        }


class PerformanceMetrics(BaseModel):
    period: str = Field(..., description="Time period")
    efficiency: float = Field(..., description="Average efficiency")
    downtime_minutes: float = Field(..., description="Total downtime in minutes")
    jobs_completed: int = Field(..., description="Number of jobs completed")
    quality_score: Optional[float] = Field(
        None, description="Quality score if available"
    )


class PerformanceAnalyticsResponse(BaseModel):
    summary: Dict[str, Any] = Field(..., description="Overall performance summary")
    machine_performance: Dict[str, List[PerformanceMetrics]] = Field(
        default_factory=dict
    )
    operator_performance: Dict[str, List[PerformanceMetrics]] = Field(
        default_factory=dict
    )
    trends: Dict[str, Any] = Field(
        default_factory=dict, description="Performance trends"
    )
    insights: List[str] = Field(default_factory=list, description="Key insights")

    class Config:
        json_schema_extra = {
            "example": {
                "summary": {
                    "total_efficiency": 0.84,
                    "total_downtime_hours": 24.5,
                    "total_jobs": 1250,
                    "top_performer": "OP_001",
                },
                "insights": [
                    "Machine MILL_001 shows 15% efficiency improvement",
                    "Operator OP_001 consistently outperforms on complex jobs",
                ],
            }
        }


# Batch Processing
class BatchPredictionRequest(BaseModel):
    jobs: List[Dict[str, Any]] = Field(
        ..., min_items=1, max_items=1000, description="List of jobs to process"
    )
    prediction_type: str = Field(
        ...,
        pattern="^(efficiency|downtime|both)$",
        description="Type of predictions to generate",
    )
    include_recommendations: bool = Field(
        True, description="Include operator recommendations"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "jobs": [
                    {
                        "machine": "MILL_001",
                        "part": "PART_12345",
                        "operator": "OP_001",
                        "planned_cycle_time": 45.5,
                        "quantity": 100,
                    }
                ],
                "prediction_type": "both",
                "include_recommendations": True,
            }
        }


class BatchJobResult(BaseModel):
    job_index: int = Field(..., description="Index of job in the batch")
    efficiency_prediction: Optional[EfficiencyPredictionResponse] = None
    downtime_prediction: Optional[DowntimePredictionResponse] = None
    operator_recommendations: Optional[OperatorRecommendationResponse] = None
    processing_time_ms: float = Field(..., description="Processing time for this job")
    error: Optional[str] = Field(None, description="Error message if processing failed")


class BatchPredictionResponse(BaseModel):
    job_results: List[BatchJobResult] = Field(..., description="Results for each job")
    summary: Dict[str, Any] = Field(..., description="Batch processing summary")
    total_processing_time_ms: float = Field(..., description="Total processing time")
    success_rate: float = Field(
        ..., ge=0, le=1, description="Success rate of batch processing"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "summary": {
                    "total_jobs": 100,
                    "successful": 98,
                    "failed": 2,
                    "avg_efficiency": 0.86,
                    "high_downtime_risk": 5,
                },
                "total_processing_time_ms": 2450.5,
                "success_rate": 0.98,
            }
        }


# Health Check
class HealthCheckResponse(BaseModel):
    status: str = Field(..., description="Overall health status")
    timestamp: datetime = Field(..., description="Health check timestamp")
    version: str = Field(..., description="Application version")
    database: Dict[str, Any] = Field(..., description="Database connection status")
    redis: Dict[str, Any] = Field(..., description="Redis connection status")
    models: Dict[str, Any] = Field(..., description="ML models status")
    uptime_seconds: float = Field(..., description="Application uptime in seconds")

    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "timestamp": "2024-01-15T10:30:00",
                "version": "1.0.0",
                "database": {"status": "connected", "pool_size": 10},
                "redis": {"status": "connected", "used_memory": "2.1MB"},
                "models": {"efficiency_model": "loaded", "downtime_model": "loaded"},
                "uptime_seconds": 86400,
            }
        }


# Error responses
class ErrorResponse(BaseModel):
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="Error timestamp"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "error": "Validation Error",
                "detail": "planned_cycle_time must be greater than 0",
                "timestamp": "2024-01-15T10:30:00",
            }
        }


# Simple Health Check Models
class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
    version: str


class DetailedHealthResponse(BaseModel):
    status: str
    timestamp: datetime
    services: Dict[str, bool]
    uptime: float


# Downtime Models
class DowntimeCategory(BaseModel):
    category: str
    probability: float
    estimated_duration: int


# Performance Overview Model
class PerformanceOverviewResponse(BaseModel):
    period_days: int
    total_jobs: int
    average_efficiency: float
    efficiency_trend: str
    top_machines: List[Dict[str, Any]]
    top_operators: List[Dict[str, Any]]
    downtime_summary: Dict[str, Any]
    quality_indicators: Dict[str, Any]


# Operator Analytics Model
class OperatorAnalyticsResponse(BaseModel):
    operator_name: str
    period_days: int
    total_jobs: int
    machines_operated: List[str]
    parts_worked: List[str]
    efficiency_stats: Dict[str, Any]
    skill_areas: List[str]
    improvement_suggestions: List[str]
    performance_trend: str


# Report Generation Model
class ReportGenerationRequest(BaseModel):
    start_date: str
    end_date: str
    report_type: str
    include_visualizations: bool = True
