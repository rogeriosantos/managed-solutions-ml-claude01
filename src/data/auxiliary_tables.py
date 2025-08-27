"""Auxiliary tables creation for performance summaries and analytics."""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import logging

from .database import DatabaseManager
from ..utils.helpers import setup_logging, calculate_performance_score, safe_divide

logger = setup_logging()


class AuxiliaryTableManager:
    """Manager for creating and updating auxiliary performance tables."""
    
    def __init__(self, db_manager: DatabaseManager):
        """Initialize with database manager."""
        self.db = db_manager
        
    def create_machine_performance_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create machine performance summary table."""
        logger.info("Creating machine performance summary...")
        
        machine_stats = df.groupby('machine').agg({
            'JobDuration': ['count', 'mean', 'std'],
            'RunningTime': ['mean', 'sum'],
            'efficiency': ['mean', 'std', 'min', 'max'],
            'PartsProduced': ['sum', 'mean'],
            'parts_per_hour': ['mean', 'std'],
            'total_downtime': ['mean', 'sum'],
            'SetupTime': ['mean', 'sum'],
            'MaintenanceTime': ['mean', 'sum'],
            'IdleTime': ['mean', 'sum']
        }).round(2)
        
        # Flatten column names
        machine_stats.columns = ['_'.join(col).strip() for col in machine_stats.columns]
        
        # Rename for clarity
        column_mapping = {
            'JobDuration_count': 'total_jobs',
            'JobDuration_mean': 'avg_job_duration',
            'JobDuration_std': 'job_duration_std',
            'RunningTime_mean': 'avg_running_time',
            'RunningTime_sum': 'total_running_time',
            'efficiency_mean': 'avg_efficiency',
            'efficiency_std': 'efficiency_std',
            'efficiency_min': 'min_efficiency',
            'efficiency_max': 'max_efficiency',
            'PartsProduced_sum': 'total_parts',
            'PartsProduced_mean': 'avg_parts_per_job',
            'parts_per_hour_mean': 'avg_parts_per_hour',
            'parts_per_hour_std': 'parts_per_hour_std',
            'total_downtime_mean': 'avg_downtime_per_job',
            'total_downtime_sum': 'total_downtime',
            'SetupTime_mean': 'avg_setup_time',
            'SetupTime_sum': 'total_setup_time',
            'MaintenanceTime_mean': 'avg_maintenance_time',
            'MaintenanceTime_sum': 'total_maintenance_time',
            'IdleTime_mean': 'avg_idle_time',
            'IdleTime_sum': 'total_idle_time'
        }
        
        machine_stats = machine_stats.rename(columns=column_mapping)
        
        # Calculate utilization rate
        machine_stats['utilization_rate'] = safe_divide(
            machine_stats['total_running_time'],
            machine_stats['total_running_time'] + machine_stats['total_downtime']
        )
        
        # Performance ranking
        machine_stats['efficiency_rank'] = machine_stats['avg_efficiency'].rank(ascending=False)
        machine_stats['productivity_rank'] = machine_stats['avg_parts_per_hour'].rank(ascending=False)
        
        # Overall performance score
        machine_stats['performance_score'] = machine_stats.apply(
            lambda row: calculate_performance_score(
                efficiency=row['avg_efficiency'],
                quality_score=min(row['utilization_rate'], 1.0),
                consistency_score=1 - (row['efficiency_std'] / row['avg_efficiency'] if row['avg_efficiency'] > 0 else 0)
            ), axis=1
        )
        
        machine_stats = machine_stats.reset_index()
        logger.info(f"Machine performance summary created with {len(machine_stats)} machines")
        
        return machine_stats
    
    def create_operator_performance_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create operator performance summary table."""
        logger.info("Creating operator performance summary...")
        
        operator_stats = df[df['OperatorName'] != 'Unknown'].groupby('OperatorName').agg({
            'JobDuration': ['count', 'mean'],
            'machine': 'nunique',
            'PartNumber': 'nunique',
            'efficiency': ['mean', 'std', 'min', 'max'],
            'PartsProduced': ['sum', 'mean'],
            'parts_per_hour': ['mean', 'std'],
            'SetupTime': ['mean', 'sum'],
            'total_downtime': ['mean', 'sum'],
            'StartTime': ['min', 'max']  # Experience range
        }).round(2)
        
        # Flatten column names
        operator_stats.columns = ['_'.join(col).strip() for col in operator_stats.columns]
        
        # Rename columns
        column_mapping = {
            'JobDuration_count': 'total_jobs',
            'JobDuration_mean': 'avg_job_duration',
            'machine_nunique': 'machines_operated',
            'PartNumber_nunique': 'unique_parts',
            'efficiency_mean': 'avg_efficiency',
            'efficiency_std': 'efficiency_std',
            'efficiency_min': 'min_efficiency',
            'efficiency_max': 'max_efficiency',
            'PartsProduced_sum': 'total_parts',
            'PartsProduced_mean': 'avg_parts_per_job',
            'parts_per_hour_mean': 'avg_parts_per_hour',
            'parts_per_hour_std': 'parts_per_hour_std',
            'SetupTime_mean': 'avg_setup_time',
            'SetupTime_sum': 'total_setup_time',
            'total_downtime_mean': 'avg_downtime_per_job',
            'total_downtime_sum': 'total_downtime',
            'StartTime_min': 'first_job_date',
            'StartTime_max': 'last_job_date'
        }
        
        operator_stats = operator_stats.rename(columns=column_mapping)
        
        # Calculate experience metrics
        operator_stats['experience_days'] = (
            pd.to_datetime(operator_stats['last_job_date']) - 
            pd.to_datetime(operator_stats['first_job_date'])
        ).dt.days
        
        operator_stats['jobs_per_day'] = safe_divide(
            operator_stats['total_jobs'],
            operator_stats['experience_days'] + 1  # Avoid division by zero
        )
        
        # Versatility score (machines × parts)
        operator_stats['versatility_score'] = operator_stats['machines_operated'] * operator_stats['unique_parts']
        
        # Consistency score
        operator_stats['consistency_score'] = operator_stats.apply(
            lambda row: 1 - (row['efficiency_std'] / row['avg_efficiency'] if row['avg_efficiency'] > 0 else 0),
            axis=1
        )
        
        # Performance rankings
        operator_stats['efficiency_rank'] = operator_stats['avg_efficiency'].rank(ascending=False)
        operator_stats['productivity_rank'] = operator_stats['avg_parts_per_hour'].rank(ascending=False)
        operator_stats['versatility_rank'] = operator_stats['versatility_score'].rank(ascending=False)
        
        # Overall performance score
        operator_stats['performance_score'] = operator_stats.apply(
            lambda row: calculate_performance_score(
                efficiency=row['avg_efficiency'],
                quality_score=min(row['consistency_score'], 1.0),
                consistency_score=row['consistency_score']
            ), axis=1
        )
        
        operator_stats = operator_stats.reset_index()
        logger.info(f"Operator performance summary created with {len(operator_stats)} operators")
        
        return operator_stats
    
    def create_operator_machine_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create operator-machine performance matrix."""
        logger.info("Creating operator-machine performance matrix...")
        
        matrix_data = df[df['OperatorName'] != 'Unknown'].groupby(['OperatorName', 'machine']).agg({
            'JobDuration': 'count',
            'efficiency': ['mean', 'std'],
            'parts_per_hour': 'mean',
            'SetupTime': 'mean',
            'total_downtime': 'mean'
        }).round(3)
        
        # Flatten column names
        matrix_data.columns = ['_'.join(col).strip() for col in matrix_data.columns]
        
        # Rename columns
        matrix_data = matrix_data.rename(columns={
            'JobDuration_count': 'job_count',
            'efficiency_mean': 'avg_efficiency',
            'efficiency_std': 'efficiency_std',
            'parts_per_hour_mean': 'avg_parts_per_hour',
            'SetupTime_mean': 'avg_setup_time',
            'total_downtime_mean': 'avg_downtime'
        })
        
        # Calculate proficiency score
        matrix_data['proficiency_score'] = matrix_data.apply(
            lambda row: calculate_performance_score(
                efficiency=row['avg_efficiency'],
                quality_score=1.0,  # Assume quality is good
                consistency_score=1 - (row['efficiency_std'] / row['avg_efficiency'] if row['avg_efficiency'] > 0 else 0)
            ), axis=1
        )
        
        # Add confidence level based on job count
        matrix_data['confidence_level'] = pd.cut(
            matrix_data['job_count'],
            bins=[0, 5, 15, 50, float('inf')],
            labels=['Low', 'Medium', 'High', 'Very High']
        )
        
        matrix_data = matrix_data.reset_index()
        logger.info(f"Operator-machine matrix created with {len(matrix_data)} combinations")
        
        return matrix_data
    
    def create_operator_part_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create operator-part performance matrix."""
        logger.info("Creating operator-part performance matrix...")
        
        part_matrix = df[(df['OperatorName'] != 'Unknown') & (df['PartNumber'] != 'Unknown')].groupby(['OperatorName', 'PartNumber']).agg({
            'JobDuration': 'count',
            'efficiency': ['mean', 'std'],
            'parts_per_hour': 'mean',
            'SetupTime': 'mean',
            'PartsProduced': 'mean'
        }).round(3)
        
        # Flatten and rename columns
        part_matrix.columns = ['_'.join(col).strip() for col in part_matrix.columns]
        part_matrix = part_matrix.rename(columns={
            'JobDuration_count': 'job_count',
            'efficiency_mean': 'avg_efficiency',
            'efficiency_std': 'efficiency_std',
            'parts_per_hour_mean': 'avg_parts_per_hour',
            'SetupTime_mean': 'avg_setup_time',
            'PartsProduced_mean': 'avg_parts_per_job'
        })
        
        # Calculate specialization score
        part_matrix['specialization_score'] = part_matrix.apply(
            lambda row: calculate_performance_score(
                efficiency=row['avg_efficiency'],
                quality_score=min(row['avg_parts_per_hour'] / 10, 1.0),  # Normalize parts per hour
                consistency_score=1 - (row['efficiency_std'] / row['avg_efficiency'] if row['avg_efficiency'] > 0 else 0)
            ), axis=1
        )
        
        part_matrix = part_matrix.reset_index()
        logger.info(f"Operator-part matrix created with {len(part_matrix)} combinations")
        
        return part_matrix
    
    def create_daily_utilization_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create daily utilization summary."""
        logger.info("Creating daily utilization summary...")
        
        # Extract date from StartTime
        df_daily = df.copy()
        df_daily['job_date'] = pd.to_datetime(df_daily['StartTime']).dt.date
        
        daily_stats = df_daily.groupby(['job_date', 'machine']).agg({
            'JobDuration': ['count', 'sum'],
            'RunningTime': 'sum',
            'total_downtime': 'sum',
            'PartsProduced': 'sum',
            'efficiency': 'mean'
        }).round(2)
        
        # Flatten columns
        daily_stats.columns = ['_'.join(col).strip() for col in daily_stats.columns]
        daily_stats = daily_stats.rename(columns={
            'JobDuration_count': 'job_count',
            'JobDuration_sum': 'total_job_duration',
            'RunningTime_sum': 'total_running_time',
            'total_downtime_sum': 'total_downtime',
            'PartsProduced_sum': 'total_parts',
            'efficiency_mean': 'avg_efficiency'
        })
        
        # Calculate daily utilization
        daily_stats['utilization_rate'] = safe_divide(
            daily_stats['total_running_time'],
            daily_stats['total_running_time'] + daily_stats['total_downtime']
        )
        
        # Calculate daily efficiency vs machine average
        machine_avg_efficiency = df.groupby('machine')['efficiency'].mean()
        daily_stats = daily_stats.reset_index()
        daily_stats['machine_avg_efficiency'] = daily_stats['machine'].map(machine_avg_efficiency)
        daily_stats['efficiency_vs_avg'] = daily_stats['avg_efficiency'] / daily_stats['machine_avg_efficiency']
        
        logger.info(f"Daily utilization summary created with {len(daily_stats)} records")
        
        return daily_stats
    
    def create_performance_trends(self, df: pd.DataFrame, window_days: int = 7) -> Dict[str, pd.DataFrame]:
        """Create performance trend analysis."""
        logger.info(f"Creating performance trends with {window_days}-day window...")
        
        df_trends = df.copy()
        df_trends['job_date'] = pd.to_datetime(df_trends['StartTime']).dt.date
        df_trends = df_trends.sort_values('job_date')
        
        trends = {}
        
        # Machine performance trends
        machine_trends = df_trends.groupby(['machine', 'job_date']).agg({
            'efficiency': 'mean',
            'parts_per_hour': 'mean',
            'total_downtime': 'mean'
        }).reset_index()
        
        for metric in ['efficiency', 'parts_per_hour', 'total_downtime']:
            machine_trends[f'{metric}_trend'] = machine_trends.groupby('machine')[metric].transform(
                lambda x: x.rolling(window=window_days, min_periods=1).mean()
            )
        
        trends['machine_trends'] = machine_trends
        
        # Operator performance trends
        operator_trends = df_trends[df_trends['OperatorName'] != 'Unknown'].groupby(['OperatorName', 'job_date']).agg({
            'efficiency': 'mean',
            'parts_per_hour': 'mean',
            'SetupTime': 'mean'
        }).reset_index()
        
        for metric in ['efficiency', 'parts_per_hour', 'SetupTime']:
            operator_trends[f'{metric}_trend'] = operator_trends.groupby('OperatorName')[metric].transform(
                lambda x: x.rolling(window=window_days, min_periods=1).mean()
            )
        
        trends['operator_trends'] = operator_trends
        
        logger.info("Performance trends created successfully")
        return trends
    
    def generate_all_auxiliary_tables(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Generate all auxiliary tables at once."""
        logger.info("Generating all auxiliary tables...")
        
        tables = {
            'machine_performance': self.create_machine_performance_summary(df),
            'operator_performance': self.create_operator_performance_summary(df),
            'operator_machine_matrix': self.create_operator_machine_matrix(df),
            'operator_part_matrix': self.create_operator_part_matrix(df),
            'daily_utilization': self.create_daily_utilization_summary(df)
        }
        
        # Add performance trends
        trends = self.create_performance_trends(df)
        tables.update(trends)
        
        logger.info("All auxiliary tables generated successfully")
        return tables
    
    def save_tables_to_database(self, tables: Dict[str, pd.DataFrame]) -> bool:
        """Save auxiliary tables to database."""
        try:
            with self.db.get_session() as session:
                for table_name, df in tables.items():
                    # Convert to SQL table
                    df.to_sql(
                        name=f'{table_name}_summary',
                        con=self.db.engine,
                        if_exists='replace',
                        index=False
                    )
                    logger.info(f"Saved {table_name}_summary to database ({len(df)} records)")
            
            logger.info("All auxiliary tables saved to database")
            return True
            
        except Exception as e:
            logger.error(f"Error saving tables to database: {e}")
            return False