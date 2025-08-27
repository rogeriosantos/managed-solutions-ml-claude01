"""Database connection and query module for CNC ML Project."""

import logging
import pandas as pd
from typing import Optional, Dict, Any, List
from sqlalchemy import create_engine, text, MetaData, Table
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import QueuePool
from sqlalchemy.exc import SQLAlchemyError
import mysql.connector
from contextlib import contextmanager

from ..utils.config import config
from ..utils.helpers import setup_logging, validate_data_quality

logger = setup_logging()


class DatabaseManager:
    """Database connection and query manager for CNC data."""
    
    def __init__(self):
        """Initialize database connection."""
        self.engine = None
        self.Session = None
        self._connect()
    
    def _connect(self):
        """Establish database connection with connection pooling."""
        try:
            connection_url = config.get_database_url()
            
            self.engine = create_engine(
                connection_url,
                poolclass=QueuePool,
                pool_size=config.DB_POOL_SIZE,
                max_overflow=config.DB_MAX_OVERFLOW,
                pool_timeout=config.DB_POOL_TIMEOUT,
                pool_pre_ping=True,
                echo=False
            )
            
            self.Session = sessionmaker(bind=self.engine)
            logger.info("Database connection established successfully")
            
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise
    
    @contextmanager
    def get_session(self):
        """Context manager for database sessions."""
        session = self.Session()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database session error: {e}")
            raise
        finally:
            session.close()
    
    def test_connection(self) -> bool:
        """Test database connection."""
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            logger.info("Database connection test successful")
            return True
        except Exception as e:
            logger.error(f"Database connection test failed: {e}")
            return False
    
    def get_table_info(self, table_name: str = 'joblog_ob') -> Dict[str, Any]:
        """Get table schema information."""
        try:
            metadata = MetaData()
            table = Table(table_name, metadata, autoload_with=self.engine)
            
            columns_info = {}
            for column in table.columns:
                columns_info[column.name] = {
                    'type': str(column.type),
                    'nullable': column.nullable,
                    'primary_key': column.primary_key
                }
            
            return {
                'table_name': table_name,
                'columns': columns_info,
                'column_count': len(table.columns)
            }
        
        except Exception as e:
            logger.error(f"Error getting table info: {e}")
            return {}
    
    def execute_query(self, query: str, params: Dict = None) -> pd.DataFrame:
        """Execute SQL query and return DataFrame."""
        try:
            with self.engine.connect() as conn:
                df = pd.read_sql(text(query), conn, params=params)
            
            logger.info(f"Query executed successfully, returned {len(df)} rows")
            return df
            
        except Exception as e:
            logger.error(f"Error executing query: {e}")
            raise
    
    def get_all_data(self, limit: Optional[int] = None) -> pd.DataFrame:
        """Retrieve all data from joblog_ob table."""
        query = "SELECT * FROM joblog_ob"
        if limit:
            query += f" LIMIT {limit}"
        
        return self.execute_query(query)
    
    def get_data_by_date_range(
        self, 
        start_date: str, 
        end_date: str,
        columns: List[str] = None
    ) -> pd.DataFrame:
        """Get data within specified date range."""
        cols = "*" if not columns else ", ".join(columns)
        
        query = f"""
        SELECT {cols}
        FROM joblog_ob 
        WHERE StartTime BETWEEN :start_date AND :end_date
        AND YEAR(StartTime) > 1970
        ORDER BY StartTime
        """
        
        params = {'start_date': start_date, 'end_date': end_date}
        return self.execute_query(query, params)
    
    def get_machine_data(self, machine_name: str) -> pd.DataFrame:
        """Get all data for specific machine."""
        query = "SELECT * FROM joblog_ob WHERE machine = :machine"
        params = {'machine': machine_name}
        return self.execute_query(query, params)
    
    def get_operator_data(self, operator_name: str) -> pd.DataFrame:
        """Get all data for specific operator."""
        query = "SELECT * FROM joblog_ob WHERE OperatorName = :operator"
        params = {'operator': operator_name}
        return self.execute_query(query, params)
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics from the database."""
        queries = {
            'total_records': "SELECT COUNT(*) as count FROM joblog_ob",
            'unique_machines': "SELECT COUNT(DISTINCT machine) as count FROM joblog_ob",
            'unique_operators': "SELECT COUNT(DISTINCT OperatorName) as count FROM joblog_ob WHERE OperatorName IS NOT NULL",
            'unique_parts': "SELECT COUNT(DISTINCT PartNumber) as count FROM joblog_ob WHERE PartNumber IS NOT NULL",
            'date_range': """
                SELECT 
                    MIN(StartTime) as min_date,
                    MAX(StartTime) as max_date 
                FROM joblog_ob 
                WHERE YEAR(StartTime) > 1970
            """,
            'job_states': """
                SELECT State, COUNT(*) as count 
                FROM joblog_ob 
                GROUP BY State
            """
        }
        
        results = {}
        for key, query in queries.items():
            try:
                df = self.execute_query(query)
                if key == 'job_states':
                    results[key] = df.to_dict('records')
                else:
                    results[key] = df.iloc[0].to_dict()
            except Exception as e:
                logger.error(f"Error executing summary query {key}: {e}")
                results[key] = None
        
        return results
    
    def get_performance_data(self) -> pd.DataFrame:
        """Get performance metrics data with calculated fields."""
        query = """
        SELECT 
            machine,
            OperatorName,
            PartNumber,
            StartTime,
            EndTime,
            JobDuration,
            RunningTime,
            PartsProduced,
            SetupTime,
            WaitingSetupTime,
            NotFeedingTime,
            AdjustmentTime,
            DressingTime,
            ToolingTime,
            EngineeringTime,
            MaintenanceTime,
            BuyInTime,
            BreakShiftChangeTime,
            IdleTime,
            CASE 
                WHEN JobDuration > 0 THEN RunningTime / JobDuration 
                ELSE 0 
            END as efficiency,
            CASE 
                WHEN RunningTime > 0 AND PartsProduced > 0 THEN PartsProduced / (RunningTime / 3600.0)
                ELSE 0 
            END as parts_per_hour
        FROM joblog_ob
        WHERE JobDuration > 0 
        AND YEAR(StartTime) > 1970
        ORDER BY StartTime DESC
        """
        
        return self.execute_query(query)
    
    def get_downtime_analysis(self) -> pd.DataFrame:
        """Get downtime analysis data."""
        query = """
        SELECT 
            machine,
            OperatorName,
            PartNumber,
            DATE(StartTime) as job_date,
            COUNT(*) as job_count,
            AVG(JobDuration) as avg_job_duration,
            AVG(RunningTime) as avg_running_time,
            AVG(SetupTime) as avg_setup_time,
            AVG(MaintenanceTime) as avg_maintenance_time,
            AVG(IdleTime) as avg_idle_time,
            SUM(SetupTime + WaitingSetupTime + NotFeedingTime + AdjustmentTime + 
                DressingTime + ToolingTime + EngineeringTime + MaintenanceTime + 
                BuyInTime + BreakShiftChangeTime + IdleTime) as total_downtime
        FROM joblog_ob
        WHERE YEAR(StartTime) > 1970
        GROUP BY machine, OperatorName, PartNumber, DATE(StartTime)
        HAVING total_downtime > 0
        ORDER BY job_date DESC, total_downtime DESC
        """
        
        return self.execute_query(query)
    
    def get_clean_data(self, apply_filters: bool = True) -> pd.DataFrame:
        """Get cleaned dataset with quality filters applied."""
        df = self.get_all_data()
        
        if apply_filters:
            # Remove invalid timestamps
            df = df[pd.to_datetime(df['StartTime'], errors='coerce').dt.year > 1970]
            
            # Remove negative durations
            duration_cols = [
                'JobDuration', 'RunningTime', 'SetupTime', 'WaitingSetupTime',
                'NotFeedingTime', 'AdjustmentTime', 'DressingTime', 'ToolingTime',
                'EngineeringTime', 'MaintenanceTime', 'BuyInTime', 
                'BreakShiftChangeTime', 'IdleTime'
            ]
            
            for col in duration_cols:
                if col in df.columns:
                    df = df[df[col] >= 0]
            
            # Remove extremely high idle times (> 24 hours)
            if 'IdleTime' in df.columns:
                df = df[df['IdleTime'] <= 86400]
            
            # Remove jobs with zero or negative duration
            df = df[df['JobDuration'] > 0]
            
            logger.info(f"Cleaned dataset contains {len(df)} records")
        
        return df
    
    def create_auxiliary_tables(self) -> bool:
        """Create auxiliary summary tables for performance analysis."""
        try:
            with self.get_session() as session:
                # Machine performance summary
                machine_summary_query = """
                CREATE TABLE IF NOT EXISTS machine_performance_summary AS
                SELECT 
                    machine,
                    COUNT(*) as total_jobs,
                    AVG(CASE WHEN JobDuration > 0 THEN RunningTime / JobDuration ELSE 0 END) as avg_efficiency,
                    AVG(JobDuration) as avg_job_duration,
                    SUM(PartsProduced) as total_parts_produced,
                    AVG(PartsProduced / (RunningTime / 3600.0)) as avg_parts_per_hour
                FROM joblog_ob
                WHERE JobDuration > 0 AND YEAR(StartTime) > 1970
                GROUP BY machine
                """
                
                # Operator performance summary
                operator_summary_query = """
                CREATE TABLE IF NOT EXISTS operator_performance_summary AS
                SELECT 
                    OperatorName,
                    COUNT(*) as total_jobs,
                    COUNT(DISTINCT machine) as machines_operated,
                    AVG(CASE WHEN JobDuration > 0 THEN RunningTime / JobDuration ELSE 0 END) as avg_efficiency,
                    AVG(SetupTime) as avg_setup_time,
                    SUM(PartsProduced) as total_parts_produced
                FROM joblog_ob
                WHERE OperatorName IS NOT NULL AND JobDuration > 0 AND YEAR(StartTime) > 1970
                GROUP BY OperatorName
                """
                
                session.execute(text(machine_summary_query))
                session.execute(text(operator_summary_query))
                
                logger.info("Auxiliary tables created successfully")
                return True
                
        except Exception as e:
            logger.error(f"Error creating auxiliary tables: {e}")
            return False


# Global database instance
db_manager = DatabaseManager()


def get_database() -> DatabaseManager:
    """Get database manager instance."""
    return db_manager