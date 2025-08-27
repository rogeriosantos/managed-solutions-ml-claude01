"""Performance matrix analysis for operator-machine-part combinations."""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from itertools import combinations

from ..utils.helpers import setup_logging, calculate_performance_score, safe_divide

logger = setup_logging()


class PerformanceMatrixAnalyzer:
    """Analyze performance across operator-machine-part combinations."""
    
    def __init__(self):
        """Initialize performance matrix analyzer."""
        self.matrices = {}
        self.performance_scores = {}
        self.recommendations = {}
    
    def create_performance_matrices(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Create various performance matrices."""
        logger.info("Creating performance matrices...")
        
        matrices = {}
        
        # Operator-Machine matrix
        matrices['operator_machine'] = self._create_operator_machine_matrix(df)
        
        # Operator-Part matrix
        matrices['operator_part'] = self._create_operator_part_matrix(df)
        
        # Machine-Part matrix
        matrices['machine_part'] = self._create_machine_part_matrix(df)
        
        # 3D Operator-Machine-Part analysis
        matrices['operator_machine_part'] = self._create_3d_performance_matrix(df)
        
        # Time-based performance matrix
        matrices['time_performance'] = self._create_time_performance_matrix(df)
        
        self.matrices = matrices
        logger.info("Performance matrices created successfully")
        
        return matrices
    
    def _create_operator_machine_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create operator-machine performance matrix."""
        logger.info("Creating operator-machine performance matrix...")
        
        # Filter out unknown operators
        operator_data = df[df['OperatorName'] != 'Unknown'].copy()
        
        if len(operator_data) == 0:
            return pd.DataFrame()
        
        # Group by operator and machine
        matrix = operator_data.groupby(['OperatorName', 'machine']).agg({
            'JobDuration': 'count',
            'efficiency': ['mean', 'std', 'min', 'max'],
            'parts_per_hour': 'mean',
            'SetupTime': 'mean',
            'total_downtime': 'mean',
            'PartsProduced': 'sum'
        }).round(3)
        
        # Flatten column names
        matrix.columns = ['_'.join(col).strip() for col in matrix.columns]
        
        # Rename for clarity
        matrix = matrix.rename(columns={
            'JobDuration_count': 'job_count',
            'efficiency_mean': 'avg_efficiency',
            'efficiency_std': 'efficiency_std',
            'efficiency_min': 'min_efficiency',
            'efficiency_max': 'max_efficiency',
            'parts_per_hour_mean': 'avg_parts_per_hour',
            'SetupTime_mean': 'avg_setup_time',
            'total_downtime_mean': 'avg_downtime',
            'PartsProduced_sum': 'total_parts'
        })
        
        # Calculate performance score
        matrix['performance_score'] = matrix.apply(
            lambda row: calculate_performance_score(
                efficiency=row['avg_efficiency'],
                quality_score=min(row['avg_parts_per_hour'] / 10, 1.0),  # Normalize
                consistency_score=1 - (row['efficiency_std'] / row['avg_efficiency'] if row['avg_efficiency'] > 0 else 0)
            ), axis=1
        )
        
        # Confidence level based on job count
        matrix['confidence_level'] = pd.cut(
            matrix['job_count'],
            bins=[0, 2, 5, 15, 50, float('inf')],
            labels=['Very Low', 'Low', 'Medium', 'High', 'Very High']
        )
        
        # Specialization indicator (how much better than operator average)
        operator_avg_efficiency = operator_data.groupby('OperatorName')['efficiency'].mean()
        matrix = matrix.reset_index()
        matrix['operator_avg_efficiency'] = matrix['OperatorName'].map(operator_avg_efficiency)
        matrix['specialization_factor'] = matrix['avg_efficiency'] / matrix['operator_avg_efficiency']
        
        return matrix
    
    def _create_operator_part_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create operator-part performance matrix."""
        logger.info("Creating operator-part performance matrix...")
        
        # Filter out unknown operators and parts
        valid_data = df[(df['OperatorName'] != 'Unknown') & (df['PartNumber'] != 'Unknown')].copy()
        
        if len(valid_data) == 0:
            return pd.DataFrame()
        
        # Group by operator and part
        matrix = valid_data.groupby(['OperatorName', 'PartNumber']).agg({
            'JobDuration': 'count',
            'efficiency': ['mean', 'std'],
            'SetupTime': 'mean',
            'PartsProduced': ['sum', 'mean'],
            'parts_per_hour': 'mean'
        }).round(3)
        
        # Flatten columns
        matrix.columns = ['_'.join(col).strip() for col in matrix.columns]
        matrix = matrix.rename(columns={
            'JobDuration_count': 'job_count',
            'efficiency_mean': 'avg_efficiency',
            'efficiency_std': 'efficiency_std',
            'SetupTime_mean': 'avg_setup_time',
            'PartsProduced_sum': 'total_parts',
            'PartsProduced_mean': 'avg_parts_per_job',
            'parts_per_hour_mean': 'avg_parts_per_hour'
        })
        
        # Calculate part expertise score
        matrix['expertise_score'] = matrix.apply(
            lambda row: calculate_performance_score(
                efficiency=row['avg_efficiency'],
                quality_score=min(row['avg_parts_per_hour'] / 5, 1.0),  # Normalize
                consistency_score=1 - (row['efficiency_std'] / row['avg_efficiency'] if row['avg_efficiency'] > 0 else 0)
            ), axis=1
        )
        
        # Learning indicator (improvement over time would require time-based analysis)
        matrix = matrix.reset_index()
        
        # Part complexity relative to operator's typical parts
        part_complexity = valid_data.groupby('PartNumber')['SetupTime'].mean()
        matrix['part_complexity'] = matrix['PartNumber'].map(part_complexity)
        
        operator_avg_complexity = valid_data.groupby('OperatorName')['SetupTime'].mean()
        matrix['operator_avg_complexity'] = matrix['OperatorName'].map(operator_avg_complexity)
        matrix['complexity_ratio'] = matrix['part_complexity'] / matrix['operator_avg_complexity']
        
        return matrix
    
    def _create_machine_part_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create machine-part performance matrix."""
        logger.info("Creating machine-part performance matrix...")
        
        valid_data = df[df['PartNumber'] != 'Unknown'].copy()
        
        if len(valid_data) == 0:
            return pd.DataFrame()
        
        # Group by machine and part
        matrix = valid_data.groupby(['machine', 'PartNumber']).agg({
            'JobDuration': 'count',
            'efficiency': ['mean', 'std'],
            'SetupTime': 'mean',
            'RunningTime': 'mean',
            'PartsProduced': 'sum',
            'parts_per_hour': 'mean'
        }).round(3)
        
        # Flatten columns
        matrix.columns = ['_'.join(col).strip() for col in matrix.columns]
        matrix = matrix.rename(columns={
            'JobDuration_count': 'job_count',
            'efficiency_mean': 'avg_efficiency',
            'efficiency_std': 'efficiency_std',
            'SetupTime_mean': 'avg_setup_time',
            'RunningTime_mean': 'avg_running_time',
            'PartsProduced_sum': 'total_parts',
            'parts_per_hour_mean': 'avg_parts_per_hour'
        })
        
        # Machine-Part suitability score
        matrix['suitability_score'] = matrix.apply(
            lambda row: calculate_performance_score(
                efficiency=row['avg_efficiency'],
                quality_score=min(row['avg_parts_per_hour'] / 8, 1.0),
                consistency_score=1 - (row['efficiency_std'] / row['avg_efficiency'] if row['avg_efficiency'] > 0 else 0)
            ), axis=1
        )
        
        matrix = matrix.reset_index()
        
        # Calculate setup efficiency (how well machine handles this part's setup)
        part_avg_setup = valid_data.groupby('PartNumber')['SetupTime'].mean()
        matrix['part_avg_setup'] = matrix['PartNumber'].map(part_avg_setup)
        matrix['setup_efficiency'] = matrix['part_avg_setup'] / matrix['avg_setup_time']
        
        return matrix
    
    def _create_3d_performance_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create 3D operator-machine-part performance matrix."""
        logger.info("Creating 3D performance matrix...")
        
        valid_data = df[
            (df['OperatorName'] != 'Unknown') & 
            (df['PartNumber'] != 'Unknown')
        ].copy()
        
        if len(valid_data) == 0:
            return pd.DataFrame()
        
        # Group by all three dimensions
        matrix = valid_data.groupby(['OperatorName', 'machine', 'PartNumber']).agg({
            'JobDuration': 'count',
            'efficiency': ['mean', 'std'],
            'SetupTime': 'mean',
            'parts_per_hour': 'mean',
            'total_downtime': 'mean'
        }).round(3)
        
        # Flatten columns
        matrix.columns = ['_'.join(col).strip() for col in matrix.columns]
        matrix = matrix.rename(columns={
            'JobDuration_count': 'job_count',
            'efficiency_mean': 'avg_efficiency',
            'efficiency_std': 'efficiency_std',
            'SetupTime_mean': 'avg_setup_time',
            'parts_per_hour_mean': 'avg_parts_per_hour',
            'total_downtime_mean': 'avg_downtime'
        })
        
        # Only keep combinations with sufficient data
        matrix = matrix[matrix['job_count'] >= 2].copy()
        
        if len(matrix) == 0:
            return matrix
        
        # Calculate synergy score (how well the combination works)
        matrix['synergy_score'] = matrix.apply(
            lambda row: calculate_performance_score(
                efficiency=row['avg_efficiency'],
                quality_score=min(row['avg_parts_per_hour'] / 6, 1.0),
                consistency_score=1 - (row['efficiency_std'] / row['avg_efficiency'] if row['avg_efficiency'] > 0 else 0)
            ), axis=1
        )
        
        matrix = matrix.reset_index()
        
        # Compare to individual component performances
        operator_performance = valid_data.groupby('OperatorName')['efficiency'].mean()
        machine_performance = valid_data.groupby('machine')['efficiency'].mean()
        part_avg_efficiency = valid_data.groupby('PartNumber')['efficiency'].mean()
        
        matrix['operator_baseline'] = matrix['OperatorName'].map(operator_performance)
        matrix['machine_baseline'] = matrix['machine'].map(machine_performance)
        matrix['part_baseline'] = matrix['PartNumber'].map(part_avg_efficiency)
        
        # Calculate synergy indicators
        matrix['operator_synergy'] = matrix['avg_efficiency'] - matrix['operator_baseline']
        matrix['machine_synergy'] = matrix['avg_efficiency'] - matrix['machine_baseline']
        matrix['overall_synergy'] = matrix['avg_efficiency'] - (
            (matrix['operator_baseline'] + matrix['machine_baseline'] + matrix['part_baseline']) / 3
        )
        
        return matrix
    
    def _create_time_performance_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based performance analysis."""
        logger.info("Creating time-based performance matrix...")
        
        if 'StartTime' not in df.columns:
            return pd.DataFrame()
        
        # Add time features if not present
        df_time = df.copy()
        if 'hour' not in df_time.columns:
            df_time['hour'] = pd.to_datetime(df_time['StartTime']).dt.hour
        if 'day_of_week' not in df_time.columns:
            df_time['day_of_week'] = pd.to_datetime(df_time['StartTime']).dt.dayofweek
        
        # Performance by hour and day
        time_matrix = df_time.groupby(['hour', 'day_of_week']).agg({
            'JobDuration': 'count',
            'efficiency': ['mean', 'std'],
            'parts_per_hour': 'mean',
            'total_downtime': 'mean'
        }).round(3)
        
        # Flatten columns
        time_matrix.columns = ['_'.join(col).strip() for col in time_matrix.columns]
        time_matrix = time_matrix.rename(columns={
            'JobDuration_count': 'job_count',
            'efficiency_mean': 'avg_efficiency',
            'efficiency_std': 'efficiency_std',
            'parts_per_hour_mean': 'avg_parts_per_hour',
            'total_downtime_mean': 'avg_downtime'
        })
        
        time_matrix = time_matrix.reset_index()
        
        # Add shift classification
        def classify_shift(hour):
            if 6 <= hour < 14:
                return 'Day'
            elif 14 <= hour < 22:
                return 'Evening'
            else:
                return 'Night'
        
        time_matrix['shift'] = time_matrix['hour'].apply(classify_shift)
        
        # Day name for clarity
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        time_matrix['day_name'] = time_matrix['day_of_week'].apply(lambda x: day_names[x])
        
        return time_matrix
    
    def identify_optimal_combinations(self, matrices: Dict[str, pd.DataFrame] = None) -> Dict[str, Any]:
        """Identify optimal operator-machine-part combinations."""
        logger.info("Identifying optimal combinations...")
        
        if matrices is None:
            matrices = self.matrices
        
        optimal_combinations = {}
        
        # Best operator-machine combinations
        if 'operator_machine' in matrices and not matrices['operator_machine'].empty:
            om_matrix = matrices['operator_machine']
            
            # Top combinations by performance score
            top_om = om_matrix[om_matrix['confidence_level'].isin(['High', 'Very High'])].nlargest(10, 'performance_score')
            optimal_combinations['top_operator_machine'] = top_om[
                ['OperatorName', 'machine', 'performance_score', 'avg_efficiency', 'job_count']
            ].to_dict('records')
            
            # Most specialized combinations
            specialized_om = om_matrix[om_matrix['specialization_factor'] > 1.1].nlargest(5, 'specialization_factor')
            optimal_combinations['specialized_combinations'] = specialized_om[
                ['OperatorName', 'machine', 'specialization_factor', 'avg_efficiency']
            ].to_dict('records')
        
        # Best operator-part combinations
        if 'operator_part' in matrices and not matrices['operator_part'].empty:
            op_matrix = matrices['operator_part']
            
            top_op = op_matrix[op_matrix['job_count'] >= 3].nlargest(10, 'expertise_score')
            optimal_combinations['top_operator_part'] = top_op[
                ['OperatorName', 'PartNumber', 'expertise_score', 'avg_efficiency', 'job_count']
            ].to_dict('records')
        
        # Best machine-part combinations
        if 'machine_part' in matrices and not matrices['machine_part'].empty:
            mp_matrix = matrices['machine_part']
            
            top_mp = mp_matrix[mp_matrix['job_count'] >= 5].nlargest(10, 'suitability_score')
            optimal_combinations['top_machine_part'] = top_mp[
                ['machine', 'PartNumber', 'suitability_score', 'avg_efficiency', 'job_count']
            ].to_dict('records')
        
        # Best 3D combinations
        if 'operator_machine_part' in matrices and not matrices['operator_machine_part'].empty:
            omp_matrix = matrices['operator_machine_part']
            
            top_omp = omp_matrix.nlargest(10, 'synergy_score')
            optimal_combinations['top_3d_combinations'] = top_omp[
                ['OperatorName', 'machine', 'PartNumber', 'synergy_score', 'avg_efficiency', 'job_count']
            ].to_dict('records')
            
            # Highest synergy combinations (outperforming expectations)
            high_synergy = omp_matrix[omp_matrix['overall_synergy'] > 0.05].nlargest(5, 'overall_synergy')
            optimal_combinations['high_synergy_combinations'] = high_synergy[
                ['OperatorName', 'machine', 'PartNumber', 'overall_synergy', 'avg_efficiency']
            ].to_dict('records')
        
        return optimal_combinations
    
    def identify_improvement_opportunities(self, matrices: Dict[str, pd.DataFrame] = None) -> Dict[str, Any]:
        """Identify opportunities for improvement."""
        logger.info("Identifying improvement opportunities...")
        
        if matrices is None:
            matrices = self.matrices
        
        opportunities = {}
        
        # Underperforming combinations
        if 'operator_machine' in matrices and not matrices['operator_machine'].empty:
            om_matrix = matrices['operator_machine']
            
            # Combinations with high job count but low performance
            underperforming = om_matrix[
                (om_matrix['job_count'] >= 5) & 
                (om_matrix['performance_score'] < 0.6)
            ].nsmallest(10, 'performance_score')
            
            opportunities['underperforming_operator_machine'] = underperforming[
                ['OperatorName', 'machine', 'performance_score', 'avg_efficiency', 'job_count']
            ].to_dict('records')
        
        # High variability combinations (consistency issues)
        if 'operator_machine' in matrices and not matrices['operator_machine'].empty:
            om_matrix = matrices['operator_machine']
            
            inconsistent = om_matrix[
                (om_matrix['job_count'] >= 3) & 
                (om_matrix['efficiency_std'] > 0.15)
            ].nlargest(10, 'efficiency_std')
            
            opportunities['inconsistent_combinations'] = inconsistent[
                ['OperatorName', 'machine', 'efficiency_std', 'avg_efficiency', 'job_count']
            ].to_dict('records')
        
        # Cross-training opportunities
        opportunities['cross_training'] = self._identify_cross_training_opportunities(matrices)
        
        # Machine optimization opportunities
        opportunities['machine_optimization'] = self._identify_machine_optimization(matrices)
        
        return opportunities
    
    def _identify_cross_training_opportunities(self, matrices: Dict[str, pd.DataFrame]) -> List[Dict]:
        """Identify cross-training opportunities for operators."""
        opportunities = []
        
        if 'operator_machine' not in matrices or matrices['operator_machine'].empty:
            return opportunities
        
        om_matrix = matrices['operator_machine']
        
        # Find operators who perform well on specific machines
        high_performers = om_matrix[
            (om_matrix['performance_score'] > 0.8) & 
            (om_matrix['job_count'] >= 3)
        ]
        
        # Group by operator to see their strong machines
        for operator in high_performers['OperatorName'].unique():
            operator_strengths = high_performers[high_performers['OperatorName'] == operator]
            strong_machines = set(operator_strengths['machine'].tolist())
            
            # Find all machines and suggest similar ones for cross-training
            all_machines = set(om_matrix['machine'].unique())
            potential_machines = all_machines - strong_machines
            
            if potential_machines:
                # Simple recommendation based on average setup time similarity
                current_avg_setup = operator_strengths['avg_setup_time'].mean()
                
                for machine in list(potential_machines)[:3]:  # Limit to top 3
                    machine_data = om_matrix[om_matrix['machine'] == machine]
                    if not machine_data.empty:
                        machine_avg_setup = machine_data['avg_setup_time'].mean()
                        similarity = 1 / (1 + abs(current_avg_setup - machine_avg_setup) / 100)
                        
                        opportunities.append({
                            'operator': operator,
                            'recommended_machine': machine,
                            'similarity_score': similarity,
                            'current_strong_machines': list(strong_machines),
                            'reason': f"Similar setup complexity ({machine_avg_setup:.0f}s vs {current_avg_setup:.0f}s avg)"
                        })
        
        return sorted(opportunities, key=lambda x: x['similarity_score'], reverse=True)[:10]
    
    def _identify_machine_optimization(self, matrices: Dict[str, pd.DataFrame]) -> List[Dict]:
        """Identify machine optimization opportunities."""
        opportunities = []
        
        if 'machine_part' not in matrices or matrices['machine_part'].empty:
            return opportunities
        
        mp_matrix = matrices['machine_part']
        
        # Find parts that perform poorly on certain machines but well on others
        part_machine_performance = {}
        
        for part in mp_matrix['PartNumber'].unique():
            part_data = mp_matrix[mp_matrix['PartNumber'] == part]
            if len(part_data) > 1:  # Part runs on multiple machines
                best_machine = part_data.loc[part_data['avg_efficiency'].idxmax()]
                worst_machine = part_data.loc[part_data['avg_efficiency'].idxmin()]
                
                efficiency_gap = best_machine['avg_efficiency'] - worst_machine['avg_efficiency']
                
                if efficiency_gap > 0.1:  # Significant difference
                    opportunities.append({
                        'part_number': part,
                        'current_machine': worst_machine['machine'],
                        'recommended_machine': best_machine['machine'],
                        'efficiency_gain': efficiency_gap,
                        'current_efficiency': worst_machine['avg_efficiency'],
                        'potential_efficiency': best_machine['avg_efficiency']
                    })
        
        return sorted(opportunities, key=lambda x: x['efficiency_gain'], reverse=True)[:10]
    
    def generate_performance_insights(self, matrices: Dict[str, pd.DataFrame] = None) -> Dict[str, Any]:
        """Generate comprehensive performance insights."""
        logger.info("Generating performance insights...")
        
        if matrices is None:
            matrices = self.matrices
        
        insights = {
            'optimal_combinations': self.identify_optimal_combinations(matrices),
            'improvement_opportunities': self.identify_improvement_opportunities(matrices),
            'performance_statistics': self._calculate_performance_statistics(matrices),
            'trend_analysis': self._analyze_performance_trends(matrices)
        }
        
        return insights
    
    def _calculate_performance_statistics(self, matrices: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Calculate performance statistics across matrices."""
        stats = {}
        
        for matrix_name, matrix in matrices.items():
            if matrix.empty:
                continue
                
            if 'avg_efficiency' in matrix.columns:
                stats[matrix_name] = {
                    'mean_efficiency': matrix['avg_efficiency'].mean(),
                    'median_efficiency': matrix['avg_efficiency'].median(),
                    'std_efficiency': matrix['avg_efficiency'].std(),
                    'min_efficiency': matrix['avg_efficiency'].min(),
                    'max_efficiency': matrix['avg_efficiency'].max(),
                    'combinations_count': len(matrix)
                }
                
                if 'performance_score' in matrix.columns:
                    stats[matrix_name]['mean_performance_score'] = matrix['performance_score'].mean()
                
                if 'job_count' in matrix.columns:
                    stats[matrix_name]['total_jobs'] = matrix['job_count'].sum()
                    stats[matrix_name]['avg_jobs_per_combination'] = matrix['job_count'].mean()
        
        return stats
    
    def _analyze_performance_trends(self, matrices: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Analyze performance trends from the matrices."""
        trends = {}
        
        # Time-based trends
        if 'time_performance' in matrices and not matrices['time_performance'].empty:
            time_matrix = matrices['time_performance']
            
            # Best and worst hours
            hourly_performance = time_matrix.groupby('hour')['avg_efficiency'].mean()
            trends['best_hours'] = hourly_performance.nlargest(3).to_dict()
            trends['worst_hours'] = hourly_performance.nsmallest(3).to_dict()
            
            # Best and worst days
            daily_performance = time_matrix.groupby('day_name')['avg_efficiency'].mean()
            trends['best_days'] = daily_performance.nlargest(3).to_dict()
            trends['worst_days'] = daily_performance.nsmallest(3).to_dict()
            
            # Shift performance
            shift_performance = time_matrix.groupby('shift')['avg_efficiency'].mean()
            trends['shift_performance'] = shift_performance.to_dict()
        
        return trends