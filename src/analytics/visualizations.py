"""Visualization and dashboard components for CNC ML analytics."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Tuple, Optional, Any
import logging

from ..utils.helpers import setup_logging

logger = setup_logging()

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class CNCAnalyticsDashboard:
    """Dashboard and visualization components for CNC analytics."""
    
    def __init__(self):
        """Initialize dashboard components."""
        self.color_scheme = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e',
            'success': '#2ca02c',
            'warning': '#d62728',
            'info': '#9467bd',
            'light': '#f8f9fa',
            'dark': '#495057'
        }
    
    def create_performance_overview(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Create performance overview visualizations."""
        logger.info("Creating performance overview visualizations...")
        
        visualizations = {}
        
        # 1. Efficiency distribution
        fig_efficiency = go.Figure()
        fig_efficiency.add_trace(go.Histogram(
            x=df['efficiency'],
            nbinsx=30,
            name='Efficiency Distribution',
            marker_color=self.color_scheme['primary'],
            opacity=0.7
        ))
        
        fig_efficiency.update_layout(
            title='Efficiency Distribution Across All Jobs',
            xaxis_title='Efficiency',
            yaxis_title='Frequency',
            template='plotly_white'
        )
        
        visualizations['efficiency_distribution'] = fig_efficiency
        
        # 2. Performance by machine
        machine_perf = df.groupby('machine').agg({
            'efficiency': ['mean', 'std', 'count']
        }).round(3)
        machine_perf.columns = ['avg_efficiency', 'efficiency_std', 'job_count']
        machine_perf = machine_perf.reset_index()
        
        fig_machine = go.Figure()
        fig_machine.add_trace(go.Bar(
            x=machine_perf['machine'],
            y=machine_perf['avg_efficiency'],
            error_y=dict(type='data', array=machine_perf['efficiency_std']),
            name='Average Efficiency',
            marker_color=self.color_scheme['secondary'],
            text=machine_perf['job_count'],
            texttemplate='Jobs: %{text}',
            textposition='outside'
        ))
        
        fig_machine.update_layout(
            title='Average Efficiency by Machine',
            xaxis_title='Machine',
            yaxis_title='Efficiency',
            template='plotly_white'
        )
        
        visualizations['machine_performance'] = fig_machine
        
        # 3. Performance by operator (top 15)
        if 'OperatorName' in df.columns:
            operator_perf = df[df['OperatorName'] != 'Unknown'].groupby('OperatorName').agg({
                'efficiency': ['mean', 'count']
            }).round(3)
            operator_perf.columns = ['avg_efficiency', 'job_count']
            operator_perf = operator_perf[operator_perf['job_count'] >= 5]  # At least 5 jobs
            operator_perf = operator_perf.nlargest(15, 'avg_efficiency').reset_index()
            
            fig_operator = go.Figure()
            fig_operator.add_trace(go.Bar(
                x=operator_perf['OperatorName'],
                y=operator_perf['avg_efficiency'],
                name='Average Efficiency',
                marker_color=self.color_scheme['success'],
                text=operator_perf['job_count'],
                texttemplate='Jobs: %{text}',
                textposition='outside'
            ))
            
            fig_operator.update_layout(
                title='Top 15 Operators by Efficiency',
                xaxis_title='Operator',
                yaxis_title='Efficiency',
                template='plotly_white',
                xaxis_tickangle=-45
            )
            
            visualizations['operator_performance'] = fig_operator
        
        # 4. Downtime analysis
        downtime_cols = ['SetupTime', 'MaintenanceTime', 'IdleTime', 'AdjustmentTime']
        available_downtime_cols = [col for col in downtime_cols if col in df.columns]
        
        if available_downtime_cols:
            downtime_data = df[available_downtime_cols].mean() / 3600  # Convert to hours
            
            fig_downtime = go.Figure(data=go.Bar(
                x=downtime_data.index,
                y=downtime_data.values,
                marker_color=self.color_scheme['warning']
            ))
            
            fig_downtime.update_layout(
                title='Average Downtime by Category (Hours)',
                xaxis_title='Downtime Category',
                yaxis_title='Hours',
                template='plotly_white'
            )
            
            visualizations['downtime_analysis'] = fig_downtime
        
        logger.info("Performance overview visualizations created")
        return visualizations
    
    def create_time_based_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Create time-based performance analysis."""
        logger.info("Creating time-based analysis...")
        
        visualizations = {}
        
        if 'StartTime' not in df.columns:
            logger.warning("No StartTime column found, skipping time-based analysis")
            return visualizations
        
        # Add time features
        df_time = df.copy()
        df_time['StartTime'] = pd.to_datetime(df_time['StartTime'])
        df_time['hour'] = df_time['StartTime'].dt.hour
        df_time['day_of_week'] = df_time['StartTime'].dt.dayofweek
        df_time['date'] = df_time['StartTime'].dt.date
        
        # 1. Hourly performance pattern
        hourly_perf = df_time.groupby('hour').agg({
            'efficiency': ['mean', 'count'],
            'total_downtime': 'mean'
        }).round(3)
        hourly_perf.columns = ['avg_efficiency', 'job_count', 'avg_downtime']
        hourly_perf = hourly_perf.reset_index()
        
        fig_hourly = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Efficiency by Hour', 'Job Volume by Hour'),
            vertical_spacing=0.1
        )
        
        fig_hourly.add_trace(
            go.Scatter(
                x=hourly_perf['hour'],
                y=hourly_perf['avg_efficiency'],
                mode='lines+markers',
                name='Efficiency',
                marker_color=self.color_scheme['primary']
            ),
            row=1, col=1
        )
        
        fig_hourly.add_trace(
            go.Bar(
                x=hourly_perf['hour'],
                y=hourly_perf['job_count'],
                name='Job Count',
                marker_color=self.color_scheme['secondary']
            ),
            row=2, col=1
        )
        
        fig_hourly.update_layout(
            title='Performance Patterns by Hour of Day',
            template='plotly_white',
            height=600
        )
        
        visualizations['hourly_patterns'] = fig_hourly
        
        # 2. Daily performance pattern
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        daily_perf = df_time.groupby('day_of_week').agg({
            'efficiency': 'mean',
            'JobDuration': 'count'
        }).round(3)
        daily_perf.index = [day_names[i] for i in daily_perf.index]
        daily_perf.columns = ['avg_efficiency', 'job_count']
        daily_perf = daily_perf.reset_index()
        
        fig_daily = go.Figure()
        fig_daily.add_trace(go.Bar(
            x=daily_perf['day_of_week'],
            y=daily_perf['avg_efficiency'],
            name='Efficiency',
            marker_color=self.color_scheme['info'],
            text=daily_perf['job_count'],
            texttemplate='Jobs: %{text}',
            textposition='outside'
        ))
        
        fig_daily.update_layout(
            title='Performance by Day of Week',
            xaxis_title='Day of Week',
            yaxis_title='Average Efficiency',
            template='plotly_white'
        )
        
        visualizations['daily_patterns'] = fig_daily
        
        # 3. Performance trend over time
        if len(df_time) > 30:  # Only if sufficient data
            daily_trend = df_time.groupby('date')['efficiency'].mean().reset_index()
            daily_trend['date'] = pd.to_datetime(daily_trend['date'])
            daily_trend = daily_trend.sort_values('date')
            
            # Calculate 7-day rolling average
            daily_trend['efficiency_ma'] = daily_trend['efficiency'].rolling(window=7, min_periods=1).mean()
            
            fig_trend = go.Figure()
            fig_trend.add_trace(go.Scatter(
                x=daily_trend['date'],
                y=daily_trend['efficiency'],
                mode='markers',
                name='Daily Efficiency',
                marker=dict(size=4, color=self.color_scheme['primary'], opacity=0.6)
            ))
            
            fig_trend.add_trace(go.Scatter(
                x=daily_trend['date'],
                y=daily_trend['efficiency_ma'],
                mode='lines',
                name='7-Day Average',
                line=dict(color=self.color_scheme['warning'], width=3)
            ))
            
            fig_trend.update_layout(
                title='Efficiency Trend Over Time',
                xaxis_title='Date',
                yaxis_title='Efficiency',
                template='plotly_white'
            )
            
            visualizations['efficiency_trend'] = fig_trend
        
        logger.info("Time-based analysis created")
        return visualizations
    
    def create_operator_analysis(self, df: pd.DataFrame, operator_profiles: Dict = None) -> Dict[str, Any]:
        """Create operator-focused analysis visualizations."""
        logger.info("Creating operator analysis...")
        
        visualizations = {}
        
        if 'OperatorName' not in df.columns:
            logger.warning("No operator data available")
            return visualizations
        
        operator_data = df[df['OperatorName'] != 'Unknown'].copy()
        
        if len(operator_data) == 0:
            return visualizations
        
        # 1. Operator efficiency vs consistency scatter plot
        operator_stats = operator_data.groupby('OperatorName').agg({
            'efficiency': ['mean', 'std', 'count']
        }).round(3)
        operator_stats.columns = ['avg_efficiency', 'efficiency_std', 'job_count']
        operator_stats = operator_stats[operator_stats['job_count'] >= 5]  # At least 5 jobs
        operator_stats['consistency_score'] = 1 / (1 + operator_stats['efficiency_std'])
        operator_stats = operator_stats.reset_index()
        
        fig_scatter = go.Figure()
        fig_scatter.add_trace(go.Scatter(
            x=operator_stats['avg_efficiency'],
            y=operator_stats['consistency_score'],
            mode='markers+text',
            text=operator_stats['OperatorName'],
            textposition='top center',
            marker=dict(
                size=operator_stats['job_count'] / 2,  # Size by job count
                color=operator_stats['avg_efficiency'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title='Efficiency')
            ),
            name='Operators'
        ))
        
        fig_scatter.update_layout(
            title='Operator Performance: Efficiency vs Consistency',
            xaxis_title='Average Efficiency',
            yaxis_title='Consistency Score',
            template='plotly_white'
        )
        
        visualizations['operator_efficiency_consistency'] = fig_scatter
        
        # 2. Operator versatility analysis
        operator_versatility = operator_data.groupby('OperatorName').agg({
            'machine': 'nunique',
            'PartNumber': 'nunique',
            'efficiency': 'mean'
        }).round(3)
        operator_versatility.columns = ['machines_operated', 'unique_parts', 'avg_efficiency']
        operator_versatility['versatility_score'] = operator_versatility['machines_operated'] * operator_versatility['unique_parts']
        operator_versatility = operator_versatility.reset_index()
        
        fig_versatility = go.Figure()
        fig_versatility.add_trace(go.Scatter(
            x=operator_versatility['versatility_score'],
            y=operator_versatility['avg_efficiency'],
            mode='markers+text',
            text=operator_versatility['OperatorName'],
            textposition='top center',
            marker=dict(
                size=12,
                color=self.color_scheme['success'],
                line=dict(width=1, color='white')
            ),
            name='Operators'
        ))
        
        fig_versatility.update_layout(
            title='Operator Versatility vs Efficiency',
            xaxis_title='Versatility Score (Machines × Parts)',
            yaxis_title='Average Efficiency',
            template='plotly_white'
        )
        
        visualizations['operator_versatility'] = fig_versatility
        
        # 3. Top operators heatmap
        top_operators = operator_stats.nlargest(10, 'avg_efficiency')['OperatorName'].tolist()
        
        if len(top_operators) > 0:
            heatmap_data = operator_data[operator_data['OperatorName'].isin(top_operators)]
            
            # Create operator-machine performance matrix
            operator_machine_perf = heatmap_data.groupby(['OperatorName', 'machine'])['efficiency'].mean().unstack(fill_value=0)
            
            if not operator_machine_perf.empty:
                fig_heatmap = go.Figure(data=go.Heatmap(
                    z=operator_machine_perf.values,
                    x=operator_machine_perf.columns,
                    y=operator_machine_perf.index,
                    colorscale='RdYlBu_r',
                    colorbar=dict(title='Efficiency')
                ))
                
                fig_heatmap.update_layout(
                    title='Top 10 Operators: Efficiency by Machine',
                    xaxis_title='Machine',
                    yaxis_title='Operator',
                    template='plotly_white'
                )
                
                visualizations['operator_machine_heatmap'] = fig_heatmap
        
        logger.info("Operator analysis created")
        return visualizations
    
    def create_machine_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Create machine-focused analysis visualizations."""
        logger.info("Creating machine analysis...")
        
        visualizations = {}
        
        # 1. Machine utilization vs efficiency
        machine_stats = df.groupby('machine').agg({
            'efficiency': 'mean',
            'RunningTime': 'sum',
            'total_downtime': 'sum',
            'JobDuration': 'count'
        }).round(3)
        
        machine_stats['utilization_rate'] = machine_stats['RunningTime'] / (
            machine_stats['RunningTime'] + machine_stats['total_downtime']
        )
        machine_stats = machine_stats.reset_index()
        
        fig_machine_perf = go.Figure()
        fig_machine_perf.add_trace(go.Scatter(
            x=machine_stats['utilization_rate'],
            y=machine_stats['efficiency'],
            mode='markers+text',
            text=machine_stats['machine'],
            textposition='top center',
            marker=dict(
                size=machine_stats['JobDuration'] / 10,  # Size by job count
                color=self.color_scheme['primary'],
                line=dict(width=1, color='white')
            ),
            name='Machines'
        ))
        
        fig_machine_perf.update_layout(
            title='Machine Performance: Utilization vs Efficiency',
            xaxis_title='Utilization Rate',
            yaxis_title='Average Efficiency',
            template='plotly_white'
        )
        
        visualizations['machine_utilization_efficiency'] = fig_machine_perf
        
        # 2. Machine downtime breakdown
        downtime_cols = ['SetupTime', 'MaintenanceTime', 'IdleTime', 'AdjustmentTime']
        available_cols = [col for col in downtime_cols if col in df.columns]
        
        if available_cols:
            machine_downtime = df.groupby('machine')[available_cols].mean() / 3600  # Convert to hours
            
            fig_downtime = go.Figure()
            
            for col in available_cols:
                fig_downtime.add_trace(go.Bar(
                    name=col,
                    x=machine_downtime.index,
                    y=machine_downtime[col]
                ))
            
            fig_downtime.update_layout(
                title='Average Downtime by Machine (Hours)',
                xaxis_title='Machine',
                yaxis_title='Hours',
                barmode='stack',
                template='plotly_white'
            )
            
            visualizations['machine_downtime_breakdown'] = fig_downtime
        
        # 3. Machine productivity analysis
        if 'PartsProduced' in df.columns:
            machine_productivity = df.groupby('machine').agg({
                'PartsProduced': 'sum',
                'RunningTime': 'sum',
                'JobDuration': 'count'
            })
            
            machine_productivity['parts_per_hour'] = (
                machine_productivity['PartsProduced'] * 3600 / machine_productivity['RunningTime']
            )
            machine_productivity = machine_productivity.reset_index()
            
            fig_productivity = go.Figure()
            fig_productivity.add_trace(go.Bar(
                x=machine_productivity['machine'],
                y=machine_productivity['parts_per_hour'],
                name='Parts per Hour',
                marker_color=self.color_scheme['secondary'],
                text=machine_productivity['JobDuration'],
                texttemplate='Jobs: %{text}',
                textposition='outside'
            ))
            
            fig_productivity.update_layout(
                title='Machine Productivity (Parts per Hour)',
                xaxis_title='Machine',
                yaxis_title='Parts per Hour',
                template='plotly_white'
            )
            
            visualizations['machine_productivity'] = fig_productivity
        
        logger.info("Machine analysis created")
        return visualizations
    
    def create_performance_matrix_visualization(self, matrices: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Create visualizations for performance matrices."""
        logger.info("Creating performance matrix visualizations...")
        
        visualizations = {}
        
        # 1. Operator-Machine performance matrix
        if 'operator_machine' in matrices and not matrices['operator_machine'].empty:
            om_matrix = matrices['operator_machine']
            
            # Create pivot table for heatmap
            heatmap_data = om_matrix.pivot(
                index='OperatorName',
                columns='machine',
                values='performance_score'
            ).fillna(0)
            
            fig_om_heatmap = go.Figure(data=go.Heatmap(
                z=heatmap_data.values,
                x=heatmap_data.columns,
                y=heatmap_data.index,
                colorscale='RdYlBu_r',
                colorbar=dict(title='Performance Score')
            ))
            
            fig_om_heatmap.update_layout(
                title='Operator-Machine Performance Matrix',
                xaxis_title='Machine',
                yaxis_title='Operator',
                template='plotly_white'
            )
            
            visualizations['operator_machine_matrix'] = fig_om_heatmap
        
        # 2. 3D performance visualization
        if 'operator_machine_part' in matrices and not matrices['operator_machine_part'].empty:
            omp_matrix = matrices['operator_machine_part'].head(20)  # Top 20 combinations
            
            fig_3d = go.Figure(data=go.Scatter3d(
                x=omp_matrix.index,
                y=omp_matrix['synergy_score'],
                z=omp_matrix['avg_efficiency'],
                mode='markers',
                marker=dict(
                    size=omp_matrix['job_count'],
                    color=omp_matrix['synergy_score'],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title='Synergy Score')
                ),
                text=[f"{row['OperatorName']}<br>{row['machine']}<br>{row['PartNumber']}" 
                      for _, row in omp_matrix.iterrows()],
                hovertemplate='%{text}<br>Efficiency: %{z}<br>Synergy: %{y}<extra></extra>'
            ))
            
            fig_3d.update_layout(
                title='3D Performance Analysis: Top Operator-Machine-Part Combinations',
                scene=dict(
                    xaxis_title='Combination Index',
                    yaxis_title='Synergy Score',
                    zaxis_title='Efficiency'
                ),
                template='plotly_white'
            )
            
            visualizations['3d_performance_analysis'] = fig_3d
        
        logger.info("Performance matrix visualizations created")
        return visualizations
    
    def create_anomaly_detection_dashboard(self, anomaly_results: Dict[str, Any]) -> Dict[str, Any]:
        """Create anomaly detection dashboard."""
        logger.info("Creating anomaly detection dashboard...")
        
        visualizations = {}
        
        if 'specific_anomalies' not in anomaly_results:
            return visualizations
        
        # 1. Anomaly type distribution
        anomaly_counts = {}
        for anomaly_type, anomalies in anomaly_results['specific_anomalies'].items():
            anomaly_counts[anomaly_type.replace('_anomalies', '').title()] = len(anomalies)
        
        fig_dist = go.Figure(data=go.Bar(
            x=list(anomaly_counts.keys()),
            y=list(anomaly_counts.values()),
            marker_color=self.color_scheme['warning']
        ))
        
        fig_dist.update_layout(
            title='Anomaly Distribution by Type',
            xaxis_title='Anomaly Type',
            yaxis_title='Count',
            template='plotly_white'
        )
        
        visualizations['anomaly_distribution'] = fig_dist
        
        # 2. Severity breakdown
        if 'anomaly_summary' in anomaly_results:
            severity_data = []
            
            for anomaly_type, summary in anomaly_results['anomaly_summary'].items():
                for severity, count in summary.get('severity_breakdown', {}).items():
                    severity_data.append({
                        'type': anomaly_type.replace('_anomalies', '').title(),
                        'severity': severity.title(),
                        'count': count
                    })
            
            if severity_data:
                severity_df = pd.DataFrame(severity_data)
                
                fig_severity = px.sunburst(
                    severity_df,
                    path=['severity', 'type'],
                    values='count',
                    title='Anomaly Severity Breakdown'
                )
                
                visualizations['anomaly_severity'] = fig_severity
        
        logger.info("Anomaly detection dashboard created")
        return visualizations
    
    def create_recommendations_dashboard(self, recommendations: List[Dict]) -> Dict[str, Any]:
        """Create recommendations dashboard."""
        logger.info("Creating recommendations dashboard...")
        
        visualizations = {}
        
        if not recommendations:
            return visualizations
        
        # 1. Recommendation confidence distribution
        confidence_counts = {}
        for rec in recommendations:
            conf = rec.get('confidence', 'Unknown')
            confidence_counts[conf] = confidence_counts.get(conf, 0) + 1
        
        fig_confidence = go.Figure(data=go.Pie(
            labels=list(confidence_counts.keys()),
            values=list(confidence_counts.values()),
            hole=0.3
        ))
        
        fig_confidence.update_layout(
            title='Recommendation Confidence Distribution',
            template='plotly_white'
        )
        
        visualizations['recommendation_confidence'] = fig_confidence
        
        # 2. Expected performance distribution
        performance_scores = [rec.get('score', 0) for rec in recommendations]
        
        fig_performance = go.Figure(data=go.Histogram(
            x=performance_scores,
            nbinsx=20,
            marker_color=self.color_scheme['success']
        ))
        
        fig_performance.update_layout(
            title='Expected Performance Score Distribution',
            xaxis_title='Performance Score',
            yaxis_title='Frequency',
            template='plotly_white'
        )
        
        visualizations['expected_performance'] = fig_performance
        
        logger.info("Recommendations dashboard created")
        return visualizations
    
    def create_comprehensive_dashboard(self, df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """Create comprehensive dashboard with all visualizations."""
        logger.info("Creating comprehensive dashboard...")
        
        all_visualizations = {}
        
        # Performance overview
        all_visualizations.update(self.create_performance_overview(df))
        
        # Time-based analysis
        all_visualizations.update(self.create_time_based_analysis(df))
        
        # Operator analysis
        all_visualizations.update(self.create_operator_analysis(df, kwargs.get('operator_profiles')))
        
        # Machine analysis
        all_visualizations.update(self.create_machine_analysis(df))
        
        # Performance matrices
        if 'matrices' in kwargs:
            all_visualizations.update(self.create_performance_matrix_visualization(kwargs['matrices']))
        
        # Anomaly detection
        if 'anomaly_results' in kwargs:
            all_visualizations.update(self.create_anomaly_detection_dashboard(kwargs['anomaly_results']))
        
        # Recommendations
        if 'recommendations' in kwargs:
            all_visualizations.update(self.create_recommendations_dashboard(kwargs['recommendations']))
        
        logger.info(f"Comprehensive dashboard created with {len(all_visualizations)} visualizations")
        return all_visualizations
    
    def save_visualizations_html(self, visualizations: Dict[str, Any], output_dir: str = "reports"):
        """Save all visualizations as HTML files."""
        import os
        
        logger.info(f"Saving visualizations to {output_dir}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        for name, fig in visualizations.items():
            filepath = os.path.join(output_dir, f"{name}.html")
            fig.write_html(filepath)
            logger.info(f"Saved {name} to {filepath}")
        
        logger.info(f"All visualizations saved to {output_dir}")
    
    def export_static_images(self, visualizations: Dict[str, Any], output_dir: str = "images"):
        """Export visualizations as static images."""
        import os
        
        logger.info(f"Exporting static images to {output_dir}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        for name, fig in visualizations.items():
            try:
                filepath = os.path.join(output_dir, f"{name}.png")
                fig.write_image(filepath, width=1200, height=800)
                logger.info(f"Exported {name} to {filepath}")
            except Exception as e:
                logger.warning(f"Could not export {name}: {e}")
        
        logger.info(f"Static images exported to {output_dir}")


def create_matplotlib_plots(df: pd.DataFrame) -> Dict[str, plt.Figure]:
    """Create matplotlib plots for static reporting."""
    logger.info("Creating matplotlib plots...")
    
    plots = {}
    
    # Set style
    plt.style.use('seaborn-v0_8')
    
    # 1. Efficiency distribution
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.hist(df['efficiency'], bins=30, alpha=0.7, color='steelblue', edgecolor='black')
    ax1.set_xlabel('Efficiency')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Efficiency Distribution')
    ax1.grid(True, alpha=0.3)
    plots['efficiency_distribution'] = fig1
    
    # 2. Machine performance comparison
    machine_perf = df.groupby('machine')['efficiency'].mean().sort_values(ascending=True)
    
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    machine_perf.plot(kind='barh', ax=ax2, color='lightcoral')
    ax2.set_xlabel('Average Efficiency')
    ax2.set_title('Machine Performance Comparison')
    ax2.grid(True, alpha=0.3)
    plots['machine_performance'] = fig2
    
    # 3. Correlation heatmap
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    correlation_matrix = df[numeric_cols].corr()
    
    fig3, ax3 = plt.subplots(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', center=0, ax=ax3)
    ax3.set_title('Feature Correlation Heatmap')
    plots['correlation_heatmap'] = fig3
    
    logger.info("Matplotlib plots created")
    return plots