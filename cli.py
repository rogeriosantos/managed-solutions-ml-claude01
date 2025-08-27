#!/usr/bin/env python3
"""
Command Line Interface for CNC ML Project
Provides easy-to-use commands for common operations.
"""

import click
import pandas as pd
import logging
from pathlib import Path

from src.data.database import DatabaseManager
from src.data.preprocessing import DataPreprocessor
from src.models.efficiency_predictor import EfficiencyPredictor
from src.models.downtime_classifier import DowntimeClassifier
from src.analytics.visualizations import CNCAnalyticsDashboard
from src.utils.helpers import setup_logging


@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
def cli(verbose):
    """CNC ML Project - Command Line Interface"""
    log_level = 'DEBUG' if verbose else 'INFO'
    setup_logging(log_level)


@cli.command()
@click.option('--limit', '-l', default=1000, help='Number of records to load (default: 1000)')
@click.option('--save', '-s', is_flag=True, help='Save results to CSV')
def explore(limit, save):
    """Explore and analyze CNC data."""
    click.echo(f"🔍 Exploring CNC data (limit: {limit:,} records)...")
    
    try:
        # Connect to database
        db = DatabaseManager()
        if not db.test_connection():
            click.echo("❌ Database connection failed!")
            return
        
        # Load data
        df = db.get_all_data(limit=limit)
        click.echo(f"✅ Loaded {len(df):,} records")
        
        # Basic statistics
        click.echo(f"📊 Date range: {df['StartTime'].min()} to {df['StartTime'].max()}")
        click.echo(f"🏭 Machines: {df['machine'].nunique()}")
        click.echo(f"👷 Operators: {df['OperatorName'].nunique()}")
        click.echo(f"🔧 Parts: {df['PartNumber'].nunique()}")
        
        # Process data
        preprocessor = DataPreprocessor()
        df_clean = preprocessor.validate_raw_data(df)
        df_features = preprocessor.create_derived_features(df_clean)
        
        click.echo(f"⚙️ After preprocessing: {len(df_features):,} records, {df_features.shape[1]} features")
        
        if 'efficiency' in df_features.columns:
            avg_eff = df_features['efficiency'].mean()
            click.echo(f"📈 Average efficiency: {avg_eff:.3f}")
        
        # Save if requested
        if save:
            output_file = f"cnc_data_exploration_{limit}.csv"
            df_features.to_csv(output_file, index=False)
            click.echo(f"💾 Results saved to {output_file}")
        
        click.echo("✅ Exploration complete!")
        
    except Exception as e:
        click.echo(f"❌ Error: {e}")


@cli.command()
@click.option('--model', '-m', default='efficiency', type=click.Choice(['efficiency', 'downtime']),
              help='Model type to train')
@click.option('--optimize', is_flag=True, help='Enable hyperparameter optimization')
@click.option('--save-model', is_flag=True, help='Save trained model')
def train(model, optimize, save_model):
    """Train ML models on CNC data."""
    click.echo(f"🎯 Training {model} model...")
    
    try:
        # Load data
        db = DatabaseManager()
        df = db.get_clean_data(apply_filters=True)
        
        if len(df) < 100:
            click.echo("⚠️ Insufficient data for training (need at least 100 records)")
            return
        
        click.echo(f"📊 Training data: {len(df):,} records")
        
        # Preprocess
        preprocessor = DataPreprocessor()
        df_features = preprocessor.create_derived_features(df)
        
        # Train model
        if model == 'efficiency':
            predictor = EfficiencyPredictor()
            results = predictor.train(df_features, optimize_hyperparameters=optimize)
            
            click.echo(f"📈 Test R² Score: {results['test_metrics']['r2']:.3f}")
            click.echo(f"📊 Test RMSE: {results['test_metrics']['rmse']:.3f}")
            
            if save_model:
                model_path = "models/efficiency_predictor.joblib"
                predictor.save_model(model_path)
                click.echo(f"💾 Model saved to {model_path}")
        
        elif model == 'downtime':
            classifier = DowntimeClassifier()
            X, y = classifier.prepare_training_data(df_features)
            
            if len(X) < 50:
                click.echo("⚠️ Insufficient data for downtime classification")
                return
            
            results = classifier.train(X, y, optimize_hyperparameters=optimize)
            
            click.echo(f"🎯 Test Accuracy: {results['test_accuracy']:.3f}")
            click.echo(f"📊 Training samples: {len(X):,}")
            
            if save_model:
                model_path = "models/downtime_classifier.joblib"
                classifier.save_model(model_path)
                click.echo(f"💾 Model saved to {model_path}")
        
        click.echo("✅ Training complete!")
        
    except Exception as e:
        click.echo(f"❌ Training error: {e}")


@cli.command()
@click.option('--output', '-o', default='dashboard', help='Output directory for visualizations')
def dashboard(output):
    """Generate interactive dashboard and visualizations."""
    click.echo("📊 Generating CNC analytics dashboard...")
    
    try:
        # Load data
        db = DatabaseManager()
        df = db.get_all_data(limit=5000)  # Use more data for better visualizations
        
        # Preprocess
        preprocessor = DataPreprocessor()
        df_clean = preprocessor.validate_raw_data(df)
        df_features = preprocessor.create_derived_features(df_clean)
        
        # Create dashboard
        dashboard_generator = CNCAnalyticsDashboard()
        
        # Generate all visualizations
        click.echo("🎨 Creating visualizations...")
        visualizations = dashboard_generator.create_comprehensive_dashboard(df_features)
        
        # Save visualizations
        import os
        os.makedirs(output, exist_ok=True)
        dashboard_generator.save_visualizations_html(visualizations, output)
        
        click.echo(f"📊 Dashboard saved to {output}/ directory")
        click.echo(f"🌐 Open {output}/efficiency_distribution.html in your browser to view")
        click.echo("✅ Dashboard generation complete!")
        
    except Exception as e:
        click.echo(f"❌ Dashboard error: {e}")


@cli.command()
def status():
    """Check system status and configuration."""
    click.echo("🔍 Checking CNC ML system status...")
    
    try:
        # Database connection
        db = DatabaseManager()
        if db.test_connection():
            click.echo("✅ Database: Connected")
            
            # Get summary stats
            summary = db.get_summary_stats()
            total_records = summary.get('total_records', {}).get('count', 'Unknown')
            click.echo(f"📊 Total records: {total_records}")
            
        else:
            click.echo("❌ Database: Connection failed")
        
        # Check data directory
        data_dir = Path('data/processed')
        if data_dir.exists():
            files = list(data_dir.glob('*.csv'))
            click.echo(f"📁 Processed data files: {len(files)}")
        else:
            click.echo("📁 No processed data directory found")
        
        # Check models directory
        models_dir = Path('models')
        if models_dir.exists():
            models = list(models_dir.glob('*.joblib'))
            click.echo(f"🤖 Trained models: {len(models)}")
        else:
            click.echo("🤖 No models directory found")
        
        click.echo("✅ Status check complete!")
        
    except Exception as e:
        click.echo(f"❌ Status check error: {e}")


@cli.command()
@click.argument('machine')
@click.argument('part', required=False)
@click.option('--top', '-t', default=3, help='Number of recommendations')
def recommend(machine, part, top):
    """Get operator recommendations for a job."""
    click.echo(f"🎯 Getting operator recommendations for machine '{machine}'" + 
               (f" and part '{part}'" if part else ""))
    
    try:
        # This would require a trained recommendation model
        # For now, show example output
        click.echo("🤖 Recommendation engine would analyze:")
        click.echo(f"  - Historical performance on {machine}")
        if part:
            click.echo(f"  - Experience with part {part}")
        click.echo(f"  - Current operator availability")
        click.echo(f"  - Skill compatibility scores")
        
        click.echo(f"\n📋 Top {top} Recommendations:")
        click.echo("  1. Operator_A (Score: 0.92, Confidence: High)")
        click.echo("  2. Operator_B (Score: 0.87, Confidence: Medium)")
        click.echo("  3. Operator_C (Score: 0.81, Confidence: Medium)")
        
        click.echo("💡 Note: Train recommendation models with full data for actual predictions")
        
    except Exception as e:
        click.echo(f"❌ Recommendation error: {e}")


if __name__ == '__main__':
    cli()