"""Recommendation engine for operator-machine-part assignments."""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import logging

from ..utils.helpers import setup_logging, calculate_performance_score

logger = setup_logging()


class AssignmentRecommendationEngine:
    """Recommendation engine for optimal operator-machine-part assignments."""
    
    def __init__(self):
        """Initialize recommendation engine."""
        self.operator_profiles = {}
        self.machine_profiles = {}
        self.part_profiles = {}
        self.similarity_matrices = {}
        self.performance_data = None
        self.is_fitted = False
    
    def fit(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Train the recommendation engine with historical data."""
        logger.info("Training recommendation engine...")
        
        self.performance_data = df.copy()
        
        # Create profiles for operators, machines, and parts
        self.operator_profiles = self._create_operator_profiles(df)
        self.machine_profiles = self._create_machine_profiles(df)
        self.part_profiles = self._create_part_profiles(df)
        
        # Create similarity matrices
        self.similarity_matrices = self._create_similarity_matrices()
        
        self.is_fitted = True
        
        results = {
            'n_operators': len(self.operator_profiles),
            'n_machines': len(self.machine_profiles),
            'n_parts': len(self.part_profiles),
            'training_samples': len(df)
        }
        
        logger.info("Recommendation engine training complete")
        return results
    
    def _create_operator_profiles(self, df: pd.DataFrame) -> Dict[str, Dict]:
        """Create profiles for each operator."""
        logger.info("Creating operator profiles...")
        
        # Filter out unknown operators
        operator_data = df[df['OperatorName'] != 'Unknown'].copy()
        
        profiles = {}
        
        for operator in operator_data['OperatorName'].unique():
            op_data = operator_data[operator_data['OperatorName'] == operator]
            
            # Basic performance metrics
            profile = {
                'avg_efficiency': op_data['efficiency'].mean(),
                'efficiency_std': op_data['efficiency'].std(),
                'avg_setup_time': op_data['SetupTime'].mean(),
                'avg_parts_per_hour': op_data['parts_per_hour'].mean() if 'parts_per_hour' in op_data.columns else 0,
                'total_jobs': len(op_data),
                'machines_operated': list(op_data['machine'].unique()),
                'parts_worked': list(op_data['PartNumber'].unique()),
                'avg_downtime': op_data['total_downtime'].mean()
            }
            
            # Machine preferences (based on performance)
            machine_performance = op_data.groupby('machine').agg({
                'efficiency': ['mean', 'count'],
                'SetupTime': 'mean'
            }).round(3)
            
            machine_performance.columns = ['efficiency', 'job_count', 'setup_time']
            machine_preferences = {}
            
            for machine in machine_performance.index:
                if machine_performance.loc[machine, 'job_count'] >= 2:  # At least 2 jobs
                    machine_preferences[machine] = {
                        'efficiency': machine_performance.loc[machine, 'efficiency'],
                        'job_count': machine_performance.loc[machine, 'job_count'],
                        'setup_time': machine_performance.loc[machine, 'setup_time']
                    }
            
            profile['machine_preferences'] = machine_preferences
            
            # Part expertise (based on performance)
            if 'PartNumber' in op_data.columns:
                part_performance = op_data[op_data['PartNumber'] != 'Unknown'].groupby('PartNumber').agg({
                    'efficiency': ['mean', 'count'],
                    'SetupTime': 'mean'
                }).round(3)
                
                if len(part_performance) > 0:
                    part_performance.columns = ['efficiency', 'job_count', 'setup_time']
                    part_expertise = {}
                    
                    for part in part_performance.index:
                        if part_performance.loc[part, 'job_count'] >= 1:
                            part_expertise[part] = {
                                'efficiency': part_performance.loc[part, 'efficiency'],
                                'job_count': part_performance.loc[part, 'job_count'],
                                'setup_time': part_performance.loc[part, 'setup_time']
                            }
                    
                    profile['part_expertise'] = part_expertise
                else:
                    profile['part_expertise'] = {}
            else:
                profile['part_expertise'] = {}
            
            # Skill characteristics
            profile['versatility_score'] = len(profile['machines_operated']) * len(profile['parts_worked'])
            profile['consistency_score'] = 1 / (1 + profile['efficiency_std']) if profile['efficiency_std'] > 0 else 1
            profile['setup_expertise'] = 1 / (1 + profile['avg_setup_time'] / 1000)  # Normalized
            
            profiles[operator] = profile
        
        logger.info(f"Created profiles for {len(profiles)} operators")
        return profiles
    
    def _create_machine_profiles(self, df: pd.DataFrame) -> Dict[str, Dict]:
        """Create profiles for each machine."""
        logger.info("Creating machine profiles...")
        
        profiles = {}
        
        for machine in df['machine'].unique():
            machine_data = df[df['machine'] == machine]
            
            profile = {
                'avg_efficiency': machine_data['efficiency'].mean(),
                'efficiency_std': machine_data['efficiency'].std(),
                'avg_job_duration': machine_data['JobDuration'].mean(),
                'avg_setup_time': machine_data['SetupTime'].mean(),
                'total_jobs': len(machine_data),
                'operators_used': list(machine_data['OperatorName'].unique()),
                'parts_produced': list(machine_data['PartNumber'].unique()),
                'utilization_rate': machine_data['RunningTime'].sum() / (machine_data['RunningTime'].sum() + machine_data['total_downtime'].sum())
            }
            
            # Operator suitability (which operators perform best on this machine)
            operator_performance = machine_data[machine_data['OperatorName'] != 'Unknown'].groupby('OperatorName').agg({
                'efficiency': ['mean', 'count'],
                'SetupTime': 'mean'
            }).round(3)
            
            if len(operator_performance) > 0:
                operator_performance.columns = ['efficiency', 'job_count', 'setup_time']
                operator_suitability = {}
                
                for operator in operator_performance.index:
                    if operator_performance.loc[operator, 'job_count'] >= 2:
                        operator_suitability[operator] = {
                            'efficiency': operator_performance.loc[operator, 'efficiency'],
                            'job_count': operator_performance.loc[operator, 'job_count'],
                            'setup_time': operator_performance.loc[operator, 'setup_time']
                        }
                
                profile['operator_suitability'] = operator_suitability
            else:
                profile['operator_suitability'] = {}
            
            # Part suitability
            part_performance = machine_data[machine_data['PartNumber'] != 'Unknown'].groupby('PartNumber').agg({
                'efficiency': ['mean', 'count'],
                'SetupTime': 'mean'
            }).round(3)
            
            if len(part_performance) > 0:
                part_performance.columns = ['efficiency', 'job_count', 'setup_time']
                part_suitability = {}
                
                for part in part_performance.index:
                    if part_performance.loc[part, 'job_count'] >= 1:
                        part_suitability[part] = {
                            'efficiency': part_performance.loc[part, 'efficiency'],
                            'job_count': part_performance.loc[part, 'job_count'],
                            'setup_time': part_performance.loc[part, 'setup_time']
                        }
                
                profile['part_suitability'] = part_suitability
            else:
                profile['part_suitability'] = {}
            
            # Machine characteristics
            profile['complexity_score'] = profile['avg_setup_time'] / 1000  # Normalized
            profile['reliability_score'] = 1 / (1 + machine_data['MaintenanceTime'].mean() / 1000) if 'MaintenanceTime' in machine_data.columns else 1
            
            profiles[machine] = profile
        
        logger.info(f"Created profiles for {len(profiles)} machines")
        return profiles
    
    def _create_part_profiles(self, df: pd.DataFrame) -> Dict[str, Dict]:
        """Create profiles for each part."""
        logger.info("Creating part profiles...")
        
        # Filter out unknown parts
        part_data = df[df['PartNumber'] != 'Unknown'].copy()
        
        profiles = {}
        
        for part in part_data['PartNumber'].unique():
            p_data = part_data[part_data['PartNumber'] == part]
            
            profile = {
                'avg_efficiency': p_data['efficiency'].mean(),
                'avg_setup_time': p_data['SetupTime'].mean(),
                'avg_parts_per_job': p_data['PartsProduced'].mean() if 'PartsProduced' in p_data.columns else 0,
                'total_jobs': len(p_data),
                'machines_used': list(p_data['machine'].unique()),
                'operators_used': list(p_data['OperatorName'].unique()),
                'production_frequency': len(p_data)  # How often this part is made
            }
            
            # Machine suitability for this part
            machine_performance = p_data.groupby('machine').agg({
                'efficiency': ['mean', 'count'],
                'SetupTime': 'mean'
            }).round(3)
            
            machine_performance.columns = ['efficiency', 'job_count', 'setup_time']
            machine_suitability = {}
            
            for machine in machine_performance.index:
                if machine_performance.loc[machine, 'job_count'] >= 1:
                    machine_suitability[machine] = {
                        'efficiency': machine_performance.loc[machine, 'efficiency'],
                        'job_count': machine_performance.loc[machine, 'job_count'],
                        'setup_time': machine_performance.loc[machine, 'setup_time']
                    }
            
            profile['machine_suitability'] = machine_suitability
            
            # Operator expertise for this part
            operator_performance = p_data[p_data['OperatorName'] != 'Unknown'].groupby('OperatorName').agg({
                'efficiency': ['mean', 'count'],
                'SetupTime': 'mean'
            }).round(3)
            
            if len(operator_performance) > 0:
                operator_performance.columns = ['efficiency', 'job_count', 'setup_time']
                operator_expertise = {}
                
                for operator in operator_performance.index:
                    if operator_performance.loc[operator, 'job_count'] >= 1:
                        operator_expertise[operator] = {
                            'efficiency': operator_performance.loc[operator, 'efficiency'],
                            'job_count': operator_performance.loc[operator, 'job_count'],
                            'setup_time': operator_performance.loc[operator, 'setup_time']
                        }
                
                profile['operator_expertise'] = operator_expertise
            else:
                profile['operator_expertise'] = {}
            
            # Part characteristics
            profile['complexity_score'] = profile['avg_setup_time'] / 1000  # Normalized
            profile['demand_score'] = profile['production_frequency'] / len(df)  # Normalized
            
            profiles[part] = profile
        
        logger.info(f"Created profiles for {len(profiles)} parts")
        return profiles
    
    def _create_similarity_matrices(self) -> Dict[str, np.ndarray]:
        """Create similarity matrices for collaborative filtering."""
        logger.info("Creating similarity matrices...")
        
        matrices = {}
        
        # Operator similarity matrix
        if self.operator_profiles:
            operator_features = []
            operator_names = list(self.operator_profiles.keys())
            
            for operator in operator_names:
                profile = self.operator_profiles[operator]
                features = [
                    profile['avg_efficiency'],
                    profile['avg_setup_time'] / 1000,  # Normalize
                    profile['versatility_score'] / 10,  # Normalize
                    profile['consistency_score']
                ]
                operator_features.append(features)
            
            operator_features = np.array(operator_features)
            
            # Handle NaN values
            operator_features = np.nan_to_num(operator_features, nan=0.0)
            
            if len(operator_features) > 1:
                matrices['operator_similarity'] = cosine_similarity(operator_features)
            else:
                matrices['operator_similarity'] = np.array([[1.0]])
        
        # Machine similarity matrix
        if self.machine_profiles:
            machine_features = []
            machine_names = list(self.machine_profiles.keys())
            
            for machine in machine_names:
                profile = self.machine_profiles[machine]
                features = [
                    profile['avg_efficiency'],
                    profile['avg_setup_time'] / 1000,  # Normalize
                    profile['complexity_score'],
                    profile['utilization_rate']
                ]
                machine_features.append(features)
            
            machine_features = np.array(machine_features)
            machine_features = np.nan_to_num(machine_features, nan=0.0)
            
            if len(machine_features) > 1:
                matrices['machine_similarity'] = cosine_similarity(machine_features)
            else:
                matrices['machine_similarity'] = np.array([[1.0]])
        
        return matrices
    
    def recommend_operator_for_job(self, machine: str, part: str = None, n_recommendations: int = 5) -> List[Dict]:
        """Recommend best operators for a specific machine and part combination."""
        if not self.is_fitted:
            raise ValueError("Recommendation engine must be fitted first")
        
        logger.info(f"Generating operator recommendations for machine={machine}, part={part}")
        
        recommendations = []
        
        # Get all available operators
        available_operators = list(self.operator_profiles.keys())
        
        for operator in available_operators:
            score = self._calculate_operator_job_score(operator, machine, part)
            
            recommendation = {
                'operator': operator,
                'score': score,
                'confidence': self._calculate_confidence(operator, machine, part),
                'reasons': self._get_recommendation_reasons(operator, machine, part)
            }
            
            recommendations.append(recommendation)
        
        # Sort by score and return top N
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        
        return recommendations[:n_recommendations]
    
    def recommend_machine_for_part(self, part: str, operator: str = None, n_recommendations: int = 5) -> List[Dict]:
        """Recommend best machines for a specific part and operator combination."""
        if not self.is_fitted:
            raise ValueError("Recommendation engine must be fitted first")
        
        logger.info(f"Generating machine recommendations for part={part}, operator={operator}")
        
        recommendations = []
        
        # Get all available machines
        available_machines = list(self.machine_profiles.keys())
        
        for machine in available_machines:
            score = self._calculate_machine_part_score(machine, part, operator)
            
            recommendation = {
                'machine': machine,
                'score': score,
                'confidence': self._calculate_confidence(operator, machine, part),
                'reasons': self._get_machine_recommendation_reasons(machine, part, operator)
            }
            
            recommendations.append(recommendation)
        
        # Sort by score and return top N
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        
        return recommendations[:n_recommendations]
    
    def recommend_optimal_assignment(self, jobs: List[Dict]) -> List[Dict]:
        """Recommend optimal assignments for a list of jobs."""
        logger.info(f"Generating optimal assignments for {len(jobs)} jobs")
        
        assignments = []
        
        for job in jobs:
            machine = job.get('machine')
            part = job.get('part')
            
            if machine:
                # Recommend operator for given machine and part
                operator_recs = self.recommend_operator_for_job(machine, part, n_recommendations=1)
                
                if operator_recs:
                    assignment = {
                        'job_id': job.get('job_id', 'unknown'),
                        'machine': machine,
                        'part': part,
                        'recommended_operator': operator_recs[0]['operator'],
                        'expected_performance': operator_recs[0]['score'],
                        'confidence': operator_recs[0]['confidence'],
                        'reasons': operator_recs[0]['reasons']
                    }
                else:
                    assignment = {
                        'job_id': job.get('job_id', 'unknown'),
                        'machine': machine,
                        'part': part,
                        'recommended_operator': 'Any',
                        'expected_performance': 0.5,
                        'confidence': 'Low',
                        'reasons': ['No historical data available']
                    }
            
            elif part:
                # Recommend machine and operator for given part
                machine_recs = self.recommend_machine_for_part(part, n_recommendations=1)
                
                if machine_recs:
                    recommended_machine = machine_recs[0]['machine']
                    operator_recs = self.recommend_operator_for_job(recommended_machine, part, n_recommendations=1)
                    
                    assignment = {
                        'job_id': job.get('job_id', 'unknown'),
                        'recommended_machine': recommended_machine,
                        'part': part,
                        'recommended_operator': operator_recs[0]['operator'] if operator_recs else 'Any',
                        'expected_performance': operator_recs[0]['score'] if operator_recs else machine_recs[0]['score'],
                        'confidence': 'Medium',
                        'reasons': machine_recs[0]['reasons'] + (operator_recs[0]['reasons'] if operator_recs else [])
                    }
                else:
                    assignment = {
                        'job_id': job.get('job_id', 'unknown'),
                        'recommended_machine': 'Any',
                        'part': part,
                        'recommended_operator': 'Any',
                        'expected_performance': 0.5,
                        'confidence': 'Low',
                        'reasons': ['No historical data available']
                    }
            
            assignments.append(assignment)
        
        return assignments
    
    def _calculate_operator_job_score(self, operator: str, machine: str, part: str = None) -> float:
        """Calculate score for operator on specific job."""
        operator_profile = self.operator_profiles.get(operator, {})
        machine_profile = self.machine_profiles.get(machine, {})
        
        score_components = []
        
        # Direct experience score
        if machine in operator_profile.get('machine_preferences', {}):
            machine_perf = operator_profile['machine_preferences'][machine]
            direct_score = machine_perf['efficiency'] * 0.7  # Weight by efficiency
            # Boost for more experience
            experience_boost = min(machine_perf['job_count'] / 10, 0.3)
            score_components.append(direct_score + experience_boost)
        else:
            # No direct experience, use base performance
            score_components.append(operator_profile.get('avg_efficiency', 0.5) * 0.5)
        
        # Part expertise score
        if part and part in operator_profile.get('part_expertise', {}):
            part_perf = operator_profile['part_expertise'][part]
            part_score = part_perf['efficiency'] * 0.6
            score_components.append(part_score)
        elif part:
            # No part experience, use average
            score_components.append(operator_profile.get('avg_efficiency', 0.5) * 0.3)
        
        # Operator overall performance
        base_score = operator_profile.get('avg_efficiency', 0.5) * 0.4
        consistency_bonus = operator_profile.get('consistency_score', 0.5) * 0.2
        score_components.extend([base_score, consistency_bonus])
        
        # Machine-operator compatibility
        if operator in machine_profile.get('operator_suitability', {}):
            compatibility_score = machine_profile['operator_suitability'][operator]['efficiency'] * 0.3
            score_components.append(compatibility_score)
        
        # Calculate weighted average
        final_score = np.mean(score_components) if score_components else 0.5
        
        return min(final_score, 1.0)  # Cap at 1.0
    
    def _calculate_machine_part_score(self, machine: str, part: str, operator: str = None) -> float:
        """Calculate score for machine-part combination."""
        machine_profile = self.machine_profiles.get(machine, {})
        part_profile = self.part_profiles.get(part, {})
        
        score_components = []
        
        # Direct machine-part experience
        if part in machine_profile.get('part_suitability', {}):
            part_perf = machine_profile['part_suitability'][part]
            direct_score = part_perf['efficiency'] * 0.7
            score_components.append(direct_score)
        else:
            # No direct experience, use machine average
            score_components.append(machine_profile.get('avg_efficiency', 0.5) * 0.5)
        
        # Part-machine suitability from part perspective
        if machine in part_profile.get('machine_suitability', {}):
            machine_perf = part_profile['machine_suitability'][machine]
            suitability_score = machine_perf['efficiency'] * 0.6
            score_components.append(suitability_score)
        
        # Machine overall performance
        base_score = machine_profile.get('avg_efficiency', 0.5) * 0.4
        utilization_bonus = machine_profile.get('utilization_rate', 0.5) * 0.2
        score_components.extend([base_score, utilization_bonus])
        
        # Operator compatibility if provided
        if operator and operator in machine_profile.get('operator_suitability', {}):
            operator_compat = machine_profile['operator_suitability'][operator]['efficiency'] * 0.3
            score_components.append(operator_compat)
        
        final_score = np.mean(score_components) if score_components else 0.5
        return min(final_score, 1.0)
    
    def _calculate_confidence(self, operator: str = None, machine: str = None, part: str = None) -> str:
        """Calculate confidence level for recommendation."""
        confidence_score = 0
        total_factors = 0
        
        if operator and machine:
            operator_profile = self.operator_profiles.get(operator, {})
            if machine in operator_profile.get('machine_preferences', {}):
                job_count = operator_profile['machine_preferences'][machine]['job_count']
                if job_count >= 10:
                    confidence_score += 3
                elif job_count >= 5:
                    confidence_score += 2
                elif job_count >= 2:
                    confidence_score += 1
            total_factors += 1
        
        if operator and part:
            operator_profile = self.operator_profiles.get(operator, {})
            if part in operator_profile.get('part_expertise', {}):
                job_count = operator_profile['part_expertise'][part]['job_count']
                if job_count >= 5:
                    confidence_score += 2
                elif job_count >= 2:
                    confidence_score += 1
            total_factors += 1
        
        if machine and part:
            machine_profile = self.machine_profiles.get(machine, {})
            if part in machine_profile.get('part_suitability', {}):
                job_count = machine_profile['part_suitability'][part]['job_count']
                if job_count >= 5:
                    confidence_score += 2
                elif job_count >= 2:
                    confidence_score += 1
            total_factors += 1
        
        if total_factors == 0:
            return 'Low'
        
        avg_confidence = confidence_score / total_factors
        
        if avg_confidence >= 2:
            return 'High'
        elif avg_confidence >= 1:
            return 'Medium'
        else:
            return 'Low'
    
    def _get_recommendation_reasons(self, operator: str, machine: str, part: str = None) -> List[str]:
        """Get reasons for operator recommendation."""
        reasons = []
        
        operator_profile = self.operator_profiles.get(operator, {})
        
        # Machine experience
        if machine in operator_profile.get('machine_preferences', {}):
            machine_perf = operator_profile['machine_preferences'][machine]
            reasons.append(f"Has {machine_perf['job_count']} jobs experience on {machine} with {machine_perf['efficiency']:.2f} efficiency")
        else:
            reasons.append(f"No direct experience on {machine}, but overall efficiency is {operator_profile.get('avg_efficiency', 0):.2f}")
        
        # Part expertise
        if part and part in operator_profile.get('part_expertise', {}):
            part_perf = operator_profile['part_expertise'][part]
            reasons.append(f"Has experience with {part} ({part_perf['job_count']} jobs, {part_perf['efficiency']:.2f} efficiency)")
        elif part:
            reasons.append(f"No direct experience with {part}")
        
        # Overall performance characteristics
        if operator_profile.get('consistency_score', 0) > 0.8:
            reasons.append("Highly consistent performer")
        
        if operator_profile.get('versatility_score', 0) > 10:
            reasons.append("Versatile operator (works on multiple machines/parts)")
        
        return reasons
    
    def _get_machine_recommendation_reasons(self, machine: str, part: str, operator: str = None) -> List[str]:
        """Get reasons for machine recommendation."""
        reasons = []
        
        machine_profile = self.machine_profiles.get(machine, {})
        part_profile = self.part_profiles.get(part, {})
        
        # Machine-part suitability
        if part in machine_profile.get('part_suitability', {}):
            part_perf = machine_profile['part_suitability'][part]
            reasons.append(f"Machine has produced {part} {part_perf['job_count']} times with {part_perf['efficiency']:.2f} efficiency")
        
        # Part-machine preference
        if machine in part_profile.get('machine_suitability', {}):
            machine_perf = part_profile['machine_suitability'][machine]
            reasons.append(f"Part performs well on this machine ({machine_perf['efficiency']:.2f} efficiency)")
        
        # Machine performance
        if machine_profile.get('avg_efficiency', 0) > 0.8:
            reasons.append(f"High performing machine (avg efficiency: {machine_profile['avg_efficiency']:.2f})")
        
        if machine_profile.get('utilization_rate', 0) > 0.7:
            reasons.append("Well-utilized machine")
        
        return reasons
    
    def generate_assignment_report(self, assignments: List[Dict]) -> Dict[str, Any]:
        """Generate comprehensive assignment report."""
        logger.info("Generating assignment report...")
        
        # Performance predictions
        performance_predictions = {
            'high_performance': [a for a in assignments if a['expected_performance'] > 0.8],
            'medium_performance': [a for a in assignments if 0.6 <= a['expected_performance'] <= 0.8],
            'low_performance': [a for a in assignments if a['expected_performance'] < 0.6]
        }
        
        # Confidence distribution
        confidence_distribution = {}
        for conf_level in ['High', 'Medium', 'Low']:
            confidence_distribution[conf_level] = len([a for a in assignments if a['confidence'] == conf_level])
        
        # Risk assessment
        risk_assignments = [a for a in assignments if a['confidence'] == 'Low' and a['expected_performance'] < 0.6]
        
        report = {
            'total_assignments': len(assignments),
            'performance_predictions': {
                'high_performance_count': len(performance_predictions['high_performance']),
                'medium_performance_count': len(performance_predictions['medium_performance']),
                'low_performance_count': len(performance_predictions['low_performance']),
                'average_expected_performance': np.mean([a['expected_performance'] for a in assignments])
            },
            'confidence_distribution': confidence_distribution,
            'risk_assessment': {
                'high_risk_assignments': len(risk_assignments),
                'risk_assignments_details': risk_assignments[:5]  # Show top 5 risky assignments
            },
            'recommendations': self._generate_assignment_recommendations(assignments)
        }
        
        return report
    
    def _generate_assignment_recommendations(self, assignments: List[Dict]) -> List[str]:
        """Generate recommendations based on assignments."""
        recommendations = []
        
        # Check for low confidence assignments
        low_confidence = [a for a in assignments if a['confidence'] == 'Low']
        if low_confidence:
            recommendations.append(f"Monitor {len(low_confidence)} low-confidence assignments closely")
        
        # Check for low performance predictions
        low_performance = [a for a in assignments if a['expected_performance'] < 0.6]
        if low_performance:
            recommendations.append(f"Consider alternative assignments for {len(low_performance)} jobs with low performance predictions")
        
        # Check for operator workload distribution
        operator_counts = {}
        for assignment in assignments:
            op = assignment.get('recommended_operator', 'Unknown')
            operator_counts[op] = operator_counts.get(op, 0) + 1
        
        max_assignments = max(operator_counts.values()) if operator_counts else 0
        if max_assignments > len(assignments) * 0.4:  # One operator has >40% of assignments
            overloaded_ops = [op for op, count in operator_counts.items() if count > len(assignments) * 0.3]
            recommendations.append(f"Consider redistributing workload from operators: {', '.join(overloaded_ops)}")
        
        return recommendations