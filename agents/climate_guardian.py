"""
Climate Guardian - Temporal Projection for Environmental Factors
================================================================

Purpose: Manage temporal projections from 2025-2050 for:
- Sea state deterioration (resistance penalty)
- Climate change impacts on vessel performance
- Regulatory evolution (emission standards)

Provides dynamic analysis capabilities instead of static assessment
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from scipy import stats
from typing import Dict, List, Tuple, Optional
import json


class ClimateGuardian:
    """
    Climate Guardian for temporal projections and climate impact assessment
    
    Manages time-series projections of environmental factors and their
    impact on vessel performance, regulations, and operational conditions.
    """
    
    def __init__(self):
        self.base_year = 2025
        self.end_year = 2050
        self.current_year = datetime.now().year
        
        # Climate projection parameters
        self.sea_level_rise_rate = 0.003  # meters per year
        self.wave_intensification = 0.008  # annual increase in wave height
        self.wind_intensification = 0.005  # annual increase in wind speed
        self.storm_frequency_increase = 0.015  # annual increase in storm frequency
        
        # Regulatory evolution parameters
        self.cii_tightening_rate = 0.03  # 3% annual tightening
        self.carbon_tax_increase_rate = 0.05  # 5% annual increase
        self.emission_standards_tightening = 0.04  # 4% annual tightening
        self.base_carbon_tax = 100  # USD per ton CO2 (2025)
        
        # Environmental factor projections
        self.environmental_projections = {}
        self.regulatory_projections = {}
        
        # Historical data for validation
        self.historical_years = list(range(2015, self.current_year + 1))
        self.historical_data = self.generate_historical_data()
    
    def generate_historical_data(self) -> Dict:
        """
        Generate historical environmental data for model validation
        
        Returns:
            Dictionary with historical environmental data
        """
        historical_data = {
            'years': self.historical_years,
            'average_wave_height': [],
            'average_wind_speed': [],
            'storm_days_per_year': [],
            'sea_level': [],
            'average_temperature': []
        }
        
        base_wave_height = 1.8  # meters
        base_wind_speed = 12    # knots
        base_storm_days = 45    # days per year
        base_sea_level = 0      # relative to 2015
        base_temperature = 15   # Celsius
        
        for year in self.historical_years:
            year_factor = (year - 2015) / 10  # Decade factor
            
            historical_data['average_wave_height'].append(
                base_wave_height + year_factor * 0.1 + np.random.normal(0, 0.1)
            )
            historical_data['average_wind_speed'].append(
                base_wind_speed + year_factor * 0.5 + np.random.normal(0, 0.5)
            )
            historical_data['storm_days_per_year'].append(
                base_storm_days + year_factor * 5 + np.random.normal(0, 3)
            )
            historical_data['sea_level'].append(
                base_sea_level + year_factor * 0.02 + np.random.normal(0, 0.01)
            )
            historical_data['average_temperature'].append(
                base_temperature + year_factor * 0.2 + np.random.normal(0, 0.2)
            )
        
        return historical_data
    
    def project_environmental_conditions(self, target_year: int) -> Dict:
        """
        Project environmental conditions for a target year
        
        Args:
            target_year: Year to project conditions for
        
        Returns:
            Dictionary with projected environmental conditions
        """
        base_wave_height = 2.2  # 2025 baseline
        base_wind_speed = 15    # 2025 baseline
        base_storm_days = 55    # 2025 baseline
        base_sea_level = 0.08   # 2025 baseline (meters above 2015)
        base_temperature = 16.5 # 2025 baseline
        
        # Calculate projection factor
        year_diff = target_year - self.base_year
        projection_factor = year_diff
        
        # Project conditions with uncertainty bands
        conditions = {
            'target_year': target_year,
            'wave_height': {
                'mean': base_wave_height + projection_factor * self.wave_intensification * 10,
                'std': 0.3 + projection_factor * 0.02,
                'percentile_95': base_wave_height + projection_factor * self.wave_intensification * 15
            },
            'wind_speed': {
                'mean': base_wind_speed + projection_factor * self.wind_intensification * 10,
                'std': 2.0 + projection_factor * 0.1,
                'percentile_95': base_wind_speed + projection_factor * self.wind_intensification * 15
            },
            'storm_days': {
                'mean': base_storm_days + projection_factor * self.storm_frequency_increase * 20,
                'std': 8 + projection_factor * 0.5,
                'percentile_95': base_storm_days + projection_factor * self.storm_frequency_increase * 30
            },
            'sea_level': {
                'mean': base_sea_level + projection_factor * self.sea_level_rise_rate,
                'std': 0.02,
                'percentile_95': base_sea_level + projection_factor * self.sea_level_rise_rate * 1.5
            },
            'temperature': {
                'mean': base_temperature + projection_factor * 0.03,
                'std': 0.5,
                'percentile_95': base_temperature + projection_factor * 0.05
            }
        }
        
        return conditions
    
    def project_regulatory_changes(self, target_year: int) -> Dict:
        """
        Project regulatory changes and their impact
        
        Args:
            target_year: Year to project regulations for
        
        Returns:
            Dictionary with projected regulatory framework
        """
        base_carbon_tax = getattr(self, 'base_carbon_tax', 100)
        base_cii_limit = 3.5       # CII rating limit (2025)
        base_eedi_limit = 15       # EEDI limit (2025)
        base_sox_fee = 50          # USD per ton SOx
        base_nox_fee = 25          # USD per ton NOx
        
        year_diff = target_year - self.base_year
        
        regulations = {
            'target_year': target_year,
            'carbon_tax': {
                'rate': base_carbon_tax * (1 + self.carbon_tax_increase_rate) ** year_diff,
                'currency': 'USD/tCO2',
                'description': 'Carbon tax on CO2 emissions'
            },
            'cii_regulation': {
                'limit': base_cii_limit * (1 - self.cii_tightening_rate) ** year_diff,
                'unit': 'gCO2/t·nm',
                'description': 'Carbon Intensity Indicator limit'
            },
            'eedi_regulation': {
                'limit': base_eedi_limit * (1 - self.emission_standards_tightening) ** year_diff,
                'unit': 'gCO2/t·nm',
                'description': 'Energy Efficiency Design Index limit'
            },
            'emission_fees': {
                'sox': base_sox_fee * (1 + self.emission_standards_tightening) ** year_diff,
                'nox': base_nox_fee * (1 + self.emission_standards_tightening) ** year_diff,
                'currency': 'USD/ton'
            },
            'regulatory_tightening_factor': (1 + self.cii_tightening_rate) ** year_diff
        }
        
        return regulations
    
    def calculate_resistance_penalty(self, wave_height: float, wind_speed: float, 
                                   sea_state: int) -> float:
        """
        Calculate resistance penalty due to environmental conditions
        
        Args:
            wave_height: Significant wave height (meters)
            wind_speed: Wind speed (knots)
            sea_state: Sea state (1-9 scale)
        
        Returns:
            Resistance penalty factor (1.0 = no penalty)
        """
        # Base resistance penalty calculation
        wave_penalty = 1 + (wave_height / 10) ** 2 * 0.3
        wind_penalty = 1 + (wind_speed / 50) ** 2 * 0.2
        sea_state_penalty = 1 + (sea_state - 1) * 0.05
        
        # Combined penalty (multiplicative)
        total_penalty = wave_penalty * wind_penalty * sea_state_penalty
        
        return total_penalty
    
    def project_vessel_performance_impact(self, vessel_data: Dict, 
                                        target_year: int) -> Dict:
        """
        Project vessel performance impact due to climate change
        
        Args:
            vessel_data: Current vessel characteristics
            target_year: Year to project performance for
        
        Returns:
            Dictionary with projected performance impacts
        """
        # Get environmental projections
        env_conditions = self.project_environmental_conditions(target_year)
        regulations = self.project_regulatory_changes(target_year)
        
        # Current vessel performance
        current_fuel = vessel_data.get('fuel_consumption', 15)
        current_speed = vessel_data.get('speed', 12)
        current_cii = vessel_data.get('cii_score', 4.2)
        
        # Calculate resistance penalty
        avg_wave_height = env_conditions['wave_height']['mean']
        avg_wind_speed = env_conditions['wind_speed']['mean']
        avg_sea_state = min(6, int(avg_wave_height * 2))  # Approximate sea state
        
        resistance_penalty = self.calculate_resistance_penalty(
            avg_wave_height, avg_wind_speed, avg_sea_state
        )
        
        # Calculate performance impacts
        fuel_increase = (resistance_penalty - 1) * 0.8  # 80% of resistance affects fuel
        speed_reduction = (resistance_penalty - 1) * 0.3  # 30% affects speed
        
        # Regulatory compliance impact
        cii_impact = max(0, current_cii - regulations['cii_regulation']['limit'])
        compliance_penalty = cii_impact * regulations['cii_regulation']['limit'] * 1000  # USD/day
        
        # Projected performance metrics
        performance_impact = {
            'target_year': target_year,
            'environmental_conditions': env_conditions,
            'regulatory_framework': regulations,
            'resistance_penalty': resistance_penalty,
            'fuel_consumption_increase': fuel_increase,
            'speed_reduction': speed_reduction,
            'cii_compliance_penalty': compliance_penalty,
            'annual_additional_costs': {
                'fuel_cost': fuel_increase * current_fuel * 365 * 600,  # USD/year
                'compliance_cost': compliance_penalty * 365 if compliance_penalty > 0 else 0,
                'total_additional_cost': fuel_increase * current_fuel * 365 * 600 + 
                                       (compliance_penalty * 365 if compliance_penalty > 0 else 0)
            },
            'operational_restrictions': {
                'speed_limit': max(0, current_speed * (1 - speed_reduction)),
                'weather_routing_required': avg_wave_height > 3.5,
                'port_access_restrictions': current_cii > regulations['cii_regulation']['limit']
            }
        }
        
        return performance_impact
    
    def generate_temporal_analysis(self, vessel_data: Dict, 
                                 start_year: int = 2025, 
                                 end_year: int = 2050) -> Dict:
        """
        Generate complete temporal analysis for the vessel
        
        Args:
            vessel_data: Current vessel characteristics
            start_year: Start year for analysis
            end_year: End year for analysis
        
        Returns:
            Dictionary with temporal analysis results
        """
        analysis_years = list(range(start_year, end_year + 1, 5))  # Every 5 years
        
        temporal_analysis = {
            'vessel_data': vessel_data,
            'analysis_period': f"{start_year}-{end_year}",
            'yearly_projections': {},
            'trends': {},
            'critical_years': []
        }
        
        # Collect yearly projections
        for year in analysis_years:
            temporal_analysis['yearly_projections'][year] = (
                self.project_vessel_performance_impact(vessel_data, year)
            )
        
        # Calculate trends
        years = list(temporal_analysis['yearly_projections'].keys())
        fuel_trends = []
        cost_trends = []
        
        for year in years:
            proj = temporal_analysis['yearly_projections'][year]
            fuel_trends.append(proj['fuel_consumption_increase'])
            cost_trends.append(proj['annual_additional_costs']['total_additional_cost'])
        
        # Linear trend analysis
        if len(years) > 1:
            fuel_slope, _, fuel_r, fuel_p, _ = stats.linregress(years, fuel_trends)
            cost_slope, _, cost_r, cost_p, _ = stats.linregress(years, cost_trends)
            
            temporal_analysis['trends'] = {
                'fuel_increase_rate': fuel_slope,
                'cost_increase_rate': cost_slope,
                'fuel_trend_strength': abs(fuel_r),
                'cost_trend_strength': abs(cost_r),
                'fuel_trend_significance': fuel_p < 0.05,
                'cost_trend_significance': cost_p < 0.05
            }
        
        # Identify critical years (threshold crossings)
        current_cii = vessel_data.get('cii_score', 4.2)
        
        for year, projection in temporal_analysis['yearly_projections'].items():
            cii_limit = projection['regulatory_framework']['cii_regulation']['limit']
            
            if current_cii > cii_limit:
                temporal_analysis['critical_years'].append({
                    'year': year,
                    'reason': 'CII compliance failure',
                    'impact': current_cii - cii_limit
                })
            
            if projection['annual_additional_costs']['total_additional_cost'] > 500000:  # USD 500k
                temporal_analysis['critical_years'].append({
                    'year': year,
                    'reason': 'Excessive operational costs',
                    'impact': projection['annual_additional_costs']['total_additional_cost']
                })
        
        return temporal_analysis
    
    def calculate_climate_risk_assessment(self, vessel_data: Dict) -> Dict:
        """
        Calculate comprehensive climate risk assessment
        
        Args:
            vessel_data: Current vessel characteristics
        
        Returns:
            Dictionary with risk assessment results
        """
        temporal_analysis = self.generate_temporal_analysis(vessel_data)
        
        # Risk factors
        physical_risk = 0
        transition_risk = 0
        regulatory_risk = 0
        
        # Physical risk (environmental conditions)
        final_year = max(temporal_analysis['yearly_projections'].keys())
        final_conditions = temporal_analysis['yearly_projections'][final_year]
        
        wave_increase = final_conditions['environmental_conditions']['wave_height']['mean'] / 2.0
        wind_increase = final_conditions['environmental_conditions']['wind_speed']['mean'] / 12.0
        
        physical_risk = min(1.0, (wave_increase + wind_increase) / 2)
        
        # Transition risk (cost increases)
        base_cost = 100000  # USD base annual cost
        final_cost = final_conditions['annual_additional_costs']['total_additional_cost']
        
        transition_risk = min(1.0, final_cost / (base_cost * 5))
        
        # Regulatory risk (compliance failures)
        compliance_failures = len(temporal_analysis['critical_years'])
        regulatory_risk = min(1.0, compliance_failures / 10)
        
        # Overall risk score
        overall_risk = (physical_risk * 0.3 + transition_risk * 0.4 + regulatory_risk * 0.3)
        
        risk_assessment = {
            'overall_risk_score': overall_risk,
            'risk_category': self.categorize_risk(overall_risk),
            'physical_risk': physical_risk,
            'transition_risk': transition_risk,
            'regulatory_risk': regulatory_risk,
            'critical_timeline': temporal_analysis['critical_years'],
            'adaptation_measures': self.generate_adaptation_measures(overall_risk),
            'temporal_analysis': temporal_analysis
        }
        
        return risk_assessment
    
    def categorize_risk(self, risk_score: float) -> str:
        """
        Categorize risk level
        
        Args:
            risk_score: Risk score (0-1)
        
        Returns:
            Risk category
        """
        if risk_score < 0.2:
            return "Low Risk"
        elif risk_score < 0.4:
            return "Medium-Low Risk"
        elif risk_score < 0.6:
            return "Medium Risk"
        elif risk_score < 0.8:
            return "High Risk"
        else:
            return "Very High Risk"
    
    def generate_adaptation_measures(self, risk_score: float) -> List[str]:
        """
        Generate recommended adaptation measures
        
        Args:
            risk_score: Risk score (0-1)
        
        Returns:
            List of recommended measures
        """
        measures = []
        
        if risk_score > 0.3:
            measures.append("Implement weather routing systems")
            measures.append("Upgrade vessel monitoring systems")
        
        if risk_score > 0.5:
            measures.append("Consider operational speed optimization")
            measures.append("Invest in crew training for extreme weather")
        
        if risk_score > 0.7:
            measures.append("Evaluate vessel retrofit options")
            measures.append("Review insurance coverage")
            measures.append("Develop emergency response protocols")
        
        return measures


# Example usage
if __name__ == "__main__":
    # Create Climate Guardian
    guardian = ClimateGuardian()
    
    # Example vessel data
    vessel_data = {
        'dwt': 5000,
        'age': 15,
        'fuel_consumption': 18,
        'co2_emission': 55,
        'cii_score': 4.2,
        'speed': 12
    }
    
    # Generate temporal analysis
    print("=== Climate Guardian Analysis ===\n")
    
    temporal_analysis = guardian.generate_temporal_analysis(vessel_data)
    
    print("Yearly Projections:")
    for year, projection in temporal_analysis['yearly_projections'].items():
        print(f"  {year}:")
        print(f"    Wave Height: {projection['environmental_conditions']['wave_height']['mean']:.2f}m")
        print(f"    Resistance Penalty: {projection['resistance_penalty']:.3f}")
        print(f"    Additional Costs: ${projection['annual_additional_costs']['total_additional_cost']:,.0f}/year")
        print()
    
    # Risk assessment
    risk_assessment = guardian.calculate_climate_risk_assessment(vessel_data)
    
    print(f"Risk Assessment:")
    print(f"  Overall Risk: {risk_assessment['risk_category']} ({risk_assessment['overall_risk_score']:.2f})")
    print(f"  Physical Risk: {risk_assessment['physical_risk']:.2f}")
    print(f"  Transition Risk: {risk_assessment['transition_risk']:.2f}")
    print(f"  Regulatory Risk: {risk_assessment['regulatory_risk']:.2f}")
    print()
    
    print("Recommended Adaptation Measures:")
    for measure in risk_assessment['adaptation_measures']:
        print(f"  - {measure}")
