"""
Asset Manager - User Interface and Data Validation Layer
========================================================

Purpose:
- Manage user inputs and data validation
- Handle data preprocessing and statistical regression for missing data
- Coordinate between UI and other agents
- Provide data export/import functionality
"""

import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import json
import csv
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import os


class AssetManager:
    """
    Asset Manager for data handling, validation, and UI coordination
    
    Manages:
    - User input validation and preprocessing
    - Missing data imputation using statistical regression
    - Data export/import functionality
    - Coordination between UI and agent systems
    """
    
    def __init__(self):
        self.vessel_data = {}
        self.data_history = []
        self.validation_rules = self.initialize_validation_rules()
        self.imputation_models = {}
        self.is_data_complete = False
        
        # Data templates for different vessel types
        self.vessel_templates = {
            'koster_coaster': {
                'dwt': 5000,
                'length': 100,
                'breadth': 16,
                'draft': 6.5,
                'speed': 12,
                'age': 15,
                'engine_power': 3000,
                'fuel_type': 'HFO',
                'engine_efficiency': 0.42
            },
            'general_cargo': {
                'dwt': 8000,
                'length': 120,
                'breadth': 18,
                'draft': 7.5,
                'speed': 14,
                'age': 12,
                'engine_power': 4500,
                'fuel_type': 'MDO',
                'engine_efficiency': 0.45
            },
            'bulk_carrier': {
                'dwt': 35000,
                'length': 180,
                'breadth': 30,
                'draft': 11.0,
                'speed': 13,
                'age': 8,
                'engine_power': 12000,
                'fuel_type': 'HFO',
                'engine_efficiency': 0.48
            }
        }
    
    def initialize_validation_rules(self) -> Dict:
        """
        Initialize data validation rules
        
        Returns:
            Dictionary with validation rules
        """
        rules = {
            'dwt': {
                'type': float,
                'min': 1000,
                'max': 100000,
                'required': True,
                'description': 'Deadweight Tonnage'
            },
            'length': {
                'type': float,
                'min': 50,
                'max': 300,
                'required': True,
                'description': 'Length Overall (meters)'
            },
            'breadth': {
                'type': float,
                'min': 10,
                'max': 50,
                'required': True,
                'description': 'Breadth (meters)'
            },
            'draft': {
                'type': float,
                'min': 3,
                'max': 20,
                'required': True,
                'description': 'Draft (meters)'
            },
            'speed': {
                'type': float,
                'min': 8,
                'max': 25,
                'required': True,
                'description': 'Service Speed (knots)'
            },
            'age': {
                'type': int,
                'min': 0,
                'max': 40,
                'required': True,
                'description': 'Vessel Age (years)'
            },
            'fuel_consumption': {
                'type': float,
                'min': 5,
                'max': 100,
                'required': False,
                'description': 'Fuel Consumption (tons/day)'
            },
            'co2_emission': {
                'type': float,
                'min': 10,
                'max': 300,
                'required': False,
                'description': 'CO2 Emissions (tons/day)'
            },
            'cii_score': {
                'type': float,
                'min': 1,
                'max': 10,
                'required': False,
                'description': 'Carbon Intensity Indicator'
            },
            'eedi_score': {
                'type': float,
                'min': 5,
                'max': 50,
                'required': False,
                'description': 'Energy Efficiency Design Index'
            }
        }
        
        return rules
    
    def validate_input(self, field_name: str, value: Any) -> Tuple[bool, str]:
        """
        Validate a single input field
        
        Args:
            field_name: Name of the field to validate
            value: Value to validate
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        if field_name not in self.validation_rules:
            return False, f"Unknown field: {field_name}"
        
        rule = self.validation_rules[field_name]
        
        # Required field check
        if rule['required'] and (value is None or value == ''):
            return False, f"{rule['description']} is required"
        
        # Optional empty field
        if not rule['required'] and (value is None or value == ''):
            return True, ""
        
        # Type conversion and validation
        try:
            if rule['type'] == float:
                value = float(value)
            elif rule['type'] == int:
                value = int(value)
        except (ValueError, TypeError):
            return False, f"{rule['description']} must be a number"
        
        # Range validation
        if 'min' in rule and value < rule['min']:
            return False, f"{rule['description']} must be >= {rule['min']}"
        
        if 'max' in rule and value > rule['max']:
            return False, f"{rule['description']} must be <= {rule['max']}"
        
        return True, ""
    
    def validate_all_inputs(self, data: Dict) -> Tuple[bool, Dict]:
        """
        Validate all input data
        
        Args:
            data: Dictionary with input data
        
        Returns:
            Tuple of (all_valid, errors_dict)
        """
        all_valid = True
        errors = {}
        
        for field_name, value in data.items():
            if field_name in self.validation_rules:
                is_valid, error_msg = self.validate_input(field_name, value)
                if not is_valid:
                    all_valid = False
                    errors[field_name] = error_msg
        
        # Check required fields
        for field_name, rule in self.validation_rules.items():
            if rule['required'] and field_name not in data:
                all_valid = False
                errors[field_name] = f"{rule['description']} is required"
        
        return all_valid, errors
    
    def impute_missing_data(self, vessel_data: Dict) -> Dict:
        """
        Impute missing data using statistical regression models
        
        Args:
            vessel_data: Dictionary with vessel data (may contain missing values)
        
        Returns:
            Dictionary with complete data
        """
        # Create feature matrix for imputation
        complete_data = vessel_data.copy()
        
        # Define relationships for imputation
        # Fuel consumption can be estimated from DWT, speed, and age
        if 'fuel_consumption' not in complete_data or complete_data['fuel_consumption'] is None:
            dwt = complete_data.get('dwt', 5000)
            speed = complete_data.get('speed', 12)
            age = complete_data.get('age', 15)
            
            # Simplified model: fuel consumption based on vessel characteristics
            base_consumption = (dwt / 1000) * (speed / 10) ** 2 * 0.8
            age_factor = 1 + (age / 100)  # Age increases consumption
            
            complete_data['fuel_consumption'] = base_consumption * age_factor
        
        # CO2 emissions based on fuel consumption
        if 'co2_emission' not in complete_data or complete_data['co2_emission'] is None:
            fuel_consumption = complete_data['fuel_consumption']
            fuel_type = complete_data.get('fuel_type', 'HFO')
            
            # Emission factors
            emission_factors = {'HFO': 3.1, 'MDO': 3.2, 'LNG': 2.75}
            factor = emission_factors.get(fuel_type, 3.1)
            
            complete_data['co2_emission'] = fuel_consumption * factor
        
        # CII score based on fuel consumption and DWT
        if 'cii_score' not in complete_data or complete_data['cii_score'] is None:
            fuel_consumption = complete_data['fuel_consumption']
            dwt = complete_data.get('dwt', 5000)
            
            # Simplified CII calculation
            reference_consumption = (dwt / 1000) * 8
            complete_data['cii_score'] = (fuel_consumption / reference_consumption) * 3.0
        
        # EEDI score based on vessel efficiency
        if 'eedi_score' not in complete_data or complete_data['eedi_score'] is None:
            co2_emission = complete_data['co2_emission']
            dwt = complete_data.get('dwt', 5000)
            speed = complete_data.get('speed', 12)
            
            # Simplified EEDI calculation
            complete_data['eedi_score'] = (co2_emission / (dwt * speed)) * 2000
        
        return complete_data
    
    def load_vessel_template(self, template_name: str) -> Dict:
        """
        Load vessel data from template
        
        Args:
            template_name: Name of the template
        
        Returns:
            Dictionary with vessel data
        """
        if template_name not in self.vessel_templates:
            raise ValueError(f"Unknown template: {template_name}")
        
        template_data = self.vessel_templates[template_name].copy()
        template_data['template_name'] = template_name
        
        return template_data
    
    def export_data(self, data: Dict, filepath: str, format: str = 'json'):
        """
        Export vessel data to file
        
        Args:
            data: Data to export
            filepath: Path to export file
            format: Export format ('json' or 'csv')
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        if format == 'json':
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        
        elif format == 'csv':
            # Convert to DataFrame for CSV export
            df = pd.DataFrame([data])
            df.to_csv(filepath, index=False)
        
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def import_data(self, filepath: str) -> Dict:
        """
        Import vessel data from file
        
        Args:
            filepath: Path to import file
        
        Returns:
            Dictionary with imported data
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        
        if filepath.endswith('.json'):
            with open(filepath, 'r') as f:
                data = json.load(f)
        
        elif filepath.endswith('.csv'):
            df = pd.read_csv(filepath)
            if len(df) > 0:
                data = df.iloc[0].to_dict()
            else:
                data = {}
        
        else:
            raise ValueError(f"Unsupported file format: {filepath}")
        
        return data
    
    def create_data_summary(self, data: Dict) -> Dict:
        """
        Create summary of vessel data
        
        Args:
            data: Vessel data dictionary
        
        Returns:
            Summary dictionary
        """
        summary = {
            'vessel_type': data.get('vessel_type', 'Unknown'),
            'dwt': data.get('dwt', 0),
            'age': data.get('age', 0),
            'dimensions': {
                'length': data.get('length', 0),
                'breadth': data.get('breadth', 0),
                'draft': data.get('draft', 0)
            },
            'performance': {
                'fuel_consumption': data.get('fuel_consumption', 0),
                'co2_emission': data.get('co2_emission', 0),
                'cii_score': data.get('cii_score', 0),
                'eedi_score': data.get('eedi_score', 0)
            },
            'completeness': self.calculate_data_completeness(data),
            'data_quality_score': self.assess_data_quality(data)
        }
        
        return summary
    
    def calculate_data_completeness(self, data: Dict) -> float:
        """
        Calculate data completeness percentage
        
        Args:
            data: Vessel data dictionary
        
        Returns:
            Completeness percentage (0-100)
        """
        required_fields = [field for field, rule in self.validation_rules.items() 
                          if rule['required']]
        optional_fields = [field for field, rule in self.validation_rules.items() 
                          if not rule['required']]
        
        # Check required fields
        required_complete = sum(1 for field in required_fields 
                               if field in data and data[field] is not None)
        
        # Check optional fields
        optional_complete = sum(1 for field in optional_fields 
                               if field in data and data[field] is not None)
        
        # Weighted completeness (required fields weighted more)
        required_score = (required_complete / len(required_fields)) * 70 if required_fields else 70
        optional_score = (optional_complete / len(optional_fields)) * 30 if optional_fields else 0
        
        return min(100, required_score + optional_score)
    
    def assess_data_quality(self, data: Dict) -> float:
        """
        Assess data quality based on consistency and reasonableness
        
        Args:
            data: Vessel data dictionary
        
        Returns:
            Quality score (0-100)
        
        """
        quality_checks = []
        
        # Check dimensional consistency
        if all(key in data for key in ['length', 'breadth', 'draft']):
            length, breadth, draft = data['length'], data['breadth'], data['draft']
            
            # Reasonable ratios
            if length / breadth > 3 and length / breadth < 10:
                quality_checks.append(1.0)
            else:
                quality_checks.append(0.5)
            
            if draft / breadth > 0.2 and draft / breadth < 0.6:
                quality_checks.append(1.0)
            else:
                quality_checks.append(0.5)
        
        # Check performance consistency
        if all(key in data for key in ['fuel_consumption', 'dwt', 'speed']):
            fuel, dwt, speed = data['fuel_consumption'], data['dwt'], data['speed']
            
            # Fuel consumption per DWT per speed (simplified check)
            efficiency_metric = fuel / (dwt / 1000) / (speed / 10)
            
            if 0.5 < efficiency_metric < 3.0:
                quality_checks.append(1.0)
            else:
                quality_checks.append(0.3)
        
        # Check age reasonableness
        if 'age' in data:
            age = data['age']
            if 0 <= age <= 40:
                quality_checks.append(1.0)
            else:
                quality_checks.append(0.2)
        
        # Calculate overall quality score
        if quality_checks:
            return np.mean(quality_checks) * 100
        else:
            return 50  # Default score if no checks can be performed
    
    def add_to_history(self, action: str, data: Dict, timestamp: Optional[datetime] = None):
        """
        Add entry to data history
        
        Args:
            action: Action performed
            data: Data associated with action
            timestamp: Timestamp (defaults to now)
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        history_entry = {
            'timestamp': timestamp.isoformat(),
            'action': action,
            'data_summary': self.create_data_summary(data)
        }
        
        self.data_history.append(history_entry)
    
    def get_data_history(self) -> List[Dict]:
        """
        Get data modification history
        
        Returns:
            List of history entries
        """
        return self.data_history.copy()
    
    def clear_history(self):
        """Clear data history"""
        self.data_history = []
    
    def get_validation_errors(self, data: Dict) -> Dict:
        """
        Get all validation errors for data
        
        Args:
            data: Data to validate
        
        Returns:
            Dictionary with field names as keys and error messages as values
        """
        _, errors = self.validate_all_inputs(data)
        return errors


# Example usage
if __name__ == "__main__":
    # Create Asset Manager
    manager = AssetManager()
    
    # Test with incomplete data
    incomplete_data = {
        'dwt': 5000,
        'length': 100,
        'breadth': 16,
        'draft': 6.5,
        'speed': 12,
        'age': 15
        # Missing fuel_consumption, co2_emission, cii_score, eedi_score
    }
    
    print("=== Asset Manager Demo ===\n")
    
    print("Original Data:")
    for key, value in incomplete_data.items():
        print(f"  {key}: {value}")
    
    print(f"\nData Completeness: {manager.calculate_data_completeness(incomplete_data):.1f}%")
    print(f"Data Quality Score: {manager.assess_data_quality(incomplete_data):.1f}/100")
    
    # Impute missing data
    complete_data = manager.impute_missing_data(incomplete_data)
    
    print("\nAfter Imputation:")
    for key, value in complete_data.items():
        print(f"  {key}: {value}")
    
    print(f"\nData Completeness: {manager.calculate_data_completeness(complete_data):.1f}%")
    print(f"Data Quality Score: {manager.assess_data_quality(complete_data):.1f}/100")
    
    # Create summary
    summary = manager.create_data_summary(complete_data)
    print(f"\nData Summary:")
    print(f"  Vessel Type: {summary['vessel_type']}")
    print(f"  DWT: {summary['dwt']}")
    print(f"  Age: {summary['age']} years")
    print(f"  Fuel Consumption: {summary['performance']['fuel_consumption']:.2f} tons/day")
    print(f"  CO2 Emission: {summary['performance']['co2_emission']:.2f} tons/day")
