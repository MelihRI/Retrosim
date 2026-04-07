"""
Asset Manager - Vessel Data Validation & Intelligence Layer
============================================================

Purpose (Per SmartCAPEX AI Architecture):
- "Kullanıcı ile sistem arasındaki köprü" - Bridge between user and agent system
- Manage vessel data validation and preprocessing
- Statistical regression for missing geometry data (L/B/T ratios)
- Provide vessel templates for quick setup
- Data export/import functionality (JSON/CSV)
- Data quality scoring and completeness metrics

Integration Points:
- SettingsManager: Provides validation and auto-completion
- Predictor Agent (EANN): Receives clean, validated vessel data
- Investment Strategist: Provides consistent data for NPV calculations
"""

import numpy as np
import pandas as pd
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict, is_dataclass
import os

# PyQt6 Signal Support (Optional - for UI integration)
try:
    from PyQt6.QtCore import QObject, pyqtSignal
    HAS_PYQT = True
except ImportError:
    HAS_PYQT = False
    # Fallback for non-GUI usage
    class QObject:
        pass
    def pyqtSignal(*args):
        return None


class AssetManager:
    """
    Asset Manager - "Varlık Yöneticisi" Agent
    ==========================================
    
    Role in SmartCAPEX AI Architecture:
    - Acts as the bridge between USER and Agent System
    - Validates vessel inputs before passing to Predictor Agent (EANN)
    - Completes missing geometry data using statistical regression
    - Provides vessel templates for quick project setup
    - Manages data export/import for project persistence
    
    Key Methods:
    - validate_vessel_data(): Validate VesselData dataclass
    - apply_template(): Load preset vessel configurations
    - impute_missing_data(): Auto-fill missing fields
    - get_data_quality_report(): Returns completeness & quality metrics
    """
    
    # PyQt Signals (if available)
    if HAS_PYQT:
        validation_changed = pyqtSignal(bool, dict)  # is_valid, errors
        data_quality_changed = pyqtSignal(float, float)  # quality, completeness
    
    def __init__(self):
        self.vessel_data = {}
        self.data_history = []
        self.validation_rules = self.initialize_validation_rules()
        self.imputation_models = {}
        self.is_data_complete = False
        self.last_validation_errors = {}
        
        # Vessel templates matching VesselData dataclass structure in main_window.py
        # These provide realistic presets for common Turkish Coaster fleet vessels
        self.vessel_templates = {
            'koster_coaster': {
                # Identity
                'name': 'M/V Coaster Demo',
                'type': 'Coaster',
                # Physical Dimensions
                'dwt': 5000,
                'loa': 95.0,
                'lbp': 90.0,
                'beam': 15.5,
                'draft': 6.2,
                'depth': 8.5,
                # Coefficients
                'cb': 0.78,
                'cp': 0.80,
                'cm': 0.97,
                # Hull Details
                'bow_height': 3.5,
                'stern_height': 2.0,
                'bulb_length': 4.0,
                'bulb_radius': 1.8,
                'stern_shape': 0.7,
                # Propulsion
                'prop_dia': 4.2,
                'prop_blades': 4,
                'rudder_h': 4.5,
                'speed': 11.5,
                'engine_power': 2800,
                'sfoc': 185.0,
                # Economics
                'opex': 3500,
                'value': 8000000,
                'cii': 'D',
                'eedi': 18.5,
                'age': 18,
            },
            'general_cargo': {
                'name': 'M/V General Cargo',
                'type': 'General Cargo',
                'dwt': 12000,
                'loa': 135.0,
                'lbp': 128.0,
                'beam': 20.5,
                'draft': 8.2,
                'depth': 11.5,
                'cb': 0.75,
                'cp': 0.78,
                'cm': 0.97,
                'bow_height': 4.0,
                'stern_height': 2.5,
                'bulb_length': 5.5,
                'bulb_radius': 2.2,
                'stern_shape': 0.75,
                'prop_dia': 5.5,
                'prop_blades': 4,
                'rudder_h': 5.5,
                'speed': 13.5,
                'engine_power': 5500,
                'sfoc': 175.0,
                'opex': 4200,
                'value': 12000000,
                'cii': 'C',
                'eedi': 14.2,
                'age': 12,
            },
            'bulk_carrier': {
                'name': 'M/V Bulk Carrier',
                'type': 'Bulk Carrier',
                'dwt': 55000,
                'loa': 190.0,
                'lbp': 182.0,
                'beam': 32.2,
                'draft': 12.5,
                'depth': 18.0,
                'cb': 0.82,
                'cp': 0.84,
                'cm': 0.98,
                'bow_height': 4.5,
                'stern_height': 3.0,
                'bulb_length': 7.0,
                'bulb_radius': 3.0,
                'stern_shape': 0.8,
                'prop_dia': 6.5,
                'prop_blades': 4,
                'rudder_h': 7.0,
                'speed': 14.0,
                'engine_power': 9500,
                'sfoc': 168.0,
                'opex': 5500,
                'value': 22000000,
                'cii': 'C',
                'eedi': 11.8,
                'age': 8,
            },
            'tanker': {
                'name': 'M/V Aframax Tanker',
                'type': 'Tanker',
                'dwt': 105000,
                'loa': 244.0,
                'lbp': 234.0,
                'beam': 42.0,
                'draft': 14.9,
                'depth': 21.0,
                'cb': 0.84,
                'cp': 0.86,
                'cm': 0.98,
                'bow_height': 5.0,
                'stern_height': 3.5,
                'bulb_length': 9.0,
                'bulb_radius': 4.0,
                'stern_shape': 0.82,
                'prop_dia': 7.8,
                'prop_blades': 5,
                'rudder_h': 8.5,
                'speed': 14.5,
                'engine_power': 15000,
                'sfoc': 162.0,
                'opex': 7500,
                'value': 45000000,
                'cii': 'B',
                'eedi': 9.5,
                'age': 5,
            },
            'container_feeder': {
                'name': 'M/V Container Feeder',
                'type': 'Container',
                'dwt': 18000,
                'loa': 170.0,
                'lbp': 160.0,
                'beam': 27.5,
                'draft': 9.5,
                'depth': 14.0,
                'cb': 0.65,
                'cp': 0.70,
                'cm': 0.96,
                'bow_height': 4.5,
                'stern_height': 3.0,
                'bulb_length': 6.0,
                'bulb_radius': 2.5,
                'stern_shape': 0.7,
                'prop_dia': 5.8,
                'prop_blades': 4,
                'rudder_h': 6.0,
                'speed': 17.0,
                'engine_power': 12000,
                'sfoc': 172.0,
                'opex': 5000,
                'value': 28000000,
                'cii': 'C',
                'eedi': 16.5,
                'age': 10,
            }
        }
    
    def get_template_names(self) -> List[str]:
        """Return list of available template names for UI dropdown"""
        return list(self.vessel_templates.keys())
    
    def get_template_display_names(self) -> Dict[str, str]:
        """Return template key -> display name mapping for UI"""
        return {
            'koster_coaster': '🚢 Koster / Coaster (5K DWT)',
            'general_cargo': '📦 General Cargo (12K DWT)',
            'bulk_carrier': '🏗️ Bulk Carrier (55K DWT)',
            'tanker': '🛢️ Tanker (105K DWT)',
            'container_feeder': '📦 Container Feeder (18K DWT)'
        }
    
    def apply_template_to_dataclass(self, template_key: str, target_dataclass) -> Any:
        """
        Apply a vessel template to an existing VesselData dataclass instance.
        
        Args:
            template_key: Key of the template (e.g., 'koster_coaster')
            target_dataclass: VesselData dataclass instance to update
            
        Returns:
            Updated dataclass instance
        """
        if template_key not in self.vessel_templates:
            raise ValueError(f"Unknown template: {template_key}")
        
        template = self.vessel_templates[template_key]
        
        # Update dataclass fields that exist in the template
        for key, value in template.items():
            if hasattr(target_dataclass, key):
                setattr(target_dataclass, key, value)
        
        # Add to history
        self.add_to_history('template_applied', {'template': template_key})
        
        return target_dataclass
    
    def validate_dataclass(self, data) -> Tuple[bool, Dict[str, str]]:
        """
        Validate a VesselData dataclass instance.
        
        Args:
            data: VesselData dataclass instance or dict
            
        Returns:
            Tuple of (is_valid, errors_dict)
        """
        # Convert dataclass to dict if needed
        if is_dataclass(data):
            data_dict = asdict(data)
        else:
            data_dict = data
        
        return self.validate_all_inputs(data_dict)
    
    def initialize_validation_rules(self) -> Dict:
        """
        Initialize data validation rules matching VesselData dataclass.
        
        These rules ensure vessel parameters are within realistic naval architecture bounds.
        Based on typical ship design ratios and Turkish Coaster fleet specifications.
        
        Returns:
            Dictionary with validation rules for each VesselData field
        """
        rules = {
            # === IDENTITY ===
            'name': {
                'type': str,
                'required': True,
                'description': 'Gemi Adı'
            },
            'type': {
                'type': str,
                'required': True,
                'description': 'Gemi Tipi'
            },
            
            # === PHYSICAL DIMENSIONS ===
            'dwt': {
                'type': int,
                'min': 500,
                'max': 500000,
                'required': True,
                'description': 'Deadweight Tonnage (DWT)'
            },
            'loa': {
                'type': float,
                'min': 30,
                'max': 400,
                'required': True,
                'description': 'Length Overall (LOA, m)'
            },
            'lbp': {
                'type': float,
                'min': 25,
                'max': 390,
                'required': True,
                'description': 'Length Between Perpendiculars (LBP, m)'
            },
            'beam': {
                'type': float,
                'min': 5,
                'max': 70,
                'required': True,
                'description': 'Beam / Genişlik (m)'
            },
            'draft': {
                'type': float,
                'min': 2,
                'max': 25,
                'required': True,
                'description': 'Draft / Su Çekimi (T, m)'
            },
            'depth': {
                'type': float,
                'min': 3,
                'max': 35,
                'required': True,
                'description': 'Depth / Derinlik (D, m)'
            },
            
            # === FORM COEFFICIENTS ===
            'cb': {
                'type': float,
                'min': 0.40,
                'max': 0.90,
                'required': True,
                'description': 'Block Coefficient (Cb)'
            },
            'cp': {
                'type': float,
                'min': 0.50,
                'max': 0.95,
                'required': False,
                'description': 'Prismatic Coefficient (Cp)'
            },
            'cm': {
                'type': float,
                'min': 0.85,
                'max': 0.99,
                'required': False,
                'description': 'Midship Section Coefficient (Cm)'
            },
            
            # === HULL GEOMETRY ===
            'bow_height': {
                'type': float,
                'min': 1.0,
                'max': 15.0,
                'required': False,
                'description': 'Bow Height (m)'
            },
            'stern_height': {
                'type': float,
                'min': 0.5,
                'max': 10.0,
                'required': False,
                'description': 'Stern Height (m)'
            },
            'bulb_length': {
                'type': float,
                'min': 0.0,
                'max': 20.0,
                'required': False,
                'description': 'Bulbous Bow Length (m)'
            },
            'bulb_radius': {
                'type': float,
                'min': 0.0,
                'max': 8.0,
                'required': False,
                'description': 'Bulbous Bow Radius (m)'
            },
            'stern_shape': {
                'type': float,
                'min': 0.0,
                'max': 1.0,
                'required': False,
                'description': 'Stern Shape Factor (0-1)'
            },
            
            # === PROPULSION ===
            'prop_dia': {
                'type': float,
                'min': 1.0,
                'max': 12.0,
                'required': False,
                'description': 'Propeller Diameter (m)'
            },
            'prop_blades': {
                'type': int,
                'min': 3,
                'max': 7,
                'required': False,
                'description': 'Propeller Blade Count'
            },
            'rudder_h': {
                'type': float,
                'min': 1.0,
                'max': 15.0,
                'required': False,
                'description': 'Rudder Height (m)'
            },
            'speed': {
                'type': float,
                'min': 6,
                'max': 30,
                'required': True,
                'description': 'Design Speed (knots)'
            },
            'engine_power': {
                'type': int,
                'min': 500,
                'max': 100000,
                'required': True,
                'description': 'Engine Power (kW)'
            },
            'sfoc': {
                'type': float,
                'min': 140,
                'max': 220,
                'required': False,
                'description': 'Specific Fuel Oil Consumption (g/kWh)'
            },
            
            # === ECONOMICS ===
            'opex': {
                'type': int,
                'min': 1000,
                'max': 50000,
                'required': True,
                'description': 'Annual OPEX per Day ($)'
            },
            'value': {
                'type': int,
                'min': 100000,
                'max': 500000000,
                'required': True,
                'description': 'Vessel Market Value ($)'
            },
            'age': {
                'type': int,
                'min': 0,
                'max': 50,
                'required': True,
                'description': 'Vessel Age (years)'
            },
            
            # === REGULATORY ===
            'cii': {
                'type': str,
                'options': ['A', 'B', 'C', 'D', 'E'],
                'required': False,
                'description': 'CII Rating (A-E)'
            },
            'eedi': {
                'type': float,
                'min': 1.0,
                'max': 50.0,
                'required': False,
                'description': 'EEDI Value'
            },
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
    
    def impute_missing_data(self, vessel_data: Union[Dict, Any]) -> Dict:
        """
        Impute missing vessel data using naval architecture regression models.
        
        Uses empirical ship design ratios and correlations based on:
        - DWT-Length relationship
        - L/B, B/T, B/D typical ratios per ship type
        - Engine power from admiralty coefficient
        - SFOC from engine age/type
        
        Args:
            vessel_data: Dictionary or VesselData dataclass with vessel data
        
        Returns:
            Dictionary with complete data (all fields estimated)
        """
        # Convert dataclass to dict if needed
        if is_dataclass(vessel_data):
            complete_data = asdict(vessel_data)
        else:
            complete_data = vessel_data.copy()
        
        # Get key parameters (with defaults)
        dwt = complete_data.get('dwt', 5000)
        ship_type = complete_data.get('type', 'General Cargo')
        
        # === SHIP TYPE RATIOS ===
        # Based on Schneekluth & Bertram "Ship Design for Efficiency and Economy"
        type_ratios = {
            'Bulk Carrier': {'L_B': 6.0, 'B_T': 2.6, 'B_D': 1.8, 'cb_typical': 0.83},
            'Container': {'L_B': 6.5, 'B_T': 2.8, 'B_D': 1.9, 'cb_typical': 0.65},
            'Tanker': {'L_B': 5.8, 'B_T': 2.5, 'B_D': 1.7, 'cb_typical': 0.85},
            'General Cargo': {'L_B': 6.2, 'B_T': 2.7, 'B_D': 1.85, 'cb_typical': 0.75},
            'Coaster': {'L_B': 6.0, 'B_T': 2.5, 'B_D': 1.75, 'cb_typical': 0.78},
        }
        ratios = type_ratios.get(ship_type, type_ratios['General Cargo'])
        
        # === ESTIMATE MAIN DIMENSIONS FROM DWT ===
        # DWT ~ 0.65 * L * B * T * Cb for cargo ships
        if complete_data.get('loa') is None or complete_data.get('loa', 0) == 0:
            # Empirical: L ≈ 4.5 * DWT^0.33 for general cargo
            complete_data['loa'] = round(4.8 * (dwt ** 0.33), 1)
        
        loa = complete_data['loa']
        
        if complete_data.get('lbp') is None or complete_data.get('lbp', 0) == 0:
            complete_data['lbp'] = round(loa * 0.95, 1)  # LBP typically 95% of LOA
        
        if complete_data.get('beam') is None or complete_data.get('beam', 0) == 0:
            complete_data['beam'] = round(loa / ratios['L_B'], 1)
        
        beam = complete_data['beam']
        
        if complete_data.get('draft') is None or complete_data.get('draft', 0) == 0:
            complete_data['draft'] = round(beam / ratios['B_T'], 1)
        
        if complete_data.get('depth') is None or complete_data.get('depth', 0) == 0:
            complete_data['depth'] = round(beam / ratios['B_D'], 1)
        
        # === ESTIMATE FORM COEFFICIENTS ===
        if complete_data.get('cb') is None or complete_data.get('cb', 0) == 0:
            complete_data['cb'] = ratios['cb_typical']
        
        cb = complete_data['cb']
        
        if complete_data.get('cp') is None or complete_data.get('cp', 0) == 0:
            # Cp ≈ Cb + 0.02 to 0.06
            complete_data['cp'] = round(min(0.95, cb + 0.04), 2)
        
        if complete_data.get('cm') is None or complete_data.get('cm', 0) == 0:
            # Cm typically 0.96-0.99 for conventional ships
            complete_data['cm'] = round(0.97 + (cb - 0.7) * 0.03, 2)
        
        # === ESTIMATE HULL GEOMETRY ===
        if complete_data.get('bow_height') is None or complete_data.get('bow_height', 0) == 0:
            complete_data['bow_height'] = round(loa * 0.025, 1)  # ~2.5% of LOA
        
        if complete_data.get('stern_height') is None or complete_data.get('stern_height', 0) == 0:
            complete_data['stern_height'] = round(complete_data['bow_height'] * 0.6, 1)
        
        if complete_data.get('bulb_length') is None or complete_data.get('bulb_length', 0) == 0:
            complete_data['bulb_length'] = round(loa * 0.035, 1)  # Bulb ~3.5% of LOA
        
        if complete_data.get('bulb_radius') is None or complete_data.get('bulb_radius', 0) == 0:
            complete_data['bulb_radius'] = round(complete_data['draft'] * 0.25, 1)
        
        if complete_data.get('stern_shape') is None:
            complete_data['stern_shape'] = 0.75  # Default U-stern shape factor
        
        # === ESTIMATE PROPULSION ===
        draft = complete_data['draft']
        speed = complete_data.get('speed', 12)
        
        if complete_data.get('prop_dia') is None or complete_data.get('prop_dia', 0) == 0:
            # Propeller diameter typically 0.65-0.75 of draft
            complete_data['prop_dia'] = round(draft * 0.7, 1)
        
        if complete_data.get('prop_blades') is None or complete_data.get('prop_blades', 0) == 0:
            complete_data['prop_blades'] = 4  # Most common
        
        if complete_data.get('rudder_h') is None or complete_data.get('rudder_h', 0) == 0:
            complete_data['rudder_h'] = round(complete_data['prop_dia'] * 1.1, 1)
        
        if complete_data.get('engine_power') is None or complete_data.get('engine_power', 0) == 0:
            # Admiralty coefficient method: P = Δ^(2/3) * V^3 / C
            displacement = dwt * 1.15  # Approximate displacement from DWT
            admiralty_coeff = 400  # Typical for cargo ships
            complete_data['engine_power'] = int((displacement ** 0.67) * (speed ** 3) / admiralty_coeff)
        
        if complete_data.get('sfoc') is None or complete_data.get('sfoc', 0) == 0:
            age = complete_data.get('age', 10)
            # SFOC deteriorates ~0.5% per year
            base_sfoc = 168  # Modern 2-stroke diesel
            complete_data['sfoc'] = round(base_sfoc * (1 + age * 0.005), 1)
        
        # === ESTIMATE REGULATORY VALUES ===
        engine_power = complete_data['engine_power']
        sfoc = complete_data['sfoc']
        
        if complete_data.get('eedi') is None or complete_data.get('eedi', 0) == 0:
            # Simplified EEDI = (P * SFOC * CF) / (DWT * Vref)
            cf = 3.114  # CO2 emission factor for HFO
            complete_data['eedi'] = round((engine_power * sfoc * cf) / (dwt * speed * 1000), 2)
        
        if complete_data.get('cii') is None or complete_data.get('cii') == '':
            # Estimate CII rating based on EEDI
            eedi = complete_data['eedi']
            if eedi < 8:
                complete_data['cii'] = 'A'
            elif eedi < 12:
                complete_data['cii'] = 'B'
            elif eedi < 16:
                complete_data['cii'] = 'C'
            elif eedi < 20:
                complete_data['cii'] = 'D'
            else:
                complete_data['cii'] = 'E'
        
        return complete_data
    
    def impute_dataclass(self, data) -> Any:
        """
        Impute missing data directly on a VesselData dataclass instance.
        
        Args:
            data: VesselData dataclass instance
            
        Returns:
            Updated dataclass instance with imputed values
        """
        if not is_dataclass(data):
            raise ValueError("Input must be a dataclass instance")
        
        imputed_dict = self.impute_missing_data(data)
        
        # Update dataclass fields
        for key, value in imputed_dict.items():
            if hasattr(data, key):
                setattr(data, key, value)
        
        return data
    
    def get_data_quality_report(self, data: Union[Dict, Any]) -> Dict:
        """
        Generate a comprehensive data quality report.
        
        Args:
            data: VesselData dataclass or dict
            
        Returns:
            Report dict with quality score, completeness, and recommendations
        """
        if is_dataclass(data):
            data_dict = asdict(data)
        else:
            data_dict = data
        
        completeness = self.calculate_data_completeness(data_dict)
        quality = self.assess_data_quality(data_dict)
        is_valid, errors = self.validate_dataclass(data_dict)
        
        # Generate recommendations
        recommendations = []
        if completeness < 80:
            recommendations.append("💡 Eksik veriler var. 'Otomatik Tamamla' butonunu kullanın.")
        if quality < 70:
            recommendations.append("⚠️ Veri tutarsızlığı tespit edildi. L/B ve B/T oranlarını kontrol edin.")
        if not is_valid:
            recommendations.append(f"❌ {len(errors)} adet doğrulama hatası var.")
        if quality >= 90 and completeness >= 95:
            recommendations.append("✅ Veri kalitesi mükemmel. Analize hazır!")
        
        return {
            'completeness': round(completeness, 1),
            'quality_score': round(quality, 1),
            'is_valid': is_valid,
            'validation_errors': errors,
            'error_count': len(errors),
            'recommendations': recommendations,
            'ready_for_analysis': is_valid and completeness >= 80 and quality >= 70
        }
    
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
