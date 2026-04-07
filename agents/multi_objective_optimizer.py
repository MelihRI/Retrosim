"""
Multi-Objective Optimizer - TOPSIS + IPSO for Retrofit Decisions
=================================================================

Based on:
- TOPSIS: Multi-Criteria Decision Making (Li et al., 2025)
- IPSO: Improved Particle Swarm Optimization (Cui et al., 2008)
- Pareto Optimality: Trade-off analysis (Rosso et al., 2020)
- Maritime Decarbonization: ML-based FEP models (Nguyen et al., 2025)

Purpose: Analyze retrofit scenarios using hybrid MCDM and MOO approach

Integration with SmartCAPEX AI Agents:
- Predictor Agent: Uses EANN for fuel consumption prediction
- Investment Strategist: Provides NPV and DCF calculations
- Climate Guardian: Supplies climate penalty projections
- Asset Manager: Provides vessel data and parameters
"""

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from scipy.stats import norm
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import json
from sklearn.preprocessing import MinMaxScaler


@dataclass
class RetrofitScenario:
    """
    ÖĞRENİM: Dataclass kullanımı
    - Type hints ile data structure tanımla
    - Immutable data containers
    - Auto-generated methods (__init__, __repr__)
    """
    name: str
    capex: float  # Capital expenditure (USD)
    opex: float   # Operational expenditure (USD/year)
    fuel_consumption: float  # tons/day
    co2_emission: float      # tons/day
    cii_score: float         # Carbon Intensity Indicator
    eedi_score: float        # Energy Efficiency Design Index
    lifespan: int            # years
    retrofit_time: float     # months in shipyard
    risk_factor: float       # 0-1 scale


class TOPSIS:
    """
    ÖĞRENİM: TOPSIS (Technique for Order Preference by Similarity to Ideal Solution)
    
    TOPSIS Mantığı:
    1. Decision matrix normalize et
    2. Weighted normalized matrix oluştur
    3. Ideal ve negative-ideal solutions bul
    4. Her alternatifin uzaklıklarını hesapla
    5. Closeness coefficient ile sırala
    
    Referans: Li et al. (2025) - Multi-criteria decision making for maritime applications
    """
    
    def __init__(self):
        self.scaler = MinMaxScaler()
    
    def normalize_matrix(self, decision_matrix):
        """
        ÖĞRENİM: Vector normalization
        
        Formula: r_ij = x_ij / sqrt(sum(x_ij^2))
        
        Neden? Farklı birimlerdeki kriterleri karşılaştırılabilir hale getir
        """
        # Calculate column-wise norm
        norms = np.sqrt(np.sum(decision_matrix ** 2, axis=0))
        
        # Avoid division by zero
        norms[norms == 0] = 1
        
        normalized = decision_matrix / norms
        
        return normalized
    
    def apply_weights(self, norm_matrix, weights):
        """
        ÖĞRENİM: Ağırlıkları uygula
        
        Her kriterin önemini yansıt
        Örnek: Economic 40%, Environmental 35%, Operational 25%
        """
        return norm_matrix * weights
    
    def find_ideal_solutions(self, weighted_matrix, criteria_types):
        """
        ÖĞRENİM: Ideal ve Negative-ideal solutions
        
        - Benefit criteria: max değer (e.g., efficiency)
        - Cost criteria: min değer (e.g., fuel consumption)
        """
        ideal_best = []
        ideal_worst = []
        
        for i, ctype in enumerate(criteria_types):
            if ctype == 'max':  # Benefit criterion
                ideal_best.append(weighted_matrix[:, i].max())
                ideal_worst.append(weighted_matrix[:, i].min())
            else:  # Cost criterion ('min')
                ideal_best.append(weighted_matrix[:, i].min())
                ideal_worst.append(weighted_matrix[:, i].max())
        
        return np.array(ideal_best), np.array(ideal_worst)
    
    def calculate_distances(self, weighted_matrix, ideal_best, ideal_worst):
        """
        ÖĞRENİM: Euclidean distance hesaplama
        
        Distance to ideal = sqrt(sum((x_i - ideal_i)^2))
        """
        # Distance to positive ideal solution
        dist_to_best = np.sqrt(
            np.sum((weighted_matrix - ideal_best) ** 2, axis=1)
        )
        
        # Distance to negative ideal solution
        dist_to_worst = np.sqrt(
            np.sum((weighted_matrix - ideal_worst) ** 2, axis=1)
        )
        
        return dist_to_best, dist_to_worst
    
    def calculate_closeness_coefficient(self, dist_to_best, dist_to_worst):
        """
        ÖĞRENİM: Closeness coefficient (CC)
        
        CC = D- / (D+ + D-)
        
        D+ = distance to positive ideal
        D- = distance to negative ideal
        
        CC ∈ [0, 1], yüksek değer = ideal'e yakın
        """
        return dist_to_worst / (dist_to_best + dist_to_worst + 1e-10)
    
    def rank(self, alternatives_dict, criteria, weights):
        """
        ÖĞRENİM: Tam TOPSIS implementation
        
        Parameters:
        -----------
        alternatives_dict : dict
            {'Current': {'fuel': 25, 'npv': -500, 'cii': 60}, ...}
        criteria : dict
            {'fuel': 'min', 'npv': 'max', 'cii': 'max'}
        weights : list or array
            [0.35, 0.4, 0.25]
        
        Returns:
        --------
        dict with ranking, scores, and details
        """
        # Convert to matrix
        alt_names = list(alternatives_dict.keys())
        criteria_names = list(criteria.keys())
        
        matrix = np.array([
            [alternatives_dict[alt][crit] for crit in criteria_names]
            for alt in alt_names
        ])
        
        # Step 1: Normalize
        norm_matrix = self.normalize_matrix(matrix)
        
        # Step 2: Apply weights
        weights_array = np.array(weights)
        weighted_matrix = self.apply_weights(norm_matrix, weights_array)
        
        # Step 3: Find ideal solutions
        criteria_types = [criteria[c] for c in criteria_names]
        ideal_best, ideal_worst = self.find_ideal_solutions(
            weighted_matrix, criteria_types
        )
        
        # Step 4: Calculate distances
        dist_to_best, dist_to_worst = self.calculate_distances(
            weighted_matrix, ideal_best, ideal_worst
        )
        
        # Step 5: Calculate closeness coefficient
        closeness = self.calculate_closeness_coefficient(
            dist_to_best, dist_to_worst
        )
        
        # Step 6: Rank alternatives
        ranking_indices = np.argsort(closeness)[::-1]  # Descending order
        
        return {
            'ranking': [alt_names[i] for i in ranking_indices],
            'scores': {alt_names[i]: closeness[i] for i in range(len(alt_names))},
            'closeness_coefficients': closeness,
            'distances_to_ideal': dist_to_best,
            'distances_to_worst': dist_to_worst,
            'details': {
                'normalized_matrix': norm_matrix,
                'weighted_matrix': weighted_matrix,
                'ideal_best': ideal_best,
                'ideal_worst': ideal_worst,
                'criteria_names': criteria_names
            }
        }


class IPSO:
    """
    ÖĞRENİM: Improved Particle Swarm Optimization
    
    Improvements over standard PSO:
    1. Time-varying acceleration coefficients (Cui et al., 2008)
    2. Adaptive inertia weight
    3. Better exploration-exploitation balance
    
    c1(t) = c1_max - (c1_max - c1_min) * ((t/t_max)^2)  # Concave function
    c2(t) = 3.0 - c1(t)
    
    Referans: Cui et al. (2008) - Improved PSO algorithm with time-varying acceleration coefficients
    """
    
    def __init__(self, n_particles=30, n_iterations=100, 
                 c1_max=2.5, c1_min=0.5, w_max=0.9, w_min=0.4):
        """
        ÖĞRENİM: PSO Hyperparameters
        
        n_particles: Swarm size
        n_iterations: Maximum iterations
        c1, c2: Acceleration coefficients (cognitive and social)
        w: Inertia weight (exploration vs exploitation)
        """
        self.n_particles = n_particles
        self.n_iterations = n_iterations
        self.c1_max = c1_max
        self.c1_min = c1_min
        self.w_max = w_max
        self.w_min = w_min
        
        # Best positions
        self.gbest_position = None
        self.gbest_value = float('inf')
        self.convergence_curve = []
    
    def _update_acceleration_coefficients(self, iteration):
        """
        ÖĞRENİM: Time-varying acceleration coefficients
        
        Concave function for c1 (cognitive component):
        - Başta yüksek: particles kendi deneyimlerine güvenir
        - Sonda düşük: swarm bilgisine daha çok önem verir
        """
        t_ratio = iteration / self.n_iterations
        
        # Concave function (Cui et al., 2008)
        c1 = self.c1_max - (self.c1_max - self.c1_min) * (t_ratio ** 2)
        c2 = 3.0 - c1
        
        return c1, c2
    
    def _update_inertia_weight(self, iteration):
        """
        ÖĞRENİM: Linearly decreasing inertia weight
        
        w(t) = w_max - (w_max - w_min) * (t / t_max)
        
        Başta exploration (global search)
        Sonda exploitation (local search)
        """
        return self.w_max - (self.w_max - self.w_min) * (
            iteration / self.n_iterations
        )
    
    def optimize(self, objective_function, bounds, maximize=False):
        """
        ÖĞRENİM: Main optimization loop
        
        Parameters:
        -----------
        objective_function : callable
            Function to optimize
        bounds : list of tuples
            [(min, max), ...] for each dimension
        maximize : bool
            True for maximization, False for minimization
        
        Returns:
        --------
        Best position and value found
        """
        n_dims = len(bounds)
        
        # Initialize particles
        positions = np.random.uniform(
            low=[b[0] for b in bounds],
            high=[b[1] for b in bounds],
            size=(self.n_particles, n_dims)
        )
        
        velocities = np.random.uniform(
            low=-1, high=1,
            size=(self.n_particles, n_dims)
        )
        
        # Initialize personal best
        pbest_positions = positions.copy()
        pbest_values = np.array([
            objective_function(p) for p in positions
        ])
        
        # Initialize global best
        if maximize:
            best_idx = np.argmax(pbest_values)
        else:
            best_idx = np.argmin(pbest_values)
        
        self.gbest_position = pbest_positions[best_idx].copy()
        self.gbest_value = pbest_values[best_idx]
        
        # Main loop
        for iteration in range(self.n_iterations):
            # Update coefficients
            c1, c2 = self._update_acceleration_coefficients(iteration)
            w = self._update_inertia_weight(iteration)
            
            # Update velocities and positions
            r1 = np.random.rand(self.n_particles, n_dims)
            r2 = np.random.rand(self.n_particles, n_dims)
            
            velocities = (
                w * velocities +
                c1 * r1 * (pbest_positions - positions) +
                c2 * r2 * (self.gbest_position - positions)
            )
            
            positions = positions + velocities
            
            # Apply bounds
            for i in range(n_dims):
                positions[:, i] = np.clip(
                    positions[:, i], bounds[i][0], bounds[i][1]
                )
            
            # Evaluate fitness
            fitness_values = np.array([
                objective_function(p) for p in positions
            ])
            
            # Update personal best
            if maximize:
                improved = fitness_values > pbest_values
            else:
                improved = fitness_values < pbest_values
            
            pbest_positions[improved] = positions[improved]
            pbest_values[improved] = fitness_values[improved]
            
            # Update global best
            if maximize:
                current_best_idx = np.argmax(pbest_values)
            else:
                current_best_idx = np.argmin(pbest_values)
            
            if maximize:
                if pbest_values[current_best_idx] > self.gbest_value:
                    self.gbest_position = pbest_positions[current_best_idx].copy()
                    self.gbest_value = pbest_values[current_best_idx]
            else:
                if pbest_values[current_best_idx] < self.gbest_value:
                    self.gbest_position = pbest_positions[current_best_idx].copy()
                    self.gbest_value = pbest_values[current_best_idx]
            
            # Store convergence
            self.convergence_curve.append(self.gbest_value)
            
            # Print progress
            if (iteration + 1) % 10 == 0:
                print(f"IPSO Iteration {iteration + 1}/{self.n_iterations}, "
                      f"Best Value: {self.gbest_value:.4f}")
        
        return self.gbest_position, self.gbest_value


class MultiObjectiveOptimizer:
    """
    ÖĞRENİM: Hybrid Multi-Objective Optimizer
    
    Combines:
    1. EANN for performance prediction (from surrogate_modeler.py)
    2. TOPSIS for MCDM ranking
    3. IPSO for Pareto optimization
    
    Integration with SmartCAPEX AI Agent Architecture:
    - Communicates with Predictor Agent (EANN) for fuel predictions
    - Provides results to Investment Strategist for NPV calculations
    - Uses Climate Guardian's climate penalty factors
    """
    
    def __init__(self, surrogate_model=None):
        """
        ÖĞRENİM: Integration with EANN
        
        surrogate_model: Trained EANN model from surrogate_modeler.py
        """
        self.surrogate_model = surrogate_model
        self.scenarios = {}
        self.pareto_front = []
        self.optimization_results = {}
        self.topsis = TOPSIS()
        
        # Economic parameters
        self.discount_rate = 0.08
        self.fuel_price = 600  # USD per ton
        self.carbon_tax = 100  # USD per ton CO2
        self.inflation_rate = 0.03
        
        # Regulatory parameters
        self.cii_threshold = 3.5
        self.eedi_threshold = 15
        
        # Turkish coastal vessel penalties
        self.turkish_penalties = {
            'psc_detention': 50000,
            'environmental_fine': 25000,
            'port_ban_cost': 100000,
            'class_downgrade': 200000,
            'insurance_increase': 0.15
        }
        
        # --- Retrofit Component Database ---
        self.retrofit_components = {
            'propeller_high_eff': {'name': 'High-Efficiency Propeller', 'capex': 250000, 'saving': 0.04},
            'pbcf': {'name': 'PBCF (Propeller Boss Cap Fin)', 'capex': 60000, 'saving': 0.015},
            'bulbous_bow': {'name': 'Optimized Bulbous Bow', 'capex': 180000, 'saving': 0.035},
            'shaft_generator': {'name': 'Shaft Generator System', 'capex': 350000, 'saving': 0.05},
            'engine_derating': {'name': 'Engine De-rating / Tuning', 'capex': 120000, 'saving': 0.025},
            'hull_coating': {'name': 'Premium Anti-fouling Coating', 'capex': 80000, 'saving': 0.03},
            'flettner_rotor': {'name': 'Flettner Rotor (Wind)', 'capex': 850000, 'saving': 0.12}
        }
        
        self.current_vessel_data = None # Store latest vessel data for reporting/viz
    
    def create_base_scenarios(self, vessel_data: Dict):
        """
        ÖĞRENİM: Scenario generation
        
        Integration with EANN Predictor Agent for accurate performance prediction
        """
        dwt = vessel_data.get('dwt', 5000)
        age = vessel_data.get('age', 15)
        
        # Use EANN to predict current performance if model available
        if self.surrogate_model is not None:
            try:
                current_prediction = self.surrogate_model.predict(vessel_data)
                current_fuel = current_prediction.get('fuel_consumption', 15)
                current_co2 = current_prediction.get('co2_emission', 45)
                current_cii = current_prediction.get('cii_score', 4.2)
                current_eedi = current_prediction.get('eedi_score', 18)
            except Exception as e:
                print(f"Warning: EANN prediction failed: {e}. Using defaults.")
                current_fuel = vessel_data.get('fuel_consumption', 15)
                current_co2 = vessel_data.get('co2_emission', 45)
                current_cii = vessel_data.get('cii_score', 4.2)
                current_eedi = vessel_data.get('eedi_score', 18)
        else:
            # Fallback to defaults
            current_fuel = vessel_data.get('fuel_consumption', 15)
            current_co2 = vessel_data.get('co2_emission', 45)
            current_cii = vessel_data.get('cii_score', 4.2)
            current_eedi = vessel_data.get('eedi_score', 18)
        
        # Scenario 1: Current Status
        self.scenarios['current'] = RetrofitScenario(
            name="Current Operations",
            capex=0,
            opex=800000 + (age * 50000),
            fuel_consumption=current_fuel,
            co2_emission=current_co2,
            cii_score=current_cii,
            eedi_score=current_eedi,
            lifespan=max(25 - age, 5),
            retrofit_time=0,
            risk_factor=min(age / 25, 0.9)
        )
        
        # Scenario 2: Smart Retrofit (Calculated based on components)
        selected_components = vessel_data.get('selected_retrofit', [])
        total_capex = 0
        total_saving = 0
        
        if not selected_components:
            # Default fallback if no specific components selected
            retrofit_reduction = 0.20
            total_capex = dwt * 150
        else:
            for comp_id in selected_components:
                if comp_id in self.retrofit_components:
                    comp = self.retrofit_components[comp_id]
                    total_capex += comp['capex']
                    # Savings are multiplicative: (1-s1)*(1-s2)...
                    total_saving = 1 - (1 - total_saving) * (1 - comp['saving'])
            retrofit_reduction = total_saving

        self.scenarios['retrofit'] = RetrofitScenario(
            name="Smart Retrofit",
            capex=total_capex,
            opex=700000,
            fuel_consumption=current_fuel * (1 - retrofit_reduction),
            co2_emission=current_co2 * (1 - retrofit_reduction),
            cii_score=current_cii * (1 - retrofit_reduction),
            eedi_score=current_eedi * (1 - retrofit_reduction * 0.7),
            lifespan=20,
            retrofit_time=max(2, len(selected_components) * 1.5), # approx 1.5 months per major part
            risk_factor=0.25
        )
        
        # Scenario 3: New Build (45% improvement)
        newbuild_reduction = 0.45
        self.scenarios['newbuild'] = RetrofitScenario(
            name="New Build",
            capex=dwt * 800,
            opex=600000,
            fuel_consumption=current_fuel * (1 - newbuild_reduction),
            co2_emission=current_co2 * (1 - newbuild_reduction),
            cii_score=current_cii * (1 - newbuild_reduction),
            eedi_score=current_eedi * (1 - newbuild_reduction * 0.8),
            lifespan=25,
            retrofit_time=0,
            risk_factor=0.1
        )
    
    def calculate_npv(self, scenario: RetrofitScenario, 
                      vessel_data: Dict,
                      analysis_period: int = 15) -> float:
        """
        ÖĞRENİM: Net Present Value calculation - CLIMATE AWARE VERSION
        
        NPV = -CAPEX + Σ(Cash Flow_t / (1 + r)^t) + Residual Value
        
        This version integrates ClimateGuardian to update fuel consumption 
        each year based on projected sea state deterioration.
        """
        from agents.climate_guardian import ClimateGuardian
        guardian = ClimateGuardian()
        
        npv = -scenario.capex
        
        start_year = 2025
        
        for i in range(analysis_period):
            year = start_year + i
            
            # 1. Get Climate and Regulatory Projections for this year
            env_impact = guardian.project_vessel_performance_impact(
                vessel_data={'fuel_consumption': scenario.fuel_consumption, 
                             'speed': 12, 'cii_score': scenario.cii_score},
                target_year=year
            )
            
            # Apply resistance penalty to fuel consumption for this specific year
            resistance_factor = env_impact.get('resistance_penalty', 1.0)
            yearly_fuel_consumption = scenario.fuel_consumption * resistance_factor
            
            # 2. Get Regulatory Projections (Carbon Tax etc.)
            regulations = guardian.project_regulatory_changes(year)
            carbon_tax_rate = regulations['carbon_tax']['rate']
            cii_limit = regulations['cii_regulation']['limit']
            
            # 3. Calculate Annual costs
            annual_fuel_cost = yearly_fuel_consumption * 365 * self.fuel_price
            annual_carbon_cost = scenario.co2_emission * 365 * carbon_tax_rate
            annual_opex = scenario.opex * (1 + self.inflation_rate) ** i
            
            # 4. Calculate Penalties (Increased risk over time)
            penalty_cost = 0
            # Higher penalties if CII limit is exceeded
            if scenario.cii_score > cii_limit:
                severity = (scenario.cii_score - cii_limit) / cii_limit
                penalty_cost += self.turkish_penalties['environmental_fine'] * (1 + severity)
            
            # Increasing PSC risk with ship age
            ship_age_in_year = vessel_data.get('age', 10) + i
            if ship_age_in_year > 20: 
                penalty_cost += self.turkish_penalties['psc_detention'] * 0.5
            
            # Total annual cost (negative cash flow)
            total_annual_cost = (
                annual_fuel_cost + annual_carbon_cost + 
                annual_opex + penalty_cost
            )
            
            # Discount to present value
            pv_cost = total_annual_cost / (1 + self.discount_rate) ** i
            npv -= pv_cost
        
        # Residual value
        residual_value = scenario.capex * 0.1 * (scenario.lifespan / 25)
        npv += residual_value / (1 + self.discount_rate) ** analysis_period
        
        return npv
    
    def calculate_environmental_score(self, scenario: RetrofitScenario) -> float:
        """
        ÖĞRENİM: Environmental performance scoring
        
        Combines CII, EEDI, and CO2 emissions into a 0-100 scale
        """
        cii_component = max(0, (self.cii_threshold - scenario.cii_score) / 
                           self.cii_threshold * 40)
        eedi_component = max(0, (self.eedi_threshold - scenario.eedi_score) / 
                            self.eedi_threshold * 30)
        baseline_co2 = 50
        co2_component = max(0, (baseline_co2 - scenario.co2_emission) / 
                           baseline_co2 * 30)
        
        environmental_score = cii_component + eedi_component + co2_component
        return min(100, max(0, environmental_score))
    
    def calculate_operational_score(self, scenario: RetrofitScenario) -> float:
        """
        ÖĞRENİM: Operational efficiency scoring
        
        Considers fuel efficiency, risk, and availability
        """
        baseline_fuel = 20
        fuel_component = max(0, (baseline_fuel - scenario.fuel_consumption) / 
                            baseline_fuel * 40)
        risk_component = (1 - scenario.risk_factor) * 30
        availability_component = (1 - scenario.retrofit_time / 12) * 30
        
        operational_score = fuel_component + risk_component + availability_component
        return min(100, max(0, operational_score))
    
    def optimize_scenarios(self, vessel_data: Dict) -> Dict:
        """
        Create base scenarios and calculate KPIs to match frontend output format.
        """
        self.create_base_scenarios(vessel_data)
        alternatives = {}
        for name, scenario in self.scenarios.items():
            npv = self.calculate_npv(scenario, vessel_data=vessel_data)
            env_score = self.calculate_environmental_score(scenario)
            ops_score = self.calculate_operational_score(scenario)
            
            alternatives[name] = {
                'npv': npv,
                'environmental_score': env_score,
                'operational_score': ops_score
            }
        return alternatives

    def topsis_decision(self, vessel_data: Dict, weights: Dict[str, float] = None) -> Dict:
        """
        ÖĞRENİM: TOPSIS-based decision making
        
        Multi-Criteria Decision Making using TOPSIS algorithm
        """
        if weights is None:
            weights = {
                'economic': 0.4,
                'environmental': 0.35,
                'operational': 0.25
            }
        
        # Prepare alternatives dictionary for TOPSIS
        alternatives = {}
        for name, scenario in self.scenarios.items():
            npv = self.calculate_npv(scenario, vessel_data=vessel_data)
            env_score = self.calculate_environmental_score(scenario)
            ops_score = self.calculate_operational_score(scenario)
            
            alternatives[name] = {
                'npv': npv,
                'environmental': env_score,
                'operational': ops_score
            }
        
        # Define criteria (min or max)
        criteria = {
            'npv': 'max',  # Higher NPV is better
            'environmental': 'max',  # Higher score is better
            'operational': 'max'  # Higher score is better
        }
        
        # Weights for criteria (must sum to 1.0)
        criteria_weights = [
            weights['economic'],      # npv (0.4)
            weights['environmental'], # environmental (0.35)
            weights['operational']    # operational (0.25)
        ]
        
        # Run TOPSIS
        topsis_results = self.topsis.rank(
            alternatives_dict=alternatives,
            criteria=criteria,
            weights=criteria_weights
        )
        
        return topsis_results
    
    def pareto_optimization_ipso(self, vessel_data: Dict,
                                 objectives: List[str] = None,
                                 n_particles: int = 30,
                                 n_iterations: int = 50) -> Dict:
        """
        ÖĞRENİM: Pareto optimization using IPSO
        
        Multi-objective optimization to find trade-offs between objectives
        """
        if objectives is None:
            objectives = ['npv', 'environmental', 'operational']
        
        # Initialize IPSO
        ipso = IPSO(
            n_particles=n_particles,
            n_iterations=n_iterations
        )
        
        # Define bounds for design variables
        # Example: [fuel_efficiency, maintenance_quality, operational_hours]
        bounds = [
            (0.7, 1.0),  # Fuel efficiency factor
            (0.5, 1.0),  # Maintenance quality
            (200, 365)   # Annual operational days
        ]
        
        # Define objective function (weighted sum for demonstration)
        def objective(x):
            """
            ÖĞRENİM: Scalarization of multi-objective
            
            Convert multi-objective to single objective using weights
            """
            efficiency_factor = x[0]
            maintenance_factor = x[1]
            operational_days = x[2]
            
            # Modify vessel data based on design variables
            modified_vessel = vessel_data.copy()
            # Apply efficiency factor to fuel consumption
            if 'fuel_consumption' in modified_vessel:
                modified_vessel['fuel_consumption'] *= efficiency_factor
            
            # Create scenario with modified parameters
            self.create_base_scenarios(modified_vessel)
            
            # Calculate objectives
            npv = self.calculate_npv(self.scenarios['retrofit'], vessel_data=modified_vessel)
            env_score = self.calculate_environmental_score(
                self.scenarios['retrofit']
            )
            ops_score = self.calculate_operational_score(
                self.scenarios['retrofit']
            )
            
            # Weighted sum (convert to minimization)
            objective_value = -(
                0.4 * (npv / 1e6) +  # Normalize NPV
                0.35 * env_score / 100 +
                0.25 * ops_score / 100
            )
            
            return objective_value
        
        # Run optimization
        print("\n=== Running IPSO for Pareto Optimization ===")
        best_position, best_value = ipso.optimize(
            objective_function=objective,
            bounds=bounds,
            maximize=False
        )
        
        return {
            'best_design': best_position,
            'best_objective': -best_value,  # Convert back to maximization
            'convergence_curve': ipso.convergence_curve,
            'design_parameters': {
                'fuel_efficiency_factor': best_position[0],
                'maintenance_quality': best_position[1],
                'operational_days': best_position[2]
            }
        }
    
    def sensitivity_analysis_extended(self, 
                                     vessel_data: Dict,
                                     parameter_ranges: Dict = None) -> Dict:
        """
        ÖĞRENİM: Extended sensitivity analysis
        
        Analyzes how changes in key parameters affect NPV
        """
        if parameter_ranges is None:
            parameter_ranges = {
                'fuel_price': [400, 600, 800, 1000],
                'carbon_tax': [50, 100, 150, 200],
                'discount_rate': [0.05, 0.08, 0.10, 0.12]
            }
        
        sensitivity_results = {}
        
        for param_name, param_values in parameter_ranges.items():
            sensitivity_results[param_name] = {}
            
            # Store original value
            original_value = getattr(self, param_name)
            
            for value in param_values:
                # Set new value
                setattr(self, param_name, value)
                
                # Recalculate NPV for all scenarios
                sensitivity_results[param_name][value] = {}
                for scenario_name, scenario in self.scenarios.items():
                    npv = self.calculate_npv(scenario, vessel_data=vessel_data)
                    sensitivity_results[param_name][value][scenario_name] = npv
            
            # Restore original value
            setattr(self, param_name, original_value)
        
        return sensitivity_results
    
    def optimize_scenarios(self, vessel_data: Dict, 
                          use_topsis: bool = True,
                          use_ipso: bool = False,
                          sensitivity_analysis: bool = True) -> Dict:
        """
        ÖĞRENİM: Main optimization workflow
        
        Performs comprehensive multi-objective analysis
        """
        try:
            self.current_vessel_data = vessel_data
            # Create scenarios
            print("DEBUG: Creating base scenarios...")
            self.create_base_scenarios(vessel_data)
            print(f"DEBUG: Scenarios created: {list(self.scenarios.keys())}")
            
            # Evaluate each scenario
            results = {}
            for name, scenario in self.scenarios.items():
                print(f"DEBUG: Evaluating scenario '{name}'...")
                results[name] = {
                    'scenario': scenario,
                    'npv': self.calculate_npv(scenario, vessel_data=vessel_data),
                    'environmental_score': self.calculate_environmental_score(scenario),
                    'operational_score': self.calculate_operational_score(scenario)
                }
                print(f"DEBUG:   NPV={results[name]['npv']:.2f}, Env={results[name]['environmental_score']:.2f}")
            
            self.optimization_results = results
            
            # TOPSIS analysis
            if use_topsis:
                print("DEBUG: Running TOPSIS decision...")
                try:
                    topsis_results = self.topsis_decision(vessel_data=vessel_data)
                    print(f"DEBUG: TOPSIS completed. Ranking: {topsis_results['ranking']}")
                    for name in results:
                        results[name]['topsis_score'] = topsis_results['scores'][name]
                    results['topsis_ranking'] = topsis_results['ranking']
                    results['topsis_details'] = topsis_results
                except Exception as topsis_error:
                    print(f"DEBUG ERROR in TOPSIS: {type(topsis_error).__name__}: {topsis_error}")
                    import traceback
                    traceback.print_exc()
                    raise
            
            # IPSO optimization (optional)
            if use_ipso:
                print("DEBUG: Running IPSO optimization...")
                ipso_results = self.pareto_optimization_ipso(vessel_data)
                results['ipso_optimization'] = ipso_results
            
            # Find Pareto front
            print("DEBUG: Finding Pareto front...")
            self.find_pareto_front()
            results['pareto_front'] = self.pareto_front
            
            # Sensitivity analysis
            if sensitivity_analysis:
                print("DEBUG: Running sensitivity analysis...")
                results['sensitivity'] = self.sensitivity_analysis_extended(vessel_data=vessel_data)
            
            print("DEBUG: Optimization complete!")
            return results
            
        except Exception as e:
            print(f"DEBUG ERROR in optimize_scenarios: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def find_pareto_front(self):
        """
        ÖĞRENİM: Pareto optimality analysis
        
        Identifies solutions that are not dominated by any other solution
        """
        if not self.optimization_results:
            return
        
        # Extract objectives for each scenario
        objectives = []
        scenario_names = []
        
        # Only process actual scenarios, not metadata keys
        scenario_keys = ['current', 'retrofit', 'newbuild']
        
        for name in scenario_keys:
            if name not in self.optimization_results:
                continue
                
            result = self.optimization_results[name]
            
            # We want to maximize NPV, environmental, and operational scores
            objectives.append([
                result['npv'],  # Maximize NPV
                result['environmental_score'],  # Maximize environmental score
                result['operational_score']     # Maximize operational score
            ])
            scenario_names.append(name)
        
        objectives = np.array(objectives)
        
        # Find Pareto front
        pareto_indices = []
        n_solutions = len(objectives)
        
        for i in range(n_solutions):
            is_dominated = False
            for j in range(n_solutions):
                if i != j:
                    # Check if solution i is dominated by solution j
                    # (all objectives of j >= i, and at least one strictly better)
                    if all(objectives[j] >= objectives[i]) and any(objectives[j] > objectives[i]):
                        is_dominated = True
                        break
            
            if not is_dominated:
                pareto_indices.append(i)
        
        self.pareto_front = [
            scenario_names[i] for i in pareto_indices
        ]
        
        return self.pareto_front
    
    def generate_comprehensive_report(self, vessel_data: Dict = None) -> Dict:
        """
        ÖĞRENİM: Comprehensive reporting
        
        Generates detailed analysis report with all metrics and recommendations
        """
        if vessel_data is None:
            vessel_data = self.current_vessel_data
            
        # TOPSIS results
        topsis_results = self.topsis_decision(vessel_data=vessel_data)
        
        # Calculate all metrics for scenarios
        detailed_results = {}
        for name, scenario in self.scenarios.items():
            detailed_results[name] = {
                'scenario': scenario,
                'npv': self.calculate_npv(scenario, vessel_data=vessel_data),
                'environmental_score': self.calculate_environmental_score(scenario),
                'operational_score': self.calculate_operational_score(scenario),
                'topsis_score': topsis_results['scores'][name]
            }
        
        # Best scenario by TOPSIS
        best_scenario = topsis_results['ranking'][0]
        
        report = {
            'topsis_ranking': topsis_results['ranking'],
            'topsis_scores': topsis_results['scores'],
            'detailed_results': detailed_results,
            'pareto_front': self.pareto_front,
            'best_scenario': best_scenario,
            'recommendations': self._generate_recommendations(
                best_scenario, detailed_results
            ),
            'sensitivity': self.sensitivity_analysis_extended(vessel_data=vessel_data)
        }
        
        return report
    
    def generate_report(self) -> Dict:
        """
        Alias for generate_comprehensive_report() for backward compatibility
        """
        return self.generate_comprehensive_report()
    
    def _generate_recommendations(self, best_scenario: str, 
                                  results: Dict) -> List[str]:
        """
        ÖĞRENİM: AI-powered recommendations
        
        Generates actionable recommendations based on analysis results
        """
        recommendations = []
        
        best_result = results[best_scenario]
        
        if best_scenario == 'current':
            recommendations.append(
                "✓ Current operations remain viable but monitor CII ratings closely."
            )
            if best_result['scenario'].risk_factor > 0.7:
                recommendations.append(
                    "⚠ High operational risk detected. Consider maintenance upgrades."
                )
            recommendations.append(
                "📊 Continue monitoring fuel efficiency and regulatory compliance."
            )
        
        elif best_scenario == 'retrofit':
            recommendations.append(
                f"✓ Retrofit recommended with NPV: ${best_result['npv']:,.0f}."
            )
            recommendations.append(
                "🔧 Focus on energy efficiency technologies for maximum ROI."
            )
            recommendations.append(
                "⏱ Plan 6-month shipyard period for retrofit implementation."
            )
            recommendations.append(
                "💰 Expected 25% reduction in fuel consumption and emissions."
            )
        
        else:  # newbuild
            recommendations.append(
                f"✓ New build recommended with NPV: ${best_result['npv']:,.0f}."
            )
            recommendations.append(
                "🚢 Consider latest green technologies and alternative fuels."
            )
            recommendations.append(
                "📈 Expected 45% improvement in efficiency over current vessel."
            )
            recommendations.append(
                "♻️ Evaluate scrap value of current vessel for capital offset."
            )
        
        # Environmental compliance check
        for name, result in results.items():
            if result['environmental_score'] < 50:
                recommendations.append(
                    f"⚠ {name.title()} scenario needs environmental improvements to meet future regulations."
                )
        
        # Financial risk assessment
        if best_result['npv'] < 0:
            recommendations.append(
                "⚠ Negative NPV detected. Consider extending analysis period or revising assumptions."
            )
        
        return recommendations
    
    def visualize_results(self, save_path: str = None):
        """
        ÖĞRENİM: Visualization of results
        
        Creates comprehensive visualization with 4 subplots
        """
        if not self.optimization_results:
            print("No results to visualize. Run optimize_scenarios() first.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('SmartCAPEX AI - Multi-Objective Optimization Results', 
                     fontsize=16, fontweight='bold')
        
        scenario_names = list(self.scenarios.keys())
        
        # Subplot 1: NPV Comparison
        npvs = [self.optimization_results[name]['npv'] for name in scenario_names]
        colors = ['#FF6B6B', '#4ECDC4', '#95E1D3']
        axes[0, 0].bar(scenario_names, npvs, color=colors, alpha=0.8)
        axes[0, 0].set_title('Net Present Value (NPV) Comparison', fontweight='bold')
        axes[0, 0].set_ylabel('NPV (USD)')
        axes[0, 0].axhline(y=0, color='black', linestyle='--', linewidth=0.8)
        axes[0, 0].grid(axis='y', alpha=0.3)
        
        # Format y-axis as currency
        axes[0, 0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e6:.1f}M'))
        
        # Subplot 2: Multi-Criteria Scores
        env_scores = [self.optimization_results[name]['environmental_score'] for name in scenario_names]
        ops_scores = [self.optimization_results[name]['operational_score'] for name in scenario_names]
        
        x = np.arange(len(scenario_names))
        width = 0.35
        
        axes[0, 1].bar(x - width/2, env_scores, width, label='Environmental', color='#82CD47', alpha=0.8)
        axes[0, 1].bar(x + width/2, ops_scores, width, label='Operational', color='#F6A96A', alpha=0.8)
        axes[0, 1].set_title('Environmental & Operational Scores', fontweight='bold')
        axes[0, 1].set_ylabel('Score (0-100)')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(scenario_names)
        axes[0, 1].legend()
        axes[0, 1].grid(axis='y', alpha=0.3)
        axes[0, 1].set_ylim(0, 100)
        
        # Subplot 3: Trade-off Analysis (Pareto Front)
        axes[1, 0].scatter(npvs, env_scores, s=200, c=colors, alpha=0.6, edgecolors='black', linewidth=2)
        
        for i, name in enumerate(scenario_names):
            axes[1, 0].annotate(name.upper(), (npvs[i], env_scores[i]), 
                              textcoords="offset points", xytext=(0,10), ha='center',
                              fontweight='bold', fontsize=9)
        
        axes[1, 0].set_title('Economic vs Environmental Trade-off', fontweight='bold')
        axes[1, 0].set_xlabel('NPV (USD)')
        axes[1, 0].set_ylabel('Environmental Score')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e6:.1f}M'))
        
        # Subplot 4: TOPSIS Ranking
        if 'topsis_ranking' in self.optimization_results:
            topsis_results = self.topsis_decision(vessel_data=self.current_vessel_data)
            topsis_scores = [topsis_results['scores'][name] for name in scenario_names]
            
            axes[1, 1].barh(scenario_names, topsis_scores, color=colors, alpha=0.8)
            axes[1, 1].set_title('TOPSIS Ranking (Closeness to Ideal)', fontweight='bold')
            axes[1, 1].set_xlabel('TOPSIS Score (0-1, higher is better)')
            axes[1, 1].set_xlim(0, 1)
            axes[1, 1].grid(axis='x', alpha=0.3)
            
            # Add value labels
            for i, (name, score) in enumerate(zip(scenario_names, topsis_scores)):
                axes[1, 1].text(score + 0.02, i, f'{score:.3f}', 
                              va='center', fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to: {save_path}")
        
        plt.show()
    
    def export_results_to_json(self, filepath: str):
        """
        ÖĞRENİM: Export results to JSON
        
        Saves analysis results for integration with other systems
        """
        if not self.optimization_results:
            print("No results to export. Run optimize_scenarios() first.")
            return
        
        # Prepare export data
        t_results = self.topsis_decision(vessel_data=self.current_vessel_data)
        export_data = {
            'scenarios': {},
            'topsis_ranking': t_results['ranking'],
            'pareto_front': self.pareto_front,
            'recommendations': self._generate_recommendations(
                t_results['ranking'][0],
                {name: {'scenario': self.scenarios[name], 
                       'npv': self.optimization_results[name]['npv'],
                       'environmental_score': self.optimization_results[name]['environmental_score'],
                       'operational_score': self.optimization_results[name]['operational_score']}
                 for name in self.scenarios}
            )
        }
        
        for name, result in self.optimization_results.items():
            scenario = result['scenario']
            export_data['scenarios'][name] = {
                'name': scenario.name,
                'capex': scenario.capex,
                'opex': scenario.opex,
                'fuel_consumption': scenario.fuel_consumption,
                'co2_emission': scenario.co2_emission,
                'cii_score': scenario.cii_score,
                'eedi_score': scenario.eedi_score,
                'lifespan': scenario.lifespan,
                'retrofit_time': scenario.retrofit_time,
                'risk_factor': scenario.risk_factor,
                'npv': result['npv'],
                'environmental_score': result['environmental_score'],
                'operational_score': result['operational_score'],
                'topsis_score': t_results['scores'][name]
            }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        print(f"Results exported to: {filepath}")


# Example usage and testing
if __name__ == "__main__":
    print("="*70)
    print("SmartCAPEX AI - Multi-Objective Optimizer")
    print("TOPSIS + IPSO Integration for Maritime Retrofit Decisions")
    print("="*70)
    
    # Create optimizer
    optimizer = MultiObjectiveOptimizer()
    
    # Example vessel data (Turkish Coaster)
    vessel_data = {
        'dwt': 5000,
        'age': 15,
        'fuel_consumption': 18,
        'co2_emission': 55,
        'cii_score': 4.5,
        'eedi_score': 20,
        'vessel_name': 'M/V Turkish Coaster'
    }
    
    print(f"\nAnalyzing vessel: {vessel_data['vessel_name']}")
    print(f"DWT: {vessel_data['dwt']} tons, Age: {vessel_data['age']} years")
    
    # Run optimization
    print("\n" + "="*70)
    print("Running Multi-Objective Optimization...")
    print("="*70)
    
    results = optimizer.optimize_scenarios(
        vessel_data, 
        use_topsis=True,
        use_ipso=False,  # Set to True to run IPSO (takes longer)
        sensitivity_analysis=True
    )
    
    # Display results
    print("\n" + "="*70)
    print("OPTIMIZATION RESULTS")
    print("="*70)
    
    for name in ['current', 'retrofit', 'newbuild']:
        result = results[name]
        print(f"\n{result['scenario'].name.upper()}:")
        print(f"  💰 NPV (15 years): ${result['npv']:,.0f}")
        print(f"  🌱 Environmental Score: {result['environmental_score']:.1f}/100")
        print(f"  ⚙️  Operational Score: {result['operational_score']:.1f}/100")
        if 'topsis_score' in result:
            print(f"  📊 TOPSIS Score: {result['topsis_score']:.3f}")
        print(f"  ⛽ Fuel Consumption: {result['scenario'].fuel_consumption:.1f} tons/day")
        print(f"  ♨️  CO2 Emission: {result['scenario'].co2_emission:.1f} tons/day")
    
    # TOPSIS Ranking
    if 'topsis_ranking' in results:
        print("\n" + "="*70)
        print("TOPSIS RANKING (Best to Worst):")
        print("="*70)
        for i, scenario in enumerate(results['topsis_ranking'], 1):
            print(f"  {i}. {scenario.upper()}")
    
    # Pareto Front
    print("\n" + "="*70)
    print("PARETO OPTIMAL SOLUTIONS:")
    print("="*70)
    print(f"  {', '.join([s.upper() for s in results['pareto_front']])}")
    
    # Recommendations
    report = optimizer.generate_comprehensive_report()
    print("\n" + "="*70)
    print("AI RECOMMENDATIONS:")
    print("="*70)
    for rec in report['recommendations']:
        print(f"  {rec}")
    
    # Sensitivity Analysis
    print("\n" + "="*70)
    print("SENSITIVITY ANALYSIS (Fuel Price Impact on NPV):")
    print("="*70)
    if 'sensitivity' in results:
        for price, scenarios in results['sensitivity']['fuel_price'].items():
            print(f"\n  Fuel Price: ${price}/ton")
            for scenario, npv in scenarios.items():
                print(f"    {scenario.title()}: ${npv:,.0f}")
    
    # Visualization
    print("\n" + "="*70)
    print("Generating visualizations...")
    print("="*70)
    optimizer.visualize_results()
    
    # Export results
    optimizer.export_results_to_json('optimization_results.json')
    
    print("\n" + "="*70)
    print("Analysis Complete!")
    print("="*70)
