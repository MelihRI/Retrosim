"""
Test script for SmartCAPEX AI application
=========================================

This script tests the core functionality of the SmartCAPEX AI desktop application
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents.surrogate_modeler import SurrogateModeler
from agents.multi_objective_optimizer import MultiObjectiveOptimizer
from agents.climate_guardian import ClimateGuardian
from agents.asset_manager import AssetManager


def test_surrogate_modeler():
    """Test the Surrogate Modeler agent"""
    print("\n=== Testing Surrogate Modeler ===")
    
    # Create modeler
    modeler = SurrogateModeler()
    
    # Train model
    print("Training surrogate model...")
    results = modeler.train_models()
    print(f"EANN Score: {results['eann_score']:.4f}")
    print(f"Random Forest Score: {results['rf_score']:.4f}")
    print(f"Gradient Boosting Score: {results['gb_score']:.4f}")
    
    # Test prediction
    test_vessel = {
        'dwt': 5000,
        'age': 15,
        'length': 100,
        'breadth': 16,
        'draft': 6.5,
        'speed': 12,
        'wave_height': 2.0,
        'wind_speed': 15,
        'current_speed': 0.5,
        'sea_state': 3,
        'load_factor': 0.8,
        'fuel_type': 0,
        'engine_efficiency': 0.42
    }
    
    print("\nRunning prediction...")
    prediction = modeler.predict(test_vessel)
    
    print("Prediction Results:")
    for key, value in prediction.items():
        print(f"  {key}: {value:.3f}")
    
    # Climate impact test
    print("\nTesting climate impact...")
    climate_result = modeler.calculate_climate_impact(test_vessel, year=2050)
    print(f"2050 Fuel Consumption: {climate_result['fuel_consumption']:.2f} tons/day")
    print(f"2050 CO2 Emission: {climate_result['co2_emission']:.2f} tons/day")
    
    return True


def test_multi_objective_optimizer():
    """Test the Multi-Objective Optimizer agent"""
    print("\n=== Testing Multi-Objective Optimizer ===")
    
    # Create optimizer
    optimizer = MultiObjectiveOptimizer()
    
    # Test vessel data
    vessel_data = {
        'dwt': 5000,
        'age': 15,
        'fuel_consumption': 18,
        'co2_emission': 55,
        'cii_score': 4.2,
        'eedi_score': 20
    }
    
    # Run optimization
    print("Running optimization...")
    results = optimizer.optimize_scenarios(vessel_data)
    
    print("\nOptimization Results:")
    for name, result in results.items():
        print(f"\n{result['scenario'].name}:")
        print(f"  NPV: ${result['npv']:,.0f}")
        print(f"  Environmental Score: {result['environmental_score']:.1f}/100")
        print(f"  Operational Score: {result['operational_score']:.1f}/100")
        print(f"  MCDM Score: {result['mcdm_score']:.1f}/100")
    
    print(f"\nPareto Front: {optimizer.pareto_front}")
    
    # Generate report
    report = optimizer.generate_report()
    print(f"\nRecommended Action: {report['best_scenario'].title()}")
    
    return True


def test_climate_guardian():
    """Test the Climate Guardian agent"""
    print("\n=== Testing Climate Guardian ===")
    
    # Create guardian
    guardian = ClimateGuardian()
    
    # Test vessel data
    vessel_data = {
        'dwt': 5000,
        'age': 15,
        'fuel_consumption': 18,
        'co2_emission': 55,
        'cii_score': 4.2,
        'speed': 12
    }
    
    # Generate temporal analysis
    print("Generating temporal analysis...")
    temporal_analysis = guardian.generate_temporal_analysis(vessel_data)
    
    print("\nYearly Projections:")
    for year, projection in temporal_analysis['yearly_projections'].items():
        print(f"  {year}:")
        print(f"    Wave Height: {projection['environmental_conditions']['wave_height']['mean']:.2f}m")
        print(f"    Resistance Penalty: {projection['resistance_penalty']:.3f}")
        print(f"    Additional Costs: ${projection['annual_additional_costs']['total_additional_cost']:,.0f}/year")
    
    # Risk assessment
    print("\nCalculating risk assessment...")
    risk_assessment = guardian.calculate_climate_risk_assessment(vessel_data)
    
    print(f"\nRisk Assessment:")
    print(f"  Overall Risk: {risk_assessment['risk_category']} ({risk_assessment['overall_risk_score']:.2f})")
    print(f"  Physical Risk: {risk_assessment['physical_risk']:.2f}")
    print(f"  Transition Risk: {risk_assessment['transition_risk']:.2f}")
    print(f"  Regulatory Risk: {risk_assessment['regulatory_risk']:.2f}")
    
    print(f"\nRecommended Adaptation Measures:")
    for measure in risk_assessment['adaptation_measures']:
        print(f"  - {measure}")
    
    return True


def test_asset_manager():
    """Test the Asset Manager agent"""
    print("\n=== Testing Asset Manager ===")
    
    # Create manager
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
    
    print("Original Data Completeness:", 
          f"{manager.calculate_data_completeness(incomplete_data):.1f}%")
    print("Original Data Quality:", 
          f"{manager.assess_data_quality(incomplete_data):.1f}/100")
    
    # Impute missing data
    print("\nImputing missing data...")
    complete_data = manager.impute_missing_data(incomplete_data)
    
    print("Complete Data:")
    for key, value in complete_data.items():
        print(f"  {key}: {value}")
    
    print(f"\nComplete Data Completeness: {manager.calculate_data_completeness(complete_data):.1f}%")
    print(f"Complete Data Quality: {manager.assess_data_quality(complete_data):.1f}/100")
    
    # Test validation
    print("\nTesting validation...")
    is_valid, errors = manager.validate_all_inputs(complete_data)
    print(f"Validation Result: {'Valid' if is_valid else 'Invalid'}")
    if errors:
        print("Errors:")
        for field, error in errors.items():
            print(f"  {field}: {error}")
    
    # Test template loading
    print("\nTesting template loading...")
    template_data = manager.load_vessel_template('koster_coaster')
    print("Koster Coaster Template:")
    for key, value in template_data.items():
        print(f"  {key}: {value}")
    
    return True


def test_integration():
    """Test integration between agents"""
    print("\n=== Testing Agent Integration ===")
    
    # Create all agents
    modeler = SurrogateModeler()
    optimizer = MultiObjectiveOptimizer()
    guardian = ClimateGuardian()
    manager = AssetManager()
    
    # Test vessel data
    vessel_data = {
        'dwt': 5000,
        'length': 100,
        'breadth': 16,
        'draft': 6.5,
        'speed': 12,
        'age': 15
    }
    
    print("1. Asset Manager validates and completes data...")
    is_valid, errors = manager.validate_all_inputs(vessel_data)
    if not is_valid:
        print("Validation errors:", errors)
        return False
    
    vessel_data = manager.impute_missing_data(vessel_data)
    print("Data completed successfully")
    
    print("\n2. Surrogate Modeler makes predictions...")
    modeler.train_models(num_samples=100)  # Small sample for quick testing
    prediction = modeler.predict(vessel_data)
    print("Prediction completed:", {k: f"{v:.2f}" for k, v in list(prediction.items())[:3]})
    
    print("\n3. Multi-Objective Optimizer analyzes scenarios...")
    optimization_results = optimizer.optimize_scenarios(vessel_data, sensitivity_analysis=False)
    report = optimizer.generate_report()
    print(f"Best scenario: {report['best_scenario']}")
    
    print("\n4. Climate Guardian assesses climate risks...")
    risk_assessment = guardian.calculate_climate_risk_assessment(vessel_data)
    print(f"Risk category: {risk_assessment['risk_category']}")
    
    print("\n5. Asset Manager creates summary...")
    summary = manager.create_data_summary(vessel_data)
    print(f"Vessel: {summary['dwt']} DWT, {summary['age']} years old")
    print(f"Performance: {summary['performance']['fuel_consumption']:.1f} tons/day fuel")
    
    print("\n✓ All agents integrated successfully!")
    return True


def run_all_tests():
    """Run all tests and report results"""
    print("SmartCAPEX AI - Agent Testing")
    print("=" * 50)
    
    tests = [
        ("Surrogate Modeler", test_surrogate_modeler),
        ("Multi-Objective Optimizer", test_multi_objective_optimizer),
        ("Climate Guardian", test_climate_guardian),
        ("Asset Manager", test_asset_manager),
        ("Agent Integration", test_integration)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, "PASSED" if success else "FAILED"))
        except Exception as e:
            print(f"\nERROR in {test_name}: {str(e)}")
            results.append((test_name, "ERROR"))
    
    print("\n" + "=" * 50)
    print("TEST RESULTS SUMMARY:")
    print("=" * 50)
    
    for test_name, result in results:
        status_symbol = "✓" if result == "PASSED" else "✗"
        print(f"{status_symbol} {test_name}: {result}")
    
    passed_count = sum(1 for _, result in results if result == "PASSED")
    total_count = len(results)
    
    print(f"\n{passed_count}/{total_count} tests passed")
    
    if passed_count == total_count:
        print("\n🎉 All tests passed! SmartCAPEX AI is ready to use.")
    else:
        print(f"\n⚠️  {total_count - passed_count} test(s) failed. Please check the implementation.")
    
    return passed_count == total_count


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
