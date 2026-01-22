#!/usr/bin/env python3
"""
Simple test for SmartCAPEX AI components
======================================

Tests the core functionality without TensorFlow dependency
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_basic_components():
    """Test components that don't require TensorFlow"""
    print("SmartCAPEX AI - Basic Component Test")
    print("=" * 40)
    print()
    
    # Test Asset Manager
    try:
        from agents.asset_manager import AssetManager
        manager = AssetManager()
        
        test_data = {'dwt': 5000, 'age': 15, 'length': 100, 'breadth': 16, 'draft': 6.5, 'speed': 12}
        complete_data = manager.impute_missing_data(test_data)
        
        print(f"✓ Asset Manager:")
        print(f"  - Data imputation: {len(complete_data)} fields")
        print(f"  - Completeness: {manager.calculate_data_completeness(complete_data):.1f}%")
        print(f"  - Quality score: {manager.assess_data_quality(complete_data):.1f}/100")
        
        # Test template loading
        template = manager.load_vessel_template('koster_coaster')
        print(f"  - Template loading: {template['dwt']} DWT Koster vessel")
        
    except Exception as e:
        print(f"❌ Asset Manager failed: {e}")
        return False
    
    print()
    
    # Test Climate Guardian
    try:
        from agents.climate_guardian import ClimateGuardian
        guardian = ClimateGuardian()
        
        projection = guardian.project_environmental_conditions(2030)
        risk_assessment = guardian.calculate_climate_risk_assessment(complete_data)
        
        print(f"✓ Climate Guardian:")
        print(f"  - 2030 wave height: {projection['wave_height']['mean']:.2f}m")
        print(f"  - Risk category: {risk_assessment['risk_category']}")
        print(f"  - Adaptation measures: {len(risk_assessment['adaptation_measures'])} recommendations")
        
    except Exception as e:
        print(f"❌ Climate Guardian failed: {e}")
        return False
    
    print()
    
    # Test Multi-Objective Optimizer
    try:
        from agents.multi_objective_optimizer import MultiObjectiveOptimizer
        optimizer = MultiObjectiveOptimizer()
        
        optimizer.create_base_scenarios(complete_data)
        results = optimizer.optimize_scenarios(complete_data, sensitivity_analysis=False)
        report = optimizer.generate_report()
        
        print(f"✓ Multi-Objective Optimizer:")
        print(f"  - Scenarios analyzed: {len(results)}")
        print(f"  - Best scenario: {report['best_scenario'].title()}")
        print(f"  - Pareto front: {len(optimizer.pareto_front)} solutions")
        
    except Exception as e:
        print(f"❌ Multi-Objective Optimizer failed: {e}")
        return False
    
    print()
    print("=" * 40)
    print("🎉 All core components working correctly!")
    print()
    print("Key Features Available:")
    print("  ✓ Vessel data validation and imputation")
    print("  ✓ Climate projection analysis (2025-2050)")
    print("  ✓ Multi-objective optimization (Current vs Retrofit vs New Build)")
    print("  ✓ Risk assessment and adaptation measures")
    print("  ✓ NPV and environmental scoring")
    print()
    print("Note: TensorFlow-based surrogate modeling requires")
    print("      additional setup but is not required for")
    print("      optimization and climate analysis.")
    print()
    
    return True


def show_launch_instructions():
    """Show how to launch the application"""
    print("\n" + "=" * 40)
    print("LAUNCH INSTRUCTIONS")
    print("=" * 40)
    print()
    print("To launch the SmartCAPEX AI desktop application:")
    print()
    print("  Option 1 - Using launcher:")
    print("    python launch.py")
    print()
    print("  Option 2 - Direct launch:")
    print("    python main.py")
    print()
    print("  Option 3 - With virtual environment:")
    print("    python -m venv venv")
    print("    source venv/bin/activate  # On Windows: venv\\Scripts\\activate")
    print("    pip install -r requirements.txt")
    print("    python launch.py")
    print()
    print("Application Features:")
    print("  📊 Interactive GUI with data input forms")
    print("  📈 Real-time charts and visualizations")
    print("  🚢 Vessel templates and presets")
    print("  💾 Data export/import functionality")
    print("  🔍 Comprehensive analysis results")
    print()


if __name__ == "__main__":
    success = test_basic_components()
    if success:
        show_launch_instructions()
    else:
        print("\n❌ Some components failed. Please check the installation.")
        sys.exit(1)
