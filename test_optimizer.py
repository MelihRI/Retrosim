"""
Test the multi-objective optimizer to see the exact error
"""
import sys
import traceback
sys.path.append('.')

try:
    print("=" * 70)
    print("TESTING MULTI-OBJECTIVE OPTIMIZER")
    print("=" * 70)
    
    from agents.multi_objective_optimizer import MultiObjectiveOptimizer
    
    print("\n✓ Import successful")
    print("✓ Creating optimizer instance...")
    
    optimizer = MultiObjectiveOptimizer()
    
    print("✓ Optimizer created")
    
    # Test vessel data
    vessel_data = {
        'dwt': 5000,
        'age': 15,
        'length': 100,
        'breadth': 16,
        'draft': 6.5,
        'speed': 12,
        'fuel_consumption': 18,
        'co2_emission': 55,
        'cii_score': 4.5,
        'eedi_score': 20,
        'wave_height': 2.0,
        'wind_speed': 15,
        'sea_state': 3,
        'load_factor': 0.8,
        'engine_efficiency': 0.42,
        'fuel_type': 'HFO'
    }
    
    print("\n✓ Running optimization...")
    results = optimizer.optimize_scenarios(vessel_data)
    
    print("\n✓ Optimization complete!")
    print(f"Results keys: {list(results.keys())}")
    
    # Check what's in the results
    for key in results.keys():
        print(f"\n  {key}: {type(results[key])}")
        if isinstance(results[key], dict) and key in ['current', 'retrofit', 'newbuild']:
            print(f"    Subkeys: {list(results[key].keys())}")
    
    print("\n✓ Generating report...")
    report = optimizer.generate_report()
    
    print("\n✓ Report generated!")
    print(f"Report keys: {list(report.keys())}")
    
    print("\n" + "=" * 70)
    print("✅ TEST SUCCESSFUL!")
    print("=" * 70)
    
except Exception as e:
    print("\n" + "=" * 70)
    print(f"❌ ERROR: {type(e).__name__}")
    print("=" * 70)
    print(f"Message: {str(e)}")
    print("\nFull Traceback:")
    print("=" * 70)
    traceback.print_exc()
