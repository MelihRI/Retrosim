"""Test script to verify imports and basic syntax"""
import sys
sys.path.append('.')

try:
    # First test - can we import?
    print("Test 1: Attempting to import multi_objective_optimizer...")
    import agents.multi_objective_optimizer as moo
    print("✓ Import successful")
    
    # Test 2: Can we create an instance?
    print("\nTest 2: Creating MultiObjectiveOptimizer instance...")
    optimizer = moo.MultiObjectiveOptimizer()
    print("✓ Instance created successfully")
    
    print("\n✓ All tests passed!")
    
except SyntaxError as e:
    print(f"❌ Syntax Error: {e}")
    print(f"   File: {e.filename}")
    print(f"   Line: {e.lineno}")
    print(f"   Text: {e.text}")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
