"""Quick test to verify surrogate_modeler syntax"""
import sys
sys.path.append('.')

try:
    print("✓ Importing surrogate_modeler...")
    from agents.surrogate_modeler import SurrogateModeler
    
    print("✓ Creating instance...")
    modeler = SurrogateModeler(vessel_id="test_vessel")
    
    print("✓ Checking methods...")
    assert hasattr(modeler, 'train_models'), "Missing train_models method"
    assert hasattr(modeler, 'predict'), "Missing predict method"
    assert hasattr(modeler, 'build_emotional_ann'), "Missing build_emotional_ann method"
    assert hasattr(modeler, 'analyze_regime_detection'), "Missing analyze_regime_detection method"
    
    print("\n✅ All checks passed! No recursion error.")
    print(f"   Vessel ID: {modeler.vessel_id}")
    print(f"   Feature count: {len(modeler.feature_names)}")
    print(f"   Target count: {len(modeler.target_names)}")
    
except RecursionError as e:
    print(f"\n❌ RECURSION ERROR DETECTED:")
    print(f"   {str(e)[:200]}")
    
except Exception as e:
    print(f"\n❌ Error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
