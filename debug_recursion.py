"""
Debug script to catch and display the full recursion error traceback
"""
import sys
import traceback

sys.path.append('.')

try:
    print("=" * 70)
    print("TESTING SURROGATE MODELER TRAINING")
    print("=" * 70)
    
    from agents.surrogate_modeler import SurrogateModeler
    
    print("\n✓ Import successful")
    print("✓ Creating instance...")
    
    modeler = SurrogateModeler(vessel_id="debug_test")
    
    print("✓ Instance created")
    print("✓ Starting training (this may take a while)...")
    print()
    
    # Try to train with a small dataset for quick testing
    import pandas as pd
    test_data = modeler.generate_training_data(num_samples=100)  # Small dataset
    
    print(f"✓ Generated {len(test_data)} training samples")
    print("✓ Starting model training...")
    
    results = modeler.train_models(data_df=test_data)
    
    print("\n" + "=" * 70)
    print("✅ TRAINING SUCCESSFUL!")
    print("=" * 70)
    print(f"EANN Loss: {results['eann_loss']:.4f}")
    print(f"EANN MAE: {results['eann_mae']:.4f}")
    print(f"EANN RMSE: {results['eann_rmse']:.4f}")
    print(f"RF Score: {results['rf_score']:.4f}")
    print(f"GB Score: {results['gb_score']:.4f}")
    
except RecursionError as e:
    print("\n" + "=" * 70)
    print("❌ RECURSION ERROR DETECTED!")
    print("=" * 70)
    print(f"Error: {str(e)[:200]}")
    print("\nFull Traceback:")
    print("=" * 70)
    traceback.print_exc()
    
    # Try to find the repeating pattern
    tb = traceback.format_exc()
    lines = tb.split('\n')
    
    print("\n" + "=" * 70)
    print("RECURSION PATTERN (first 50 lines):")
    print("=" * 70)
    for line in lines[:50]:
        print(line)
    
except Exception as e:
    print("\n" + "=" * 70)
    print(f"❌ ERROR: {type(e).__name__}")
    print("=" * 70)
    print(f"Message: {str(e)}")
    print("\nFull Traceback:")
    print("=" * 70)
    traceback.print_exc()
