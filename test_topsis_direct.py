"""
Test with more debug output
"""
import sys
import traceback
sys.path.append('.')

try:
    print("=" * 70)
    print("TESTING TOPSIS DIRECTLY")
    print("=" * 70)
    
    from agents.multi_objective_optimizer import TOPSIS
    import numpy as np
    
    print("\n✓ Creating TOPSIS instance...")
    topsis = TOPSIS()
    
    # Test data
    alternatives = {
        'current': {'npv': -500000, 'environmental': 45, 'operational': 55},
        'retrofit': {'npv': 250000, 'environmental': 75, 'operational': 80},
        'newbuild': {'npv': 500000, 'environmental': 90, 'operational': 85}
    }
    
    criteria = {
        'npv': 'max',
        'environmental': 'max',
        'operational': 'max'
    }
    
    weights = [0.4, 0.35, 0.25]
    
    print("\n✓ Running TOPSIS rank...")
    print(f"Alternatives: {list(alternatives.keys())}")
    print(f"Criteria: {list(criteria.keys())}")
    print(f"Weights: {weights}")
    
    results = topsis.rank(alternatives, criteria, weights)
    
    print("\n" + "=" * 70)
    print("✅ TOPSIS TEST SUCCESSFUL!")
    print("=" * 70)
    print(f"Ranking: {results['ranking']}")
    print(f"Scores: {results['scores']}")
    
except Exception as e:
    print("\n" + "=" * 70)
    print(f"❌ ERROR: {type(e).__name__}")
    print("=" * 70)
    print(f"Message: {str(e)}")
    print("\nFull Traceback:")
    print("=" * 70)
    traceback.print_exc()
