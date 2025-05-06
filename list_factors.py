"""
Inspect the contents of factor_weights.pkl

This script checks the structure of the factor_weights.pkl file
to diagnose why no factor weights were calculated.
"""

import pickle
import pprint
import numpy as np

def main():
    # Load the factor weights
    print("Loading factor weights from factor_weights.pkl...")
    with open('factor_weights.pkl', 'rb') as f:
        data = pickle.load(f)

    # Print all top-level keys
    print("\nTop-level keys in factor_weights.pkl:")
    print(list(data.keys()))
    
    # Check window indices
    print("\nWindow indices processed:")
    print(data.get('window_indices', []))
    
    # Check factor names
    print(f"\nNumber of factor names: {len(data.get('factor_names', []))}")
    if data.get('factor_names'):
        print(f"First few factor names: {data['factor_names'][:5]}")
    
    # Check if any portfolio weights exist
    portfolio_weights = data.get('portfolio_weights', {})
    print(f"\nPortfolio weights available for {len(portfolio_weights)} windows:")
    for window_idx in sorted(portfolio_weights.keys()):
        weights = portfolio_weights[window_idx]
        print(f"  Window {window_idx}: {len(weights)} weights, shape: {weights.shape if hasattr(weights, 'shape') else 'N/A'}")
        if hasattr(weights, 'shape'):
            print(f"    First few weights: {weights[:5]}")
    
    # Check if any factor weights exist
    factor_weights = data.get('factor_weights', {})
    print(f"\nFactor weights available for {len(factor_weights)} windows:")
    for window_idx in sorted(factor_weights.keys()):
        weights = factor_weights[window_idx]
        print(f"  Window {window_idx}: {len(weights)} weights, shape: {weights.shape if hasattr(weights, 'shape') else 'N/A'}")
    
    # Check if any long-only weights exist
    long_only_weights = data.get('long_only_weights', {})
    print(f"\nLong-only weights available for {len(long_only_weights)} windows:")
    for window_idx in sorted(long_only_weights.keys()):
        weights = long_only_weights[window_idx]
        print(f"  Window {window_idx}: {len(weights)} weights, shape: {weights.shape if hasattr(weights, 'shape') else 'N/A'}")
    
    # Check predictors
    print("\nPredictors available for windows:")
    if 'predictors' in data:
        for window_idx in sorted(data['predictors'].keys()):
            predictors = data['predictors'][window_idx]
            print(f"  Window {window_idx}: {len(predictors.columns) if hasattr(predictors, 'columns') else 'N/A'} predictors")
    else:
        print("  No predictors found in the data")
    
    # Check for any error messages that might have been saved
    if 'errors' in data:
        print("\nError messages found:")
        for window_idx, error in data.get('errors', {}).items():
            print(f"  Window {window_idx}: {error}")
        
    # Check if we have window dates
    if 'window_dates' in data:
        print("\nWindow dates:")
        for window_idx, dates in list(data['window_dates'].items())[:2]:  # Show first two windows
            print(f"  Window {window_idx}:")
            for period, date in dates.items():
                print(f"    {period}: {date}")
            
    # Print diagnostic summary
    print("\nDiagnostic Summary:")
    if not portfolio_weights:
        print("  - No portfolio weights were found. Check if shrinkage_results.pkl has portfolio_weights.")
    if not factor_weights and portfolio_weights:
        print("  - Portfolio weights exist but no factor weights were calculated.")
        print("  - This may be because the factor rotation matrix could not be created.")
        print("  - Check if predictors are available for the windows.")
    if not long_only_weights and factor_weights:
        print("  - Factor weights exist but no long-only weights were calculated.")
        print("  - This may be because all factor weights were negative.")

if __name__ == "__main__":
    main() 