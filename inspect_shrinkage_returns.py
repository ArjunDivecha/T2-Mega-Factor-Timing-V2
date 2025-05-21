"""
# inspect_shrinkage_returns.py
#
# INPUT FILES:
# - shrinkage_results.pkl: Results from Step 3 (shrinkage optimization)
#
# This script examines the structure of the shrinkage_results.pkl file
# to determine if it contains returns data that can be used to create
# a spreadsheet with a similar structure to unrotated_optimal_weights.xlsx
"""
import pickle
import os
import pandas as pd

SHRINKAGE_RESULTS_FILE = "shrinkage_results.pkl"

print(f"Loading {SHRINKAGE_RESULTS_FILE} to inspect returns data...")
if not os.path.exists(SHRINKAGE_RESULTS_FILE):
    print(f"Error: {SHRINKAGE_RESULTS_FILE} not found.")
    exit(1)

try:
    with open(SHRINKAGE_RESULTS_FILE, 'rb') as f:
        results = pickle.load(f)
except Exception as e:
    print(f"Error loading pickle file: {e}")
    exit(1)

print(f"Type of loaded data: {type(results)}")

# Check for performance metrics and actual return data
if 'performance_metrics' in results:
    print("\nPerformance metrics found in shrinkage_results.pkl:")
    for metric_key in results['performance_metrics']:
        print(f"- {metric_key}")
        
    # Check the first window to see what metrics are available
    if results['window_indices'] and results['window_indices'][0] in results['performance_metrics']:
        first_window = results['window_indices'][0]
        print(f"\nAvailable metrics for window {first_window}:")
        for k, v in results['performance_metrics'][first_window].items():
            print(f"- {k}: {type(v)}")
            if isinstance(v, (float, int)):
                print(f"  Value: {v}")
            elif hasattr(v, 'shape'):
                print(f"  Shape: {v.shape}")

# Look for actual returns data
returns_available = False
for key in results:
    if 'return' in key.lower():
        returns_available = True
        print(f"\nFound returns-related key: {key}")
        if isinstance(results[key], dict):
            sample_window = next(iter(results[key]))
            print(f"Sample window {sample_window} contains: {type(results[key][sample_window])}")
            if hasattr(results[key][sample_window], 'shape'):
                print(f"Shape: {results[key][sample_window].shape}")

# Check for test window returns (these might contain the actual returns we want)
if 'test_windows' in results:
    print("\nTest windows found in results")
    sample_window = next(iter(results['test_windows']))
    print(f"Sample test window {sample_window} contains: {type(results['test_windows'][sample_window])}")
    if hasattr(results['test_windows'][sample_window], 'shape'):
        print(f"Shape: {results['test_windows'][sample_window].shape}")

# Check if we can reconstruct returns from portfolio weights and test windows
if 'portfolio_weights' in results and 'window_indices' in results:
    print("\nCan potentially reconstruct returns from portfolio weights and test returns data")
    
    # See if we can access test window data from rolling_windows.pkl
    try:
        with open("rolling_windows.pkl", 'rb') as f:
            rolling_data = pickle.load(f)
            if 'test_windows' in rolling_data:
                print("Test window data available in rolling_windows.pkl")
                if isinstance(rolling_data['test_windows'], list) and rolling_data['test_windows']:
                    sample_test = rolling_data['test_windows'][0]
                    if isinstance(sample_test, pd.DataFrame):
                        print(f"Test window contains DataFrame with shape: {sample_test.shape}")
    except Exception as e:
        print(f"Could not check rolling_windows.pkl: {e}")

# Summary
print("\nSUMMARY:")
if returns_available:
    print("Returns data appears to be available in the shrinkage_results.pkl file.")
    print("A sheet similar to unrotated_optimal_weights.xlsx with returns should be possible to create.")
else:
    print("Explicit returns data may not be directly available in shrinkage_results.pkl.")
    print("May need to calculate returns using portfolio weights and return data from test windows.") 