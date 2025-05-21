"""
# check_test_returns.py
#
# INPUT FILES:
# - shrinkage_results.pkl: Results from Step 3 (shrinkage optimization)
#
# This script examines the test_returns data in shrinkage_results.pkl
# to determine if we can create a spreadsheet with factor returns
# similar to unrotated_optimal_weights.xlsx
"""
import pickle
import os
import pandas as pd
import numpy as np

SHRINKAGE_RESULTS_FILE = "shrinkage_results.pkl"
ROLLING_WINDOWS_FILE = "rolling_windows.pkl"

print(f"Loading {SHRINKAGE_RESULTS_FILE} to examine test returns...")
if not os.path.exists(SHRINKAGE_RESULTS_FILE):
    print(f"Error: {SHRINKAGE_RESULTS_FILE} not found.")
    exit(1)

try:
    with open(SHRINKAGE_RESULTS_FILE, 'rb') as f:
        results = pickle.load(f)
except Exception as e:
    print(f"Error loading shrinkage results: {e}")
    exit(1)

print(f"Loading {ROLLING_WINDOWS_FILE} for factor names...")
if not os.path.exists(ROLLING_WINDOWS_FILE):
    print(f"Warning: {ROLLING_WINDOWS_FILE} not found. Will proceed without factor names.")
    rolling_data = None
else:
    try:
        with open(ROLLING_WINDOWS_FILE, 'rb') as f:
            rolling_data = pickle.load(f)
    except Exception as e:
        print(f"Error loading rolling windows: {e}")
        rolling_data = None

# First examine the structure of test_returns
if 'test_returns' in results:
    print("\nExamining test_returns structure:")
    
    # Get example of test_returns for first window
    if results['window_indices']:
        first_window = results['window_indices'][0]
        if first_window in results['test_returns']:
            test_return = results['test_returns'][first_window]
            print(f"Test return for window {first_window}: {test_return}")
            print(f"Type: {type(test_return)}, Shape: {test_return.shape if hasattr(test_return, 'shape') else 'N/A'}")
            
            # Check if this is a single return value or a series of returns
            if isinstance(test_return, np.ndarray) and test_return.size == 1:
                print("Each test_return appears to be a single value (portfolio return)")
                
                # Create a sample of the data we could output
                print("\nSample data structure we could create:")
                
                # Get dates if available
                dates = []
                for i in results['window_indices']:
                    if i in results['window_dates'] and 'test_end' in results['window_dates'][i]:
                        date_str = results['window_dates'][i]['test_end'].strftime('%Y-%m-%d')
                        dates.append(date_str)
                
                if dates:
                    print(f"Example dates (columns): {dates[:5]}")
                
                # Check if we have portfolio names
                portfolio_names = []
                if rolling_data and 'training_windows' in rolling_data and rolling_data['training_windows']:
                    first_training = rolling_data['training_windows'][first_window]
                    if isinstance(first_training, pd.DataFrame):
                        portfolio_names = first_training.columns.tolist()
                        print(f"Found {len(portfolio_names)} portfolio names (will be rows)")
                    
                # Create sample dataframe
                sample_df = pd.DataFrame(index=['Portfolio Return'])
                for i, date in enumerate(dates[:3]):  # Just show first 3 dates
                    if i < len(results['window_indices']):
                        window_idx = results['window_indices'][i]
                        if window_idx in results['test_returns']:
                            sample_df[date] = results['test_returns'][window_idx][0]  # Assuming single value
                
                print("\nSample output DataFrame:")
                print(sample_df)
            
            # Check if we can get individual factor returns instead of just the portfolio return
            print("\nChecking if individual factor returns are available...")
            
            # Look for factor returns in test_windows or other attributes
            factor_returns_available = False
            
            # Check if test_windows are available in shrinkage_results
            if 'test_windows' in results:
                print("Test windows found in shrinkage_results")
                if first_window in results['test_windows']:
                    test_window = results['test_windows'][first_window]
                    print(f"Type: {type(test_window)}")
                    if hasattr(test_window, 'shape'):
                        print(f"Shape: {test_window.shape}")
                    factor_returns_available = True
            
            # If not in shrinkage_results, try rolling_windows.pkl
            elif rolling_data and 'test_windows' in rolling_data:
                print("Test windows found in rolling_data")
                if isinstance(rolling_data['test_windows'], list) and len(rolling_data['test_windows']) > first_window:
                    test_window = rolling_data['test_windows'][first_window]
                    print(f"Type: {type(test_window)}")
                    if hasattr(test_window, 'shape'):
                        print(f"Shape: {test_window.shape}")
                    factor_returns_available = True
            
            if factor_returns_available:
                print("\nIndividual factor returns appear to be available.")
                print("Could create a spreadsheet with factor returns instead of just portfolio returns.")
            else:
                print("\nIndividual factor returns might not be directly available.")
                print("May need to calculate from weights and test windows.")

# Check portfolio weights to see if we can combine them with returns
if 'portfolio_weights' in results:
    print("\nExamining portfolio_weights structure:")
    first_window = results['window_indices'][0]
    if first_window in results['portfolio_weights']:
        weights = results['portfolio_weights'][first_window]
        print(f"Type: {type(weights)}, Shape: {weights.shape if hasattr(weights, 'shape') else 'N/A'}")
        if isinstance(weights, np.ndarray):
            print(f"First few weights: {weights[:5]}")

# Create final assessment
print("\nFINAL ASSESSMENT:")
can_create_similar_structure = 'test_returns' in results
if can_create_similar_structure:
    print("YES - We can create a spreadsheet with the same structure as unrotated_optimal_weights.xlsx,")
    print("      but with returns instead of weights using the test_returns data from shrinkage_results.pkl.")
    print("\nRecommended approach:")
    print("1. Load shrinkage_results.pkl")
    print("2. Create a DataFrame with the same row indices as unrotated_optimal_weights.xlsx")
    print("3. For each window/date, populate the DataFrame with test returns instead of weights")
    print("4. Save to a new Excel file")
else:
    print("NO - The data structure needed to create a returns spreadsheet with the same structure")
    print("     as unrotated_optimal_weights.xlsx does not appear to be directly available.") 