"""
# extract_returns_optimized.py
#
# INPUT FILES:
# - rolling_windows.pkl: Contains training, validation, and test windows with factor returns
# - shrinkage_results.pkl: Contains results from Step 3 (shrinkage optimization)
#
# OUTPUT FILES:
# - factor_returns.xlsx: Excel file with individual factor returns
#
# This script extracts the individual factor returns calculated in Step 3 shrinkage
# and saves them to an Excel file with the same structure as unrotated_optimal_weights.xlsx.
"""
import pickle
import pandas as pd
import numpy as np
import os
import gc

ROLLING_WINDOWS_FILE = "rolling_windows.pkl"
SHRINKAGE_RESULTS_FILE = "shrinkage_results.pkl"
OUTPUT_EXCEL_FILE = "factor_returns.xlsx"

print(f"Loading rolling windows data from {ROLLING_WINDOWS_FILE}...")
if not os.path.exists(ROLLING_WINDOWS_FILE):
    print(f"Error: {ROLLING_WINDOWS_FILE} not found.")
    exit(1)
with open(ROLLING_WINDOWS_FILE, 'rb') as f:
    rolling_data = pickle.load(f)

print(f"Loading shrinkage results from {SHRINKAGE_RESULTS_FILE}...")
if not os.path.exists(SHRINKAGE_RESULTS_FILE):
    print(f"Error: {SHRINKAGE_RESULTS_FILE} not found.")
    exit(1)
with open(SHRINKAGE_RESULTS_FILE, 'rb') as f:
    results = pickle.load(f)

# Get all factor timing portfolio names
first_window_index = results['window_indices'][0]
if 'training_windows' not in rolling_data or not isinstance(rolling_data['training_windows'], list) or first_window_index >= len(rolling_data['training_windows']):
     print(f"Error: Could not find training data for window index {first_window_index}.")
     exit(1)

# Get portfolio names (factor names)
portfolio_names = rolling_data['training_windows'][first_window_index].columns
print(f"Found {len(portfolio_names)} factor timing portfolios.")

# --- Extract individual factor returns for each window ---
print("Extracting individual factor returns...")

# Create data structure to hold returns
dates = []
factor_returns_data = {}

# Iterate through each window
for i in results['window_indices']:
    if i not in results['window_dates'] or 'test_end' not in results['window_dates'][i]:
        print(f"Warning: Test end date not found for window {i}. Skipping.")
        continue
        
    # Get the test window end date for the column label
    test_end_date = results['window_dates'][i]['test_end']
    date_str = test_end_date.strftime('%Y-%m-%d')
    dates.append(date_str)
    
    # Get the individual factor returns for this window directly from the test window data
    if i < len(rolling_data['testing_windows']):
        test_window = rolling_data['testing_windows'][i]
        
        # Calculate the factor returns for the test window
        # We need just one return per factor, so we'll take the mean of the test window
        window_returns = test_window.mean().values
        
        # Store the returns for this date
        factor_returns_data[date_str] = window_returns
    else:
        print(f"Warning: Test window {i} not found in rolling data. Skipping.")

print(f"Extracted returns for {len(dates)} windows.")

# --- Create DataFrame ---
print("Creating returns DataFrame...")
returns_df = pd.DataFrame(factor_returns_data, index=portfolio_names)
returns_df.index.name = 'Factor_Timing_Portfolio'

# --- Save to Excel ---
print(f"Saving returns to {OUTPUT_EXCEL_FILE}...")
try:
    # Reset index to make portfolio names the first column
    returns_df = returns_df.reset_index()
    returns_df.to_excel(OUTPUT_EXCEL_FILE, index=False, sheet_name='Factor_Returns')
    print("Successfully saved returns to Excel.")
except Exception as e:
    print(f"Error saving to Excel: {e}")

# Clean up to free memory
del rolling_data, results, factor_returns_data
gc.collect()

print(f"Process completed. Results saved to {OUTPUT_EXCEL_FILE}") 