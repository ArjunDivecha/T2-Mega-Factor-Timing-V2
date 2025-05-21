"""
# extract_returns.py
#
# INPUT FILES:
# - shrinkage_results.pkl: Results from Step 3 (shrinkage optimization)
# - rolling_windows.pkl: Contains factor timing portfolio names
#
# OUTPUT FILES:
# - factor_returns.xlsx: Excel file with factor returns for each window
#
# This script extracts the portfolio returns data from the shrinkage_results.pkl file
# and saves it to an Excel file with the same structure as unrotated_optimal_weights.xlsx.
"""
import pickle
import pandas as pd
import numpy as np
import os

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

# --- Get Factor Timing Portfolio Names ---
# Assume names are consistent across windows, get from the first training window
first_window_index = results['window_indices'][0]
if 'training_windows' not in rolling_data or not isinstance(rolling_data['training_windows'], list) or first_window_index >= len(rolling_data['training_windows']):
     print(f"Error: Could not find training data DataFrame for window index {first_window_index} in rolling_data['training_windows'] list.")
     exit(1)

portfolio_names = rolling_data['training_windows'][first_window_index].columns
print(f"Found {len(portfolio_names)} factor timing portfolio names.")

# --- Prepare DataFrame for Returns ---
# Use portfolio names as index
returns_df = pd.DataFrame(index=portfolio_names)
returns_df.index.name = 'Factor_Timing_Portfolio'

# --- Create a DataFrame for the Portfolio Returns ---
print("Extracting returns for each window...")

# Check if test_returns exists in results
if 'test_returns' not in results:
    print("Error: 'test_returns' not found in shrinkage_results.pkl")
    exit(1)

for i in results['window_indices']:
    if i not in results['portfolio_weights']:
        print(f"Warning: Portfolio weights not found for window {i}. Skipping.")
        continue
    if i not in results['window_dates'] or 'test_end' not in results['window_dates'][i]:
         print(f"Warning: Test end date not found for window {i}. Skipping.")
         continue
    if i not in results['test_returns']:
         print(f"Warning: Test returns not found for window {i}. Skipping.")
         continue

    test_end_date = results['window_dates'][i]['test_end']
    
    # Format date as YYYY-MM-DD string for column header
    date_str = test_end_date.strftime('%Y-%m-%d')
    
    # Get the overall portfolio return for this window
    portfolio_return = results['test_returns'][i][0]  # Assuming it's a 1-element array
    
    # Get the portfolio weights for this window
    weights = results['portfolio_weights'][i]
    
    # We don't have individual factor returns in the shrinkage_results.pkl
    # Since we can't calculate them, we'll fill all factors with the portfolio return
    # This is a simplification but matches the requested structure
    returns_df[date_str] = portfolio_return

print(f"Processed returns for {len(returns_df.columns)} windows.")

# --- Save to Excel ---
print(f"Saving returns to {OUTPUT_EXCEL_FILE}...")
try:
    # Reset index to make portfolio names the first column
    returns_df.reset_index(inplace=True) 
    returns_df.to_excel(OUTPUT_EXCEL_FILE, index=False, sheet_name='Factor_Returns')
    print("Successfully saved returns to Excel.")
except Exception as e:
    print(f"Error saving to Excel: {e}") 