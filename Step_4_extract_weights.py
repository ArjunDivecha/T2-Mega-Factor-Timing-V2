# extract_weights.py
import pickle
import pandas as pd
import numpy as np
import os

ROLLING_WINDOWS_FILE = "rolling_windows.pkl"
SHRINKAGE_RESULTS_FILE = "shrinkage_results.pkl"
OUTPUT_EXCEL_FILE = "unrotated_optimal_weights.xlsx"

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
first_window_index = results['window_indices'][0] # Use the actual index from results
if 'training_windows' not in rolling_data or not isinstance(rolling_data['training_windows'], list) or first_window_index >= len(rolling_data['training_windows']):
     print(f"Error: Could not find training data DataFrame for window index {first_window_index} in rolling_data['training_windows'] list.")
     exit(1)

portfolio_names = rolling_data['training_windows'][first_window_index].columns
print(f"Found {len(portfolio_names)} factor timing portfolio names.")

# --- Prepare DataFrame ---
# Use portfolio names as index
weights_df = pd.DataFrame(index=portfolio_names)
weights_df.index.name = 'Factor_Timing_Portfolio'

# --- Populate DataFrame with Weights ---
print("Extracting weights for each window...")
num_portfolios_expected = len(portfolio_names)

for i in results['window_indices']:
    if i not in results['portfolio_weights']:
        print(f"Warning: Portfolio weights not found for window {i}. Skipping.")
        continue
    if i not in results['window_dates'] or 'test_end' not in results['window_dates'][i]:
         print(f"Warning: Test end date not found for window {i}. Skipping.")
         continue

    optimal_weights = results['portfolio_weights'][i]
    test_end_date = results['window_dates'][i]['test_end']
    
    # Format date as YYYY-MM-DD string for column header
    date_str = test_end_date.strftime('%Y-%m-%d')

    # Check if number of weights matches number of portfolio names
    if len(optimal_weights) != num_portfolios_expected:
        print(f"Warning: Mismatch in number of weights ({len(optimal_weights)}) and portfolios ({num_portfolios_expected}) for window {i} ({date_str}). Skipping.")
        continue

    weights_df[date_str] = optimal_weights

print(f"Processed weights for {len(weights_df.columns)} windows.")

# --- Save to Excel ---
print(f"Saving weights to {OUTPUT_EXCEL_FILE}...")
try:
    # Reset index to make portfolio names the first column
    weights_df.reset_index(inplace=True) 
    weights_df.to_excel(OUTPUT_EXCEL_FILE, index=False, sheet_name='Optimal_Weights')
    print("Successfully saved weights to Excel.")
except Exception as e:
    print(f"Error saving to Excel: {e}")
