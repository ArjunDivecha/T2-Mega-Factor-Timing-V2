"""
# Factor Timing Rolling Windows Implementation
# 
# INPUT FILES:
# - prepared_data.pkl: Processed data from Phase 1 containing:
#   - factor_returns: 102 factor returns (49 CS and 53 TS variants) 
#   - macro_vars_std: Standardized conditioning variables
#   - factor_timing: Factor timing portfolios (factor returns * lagged predictors)
#
# OUTPUT FILES:
# - rolling_windows.pkl: Dictionary containing:
#   - training_windows: List of DataFrames with 60-month training windows
#   - validation_windows: List of DataFrames with 12-month validation windows
#   - testing_windows: List of DataFrames with 1-month testing windows
#   - window_dates: Dictionary with start/end dates for each window
#
# This script implements Phase 2.1 of the factor timing methodology:
# 1. Creates 60-month rolling windows for model training
# 2. Creates 12-month validation windows for hyperparameter tuning (lambda)
# 3. Creates 1-month windows for out-of-sample testing
# 4. Ensures proper time series partitioning (no look-ahead bias)
# 5. Implements basic statistical filtering of factor timing portfolios
#
# Author: Claude
# Last Updated: May 5, 2025
"""

import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime

# Define input and output files
INPUT_FILE = "prepared_data.pkl"
OUTPUT_FILE = "rolling_windows.pkl"

def load_data(input_file):
    """
    Load the prepared data pickle file
    
    Args:
        input_file (str): Path to the pickle file
        
    Returns:
        dict: Dictionary containing preprocessed data
    """
    print(f"Loading prepared data from {input_file}...")
    
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return None
    
    with open(input_file, 'rb') as f:
        data = pickle.load(f)
    
    # Verify the required keys exist
    required_keys = ['factor_returns', 'factor_timing', 'macro_vars_std']
    for key in required_keys:
        if key not in data:
            print(f"Error: Required key '{key}' not found in data.")
            return None
    
    print(f"Successfully loaded data with {len(data['factor_timing'].columns)} factor timing portfolios")
    return data

# Filtering function removed - will be handled in the next step

def create_rolling_windows(data, train_size=60, val_size=12, test_size=1):
    """
    Create rolling windows for training, validation, and testing
    
    Args:
        data (dict): Dictionary containing preprocessed data
        train_size (int): Size of training window in months
        val_size (int): Size of validation window in months
        test_size (int): Size of testing window in months
        
    Returns:
        dict: Dictionary containing training, validation, and testing windows
    """
    print(f"Creating {train_size}-month training, {val_size}-month validation, and {test_size}-month testing windows...")
    
    # Extract data
    factor_timing = data['factor_timing']
    factor_returns = data['factor_returns']
    
    # Get the dates
    dates = factor_timing.index
    
    # Initialize lists to store windows
    training_windows = []
    validation_windows = []
    testing_windows = []
    eval_windows = []
    window_dates = {}
    
    # Create rolling windows
    n_windows = 0
    
    # Need enough data for training + validation + testing
    total_window_size = train_size + val_size + test_size
    
    for i in range(len(dates) - total_window_size + 1):
        # Get training window
        train_start = dates[i]
        train_end = dates[i + train_size - 1]
        train_data = factor_timing.loc[train_start:train_end]
        
        # Get validation window (for determining optimal lambda)
        val_start = dates[i + train_size]
        val_end = dates[i + train_size + val_size - 1]
        val_data = factor_timing.loc[val_start:val_end]
        
        # Get testing window (for final evaluation)
        test_start = dates[i + train_size + val_size]
        test_end = dates[i + train_size + val_size + test_size - 1]
        test_data = factor_timing.loc[test_start:test_end]
        
        # Get evaluation window (factor returns for the test period)
        eval_data = factor_returns.loc[test_start:test_end]
        
        # Skip if we have insufficient data
        if len(train_data) < train_size or len(val_data) < val_size or len(test_data) < test_size:
            continue
            
        # Store windows
        training_windows.append(train_data)
        validation_windows.append(val_data)
        testing_windows.append(test_data)
        eval_windows.append(eval_data)
        
        # Store dates
        window_dates[n_windows] = {
            'train_start': train_start,
            'train_end': train_end,
            'val_start': val_start,
            'val_end': val_end,
            'test_start': test_start,
            'test_end': test_end
        }
        
        n_windows += 1
    
    print(f"Created {n_windows} rolling windows")
    
    # Return results
    return {
        'training_windows': training_windows,
        'validation_windows': validation_windows,
        'testing_windows': testing_windows,
        'evaluation_windows': eval_windows,
        'window_dates': window_dates,
        'all_portfolios': factor_timing.columns.tolist()
    }

def main():
    """
    Main function to create rolling windows
    """
    print("=== FACTOR TIMING ROLLING WINDOWS ===")
    
    # Load data
    data = load_data(INPUT_FILE)
    if data is None:
        return
    
    # Create rolling windows with validation period
    windows = create_rolling_windows(data, train_size=60, val_size=12, test_size=1)
    
    # Save results
    with open(OUTPUT_FILE, 'wb') as f:
        pickle.dump(windows, f)
    
    print(f"\nSaved rolling windows to {OUTPUT_FILE}")
    
    # Print summary statistics
    n_windows = len(windows['training_windows'])
    first_window_date = windows['window_dates'][0]['train_start']
    last_window_date = windows['window_dates'][n_windows-1]['test_end']
    
    print(f"Total windows: {n_windows}")
    print(f"Time period covered: {first_window_date.strftime('%Y-%m')} to {last_window_date.strftime('%Y-%m')}")
    print(f"Total portfolios: {len(windows['all_portfolios'])}")
    
    # Calculate data usage
    train_months = 60
    val_months = 12
    test_months = 1
    total_periods = len(data['factor_returns'])
    usable_periods = total_periods - (train_months + val_months + test_months) + 1
    
    print(f"Total data periods: {total_periods} months")
    print(f"Usable for rolling windows: {usable_periods} months")
    print(f"Window structure: {train_months}m train + {val_months}m validation + {test_months}m test")
    
    print("\nPhase 2.1 (Rolling Windows) completed successfully.")
    print("Ready to begin Phase 2.2 (Covariance Shrinkage Estimation)")

if __name__ == "__main__":
    main() 