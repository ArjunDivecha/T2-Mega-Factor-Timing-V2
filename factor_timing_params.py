"""
# Factor Timing Parameters
#
# Common command-line parameter handling for all factor timing scripts
# Provides consistent interface across all components of the pipeline
#
# Usage:
#   from factor_timing_params import add_common_args, parse_window_indices
#   
#   def parse_arguments():
#       parser = add_common_args('Script description')
#       # Add script-specific arguments
#       return parser.parse_args()
#
#   args = parse_arguments()
#   window_indices = parse_window_indices(args, rolling_data)
"""

import argparse
import pandas as pd
import os
import pickle
import numpy as np

# Default file paths
ROLLING_WINDOWS_FILE = "rolling_windows.pkl"
SHRINKAGE_RESULTS_FILE = "shrinkage_results.pkl"

def add_common_args(description):
    """
    Adds common command-line arguments to all scripts
    
    Parameters:
    -----------
    description : str
        Description of the script
        
    Returns:
    --------
    parser : ArgumentParser
        Parser with common arguments added
    """
    parser = argparse.ArgumentParser(description=description)
    
    # Window selection group - mutually exclusive
    window_group = parser.add_mutually_exclusive_group()
    
    # Window indices (specific windows)
    window_group.add_argument('--window_indices', type=str,
                        help='Specific window indices to process, comma-separated (e.g., "120,121,122")')
    
    # Date range (converts to window indices internally)
    # Put these in a separate parser group so they can be used together
    date_group = parser.add_argument_group('date range options')
    date_group.add_argument('--start_date', type=str,
                        help='Start date for test period (YYYY-MM-DD)')
    date_group.add_argument('--end_date', type=str,
                        help='End date for test period (YYYY-MM-DD)')
    
    # Special window finding (part of the window selection group)
    window_group.add_argument('--find_2010_window', action='store_true',
                        help='Find and process window around 2010')
    
    # Window limit
    parser.add_argument('--max_windows', type=int, default=5,
                        help='Maximum number of windows to process (default: 5, 0 for all windows)')
    
    # Portfolio limit
    parser.add_argument('--max_portfolios', type=int, default=2000,
                        help='Maximum number of portfolios to analyze (default: 2000)')
    
    return parser

def load_rolling_windows(file_path=ROLLING_WINDOWS_FILE):
    """Load the rolling windows data if needed"""
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return None
        
    with open(file_path, 'rb') as f:
        rolling_data = pickle.load(f)
    
    return rolling_data

def find_windows_by_date_range(rolling_data, start_date, end_date):
    """
    Find window indices that fall within the specified date range
    
    Parameters:
    -----------
    rolling_data : dict
        Rolling windows data
    start_date : str
        Start date in 'YYYY-MM-DD' format
    end_date : str
        End date in 'YYYY-MM-DD' format
        
    Returns:
    --------
    window_indices : list
        List of window indices that fall within the date range
    """
    print(f"Finding windows between {start_date} and {end_date}...")
    
    # Convert string dates to datetime
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    
    # Find windows where test date falls within range
    window_indices = []
    
    for idx, window_date in rolling_data['window_dates'].items():
        test_date = pd.to_datetime(window_date['test_start'])
        if start_dt <= test_date <= end_dt:
            window_indices.append(idx)
    
    window_indices.sort()
    print(f"Found {len(window_indices)} windows in the specified date range")
    return window_indices

def find_window_by_year(window_dates, target_year=2010):
    """Find window index where validation period includes the target year"""
    for idx, dates in window_dates.items():
        val_end_year = pd.to_datetime(dates['val_end']).year
        if val_end_year == target_year:
            return [idx]
    
    # If exact year not found, find closest
    closest_idx = None
    min_diff = float('inf')
    for idx, dates in window_dates.items():
        diff = abs(pd.to_datetime(dates['val_end']).year - target_year)
        if diff < min_diff:
            min_diff = diff
            closest_idx = idx
    
    return [closest_idx] if closest_idx is not None else []

def parse_window_indices(args, rolling_data=None):
    """
    Parse window indices from command-line arguments
    
    Parameters:
    -----------
    args : Namespace
        Command-line arguments
    rolling_data : dict, optional
        Rolling windows data. If None, will be loaded from disk if needed.
        
    Returns:
    --------
    window_indices : list
        List of window indices to process
    """
    # Case 1: Window indices directly specified
    if args.window_indices:
        window_indices = [int(idx) for idx in args.window_indices.split(',')]
        print(f"Processing specific windows: {window_indices}")
        return window_indices
    
    # For the remaining cases, we need rolling_data
    if rolling_data is None:
        rolling_data = load_rolling_windows()
        if rolling_data is None:
            print("Failed to load rolling windows data. Exiting.")
            return []
    
    # Case 2: Find window around 2010
    if args.find_2010_window:
        window_indices = find_window_by_year(rolling_data['window_dates'], 2010)
        if window_indices:
            print(f"Found window {window_indices[0]} with validation period ending closest to 2010")
        else:
            print("Could not find window around 2010")
        return window_indices
    
    # Case 3: Date range specified
    if args.start_date and args.end_date:
        window_indices = find_windows_by_date_range(
            rolling_data, 
            args.start_date, 
            args.end_date
        )
        return window_indices
    
    # Case 4: Use max_windows most recent windows
    max_windows = args.max_windows
    if max_windows == 0:  # 0 means all windows
        window_indices = sorted(rolling_data['window_dates'].keys())
    else:
        window_indices = sorted(rolling_data['window_dates'].keys())[-max_windows:]
    
    print(f"Processing {len(window_indices)} {'most recent ' if max_windows > 0 else ''}windows")
    return window_indices 