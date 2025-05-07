"""
# Factor Timing Complete Pipeline
# 
# This script serves as a unified pipeline for the entire factor timing workflow.
# It can execute all steps in sequence or start from any intermediate point
# in the pipeline, producing identical results to running individual components.
#
# WORKFLOW COMPONENTS:
# 1. Data Preparation: Process raw Excel data (factor_timing_data_prep.py)
# 2. Rolling Windows: Create rolling train/validate/test windows (factor_timing_rolling_windows.py)
# 3. Shrinkage Optimization: Apply L-W shrinkage with fixed lambda (factor_timing_shrinkage.py)
# 4. Weight Extraction: Extract portfolio weights (extract_weights.py)
# 5. Factor Rotation: Convert portfolio weights to factor weights (factor_rotation.py)
# 6. Performance Analysis: Calculate metrics and visualize results
#
# INPUT FILES (depending on starting point):
# - T2 Mega Factor Conditioning Variables.xlsx: Raw data for step 1
# - prepared_data.pkl: Output from step 1, input for step 2
# - rolling_windows.pkl: Output from step 2, input for step 3
# - shrinkage_results.pkl: Output from step 3, input for steps 4 and 5
#
# OUTPUT FILES:
# - prepared_data.pkl: Processed factor returns and timing portfolios
# - rolling_windows.pkl: Train/validate/test split windows
# - shrinkage_results.pkl: Optimization results with portfolio weights
# - unrotated_optimal_weights.xlsx: Extracted portfolio weights
# - rotated_optimal_weights.xlsx: Weights converted to factor form
# - test_results.pkl: Complete performance results
# - factor_timing_results.xlsx: Excel summary of all results
# - Various PDF visualizations
"""

import pandas as pd
import numpy as np
import pickle
import os
import argparse
import time
import matplotlib.pyplot as plt
from datetime import datetime
import sys
import traceback
import importlib.util
import subprocess
from factor_timing_params import add_common_args, parse_window_indices

# Import functions from other modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from factor_timing_shrinkage import run_factor_timing_optimization, calculate_sharpe_ratio
    from factor_rotation import extract_factor_names, rotate_portfolio_weights, apply_long_only_constraint
except ImportError as e:
    print(f"Error importing functions from other modules: {e}")
    print("Make sure all required scripts are in the same directory.")
    sys.exit(1)

# Define input/output files
EXCEL_INPUT_FILE = "T2 Mega Factor Conditioning Variables.xlsx"
PREPARED_DATA_FILE = "prepared_data.pkl"
ROLLING_WINDOWS_FILE = "rolling_windows.pkl"
SHRINKAGE_RESULTS_FILE = "shrinkage_results.pkl"
UNROTATED_WEIGHTS_FILE = "unrotated_optimal_weights.xlsx"
ROTATED_WEIGHTS_FILE = "rotated_optimal_weights.xlsx"
TEST_RESULTS_FILE = "test_results.pkl"
OUTPUT_EXCEL_FILE = "factor_timing_results.xlsx"

def import_module_from_file(module_name, file_path):
    """
    Import a module from a file path
    
    Args:
        module_name (str): Name to give the imported module
        file_path (str): Path to the Python file
        
    Returns:
        module: The imported module object, or None if import fails
    """
    try:
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    except Exception as e:
        print(f"Error importing module from {file_path}: {e}")
        return None

def run_data_preparation(force=False):
    """
    Run the data preparation step (Phase 1)
    
    Args:
        force (bool): Whether to regenerate output even if it exists
        
    Returns:
        bool: True if successful, False otherwise
    """
    if os.path.exists(PREPARED_DATA_FILE) and not force:
        print(f"Prepared data file {PREPARED_DATA_FILE} already exists. Use --force to regenerate.")
        return False
    
    print("\n===== PHASE 1: DATA PREPARATION =====")
    try:
        # Import the module
        data_prep = import_module_from_file("factor_timing_data_prep", "factor_timing_data_prep.py")
        if not data_prep:
            print("Failed to import factor_timing_data_prep.py")
            return False
        
        # Execute the main function
        data_prep.main()
        return True
    except Exception as e:
        print(f"Error running data preparation: {e}")
        traceback.print_exc()
        return False

def run_rolling_windows(force=False):
    """
    Run the rolling windows generation step (Phase 2)
    
    Args:
        force (bool): Whether to regenerate output even if it exists
        
    Returns:
        bool: True if successful, False otherwise
    """
    if not os.path.exists(PREPARED_DATA_FILE):
        print(f"Error: {PREPARED_DATA_FILE} not found. Run data preparation first.")
        return False
        
    if os.path.exists(ROLLING_WINDOWS_FILE) and not force:
        print(f"Rolling windows file {ROLLING_WINDOWS_FILE} already exists. Use --force to regenerate.")
        return False
    
    print("\n===== PHASE 2: ROLLING WINDOWS GENERATION =====")
    try:
        # Import the module
        rolling_windows = import_module_from_file("factor_timing_rolling_windows", "factor_timing_rolling_windows.py")
        if not rolling_windows:
            print("Failed to import factor_timing_rolling_windows.py")
            return False
        
        # Execute the main function
        rolling_windows.main()
        return True
    except Exception as e:
        print(f"Error running rolling windows generation: {e}")
        traceback.print_exc()
        return False

def run_shrinkage_optimization(window_indices=None, max_windows=None, force=False):
    """
    Run the shrinkage optimization step (Phase 3)
    
    Args:
        window_indices (str): Comma-separated window indices to process
        max_windows (int): Maximum number of windows to process
        force (bool): Whether to regenerate output even if it exists
        
    Returns:
        bool: True if successful, False otherwise
    """
    if not os.path.exists(ROLLING_WINDOWS_FILE):
        print(f"Error: {ROLLING_WINDOWS_FILE} not found. Run rolling windows generation first.")
        return False
        
    if os.path.exists(SHRINKAGE_RESULTS_FILE) and not force:
        print(f"Shrinkage results file {SHRINKAGE_RESULTS_FILE} already exists. Use --force to regenerate.")
        return False
    
    print("\n===== PHASE 3: SHRINKAGE OPTIMIZATION =====")
    try:
        # Prepare command line arguments
        cmd = ["python", "factor_timing_shrinkage.py"]
        
        if window_indices:
            cmd.extend(["--window_indices", window_indices])
        
        if max_windows:
            cmd.extend(["--max_windows", str(max_windows)])
        
        if force:
            cmd.append("--force")
        
        # Execute the command
        subprocess.run(cmd, check=True)
        return True
    except Exception as e:
        print(f"Error running shrinkage optimization: {e}")
        traceback.print_exc()
        return False

def run_extract_weights(force=False):
    """
    Run the weight extraction step (Phase 4)
    
    Args:
        force (bool): Whether to regenerate output even if it exists
        
    Returns:
        bool: True if successful, False otherwise
    """
    if not os.path.exists(SHRINKAGE_RESULTS_FILE):
        print(f"Error: {SHRINKAGE_RESULTS_FILE} not found. Run shrinkage optimization first.")
        return False
        
    if os.path.exists(UNROTATED_WEIGHTS_FILE) and not force:
        print(f"Unrotated weights file {UNROTATED_WEIGHTS_FILE} already exists. Use --force to regenerate.")
        return False
    
    print("\n===== PHASE 4: WEIGHT EXTRACTION =====")
    try:
        # Import the module
        extract_weights = import_module_from_file("extract_weights", "extract_weights.py")
        if not extract_weights:
            print("Failed to import extract_weights.py")
            return False
        
        # Execute the main function if it exists
        if hasattr(extract_weights, 'main'):
            extract_weights.main()
        else:
            # Otherwise run the whole script
            subprocess.run(["python", "extract_weights.py"], check=True)
        return True
    except Exception as e:
        print(f"Error running weight extraction: {e}")
        traceback.print_exc()
        return False

def run_factor_rotation(force=False):
    """
    Run the factor rotation step (Phase 5)
    
    Args:
        force (bool): Whether to regenerate output even if it exists
        
    Returns:
        bool: True if successful, False otherwise
    """
    if not os.path.exists(SHRINKAGE_RESULTS_FILE):
        print(f"Error: {SHRINKAGE_RESULTS_FILE} not found. Run shrinkage optimization first.")
        return False
        
    if os.path.exists(ROTATED_WEIGHTS_FILE) and not force:
        print(f"Rotated weights file {ROTATED_WEIGHTS_FILE} already exists. Use --force to regenerate.")
        return False
    
    print("\n===== PHASE 5: FACTOR ROTATION =====")
    try:
        # Import the module
        factor_rotation = import_module_from_file("factor_rotation", "factor_rotation.py")
        if not factor_rotation:
            print("Failed to import factor_rotation.py")
            return False
        
        # Execute the main function if it exists
        if hasattr(factor_rotation, 'main'):
            factor_rotation.main()
        else:
            # Otherwise run the whole script
            subprocess.run(["python", "factor_rotation.py"], check=True)
        return True
    except Exception as e:
        print(f"Error running factor rotation: {e}")
        traceback.print_exc()
        return False

def load_rolling_windows():
    """Load the rolling windows data"""
    print(f"Loading rolling windows data from {ROLLING_WINDOWS_FILE}...")
    
    if not os.path.exists(ROLLING_WINDOWS_FILE):
        print(f"Error: {ROLLING_WINDOWS_FILE} not found.")
        return None
        
    with open(ROLLING_WINDOWS_FILE, 'rb') as f:
        data = pickle.load(f)
    
    print(f"Successfully loaded rolling windows data with {len(data.get('window_dates', {}))} windows")
    return data

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

def calculate_performance_metrics(returns):
    """
    Calculate comprehensive performance metrics for a return series
    
    Parameters:
    -----------
    returns : ndarray
        Array of returns
        
    Returns:
    --------
    metrics : dict
        Dictionary of performance metrics
    """
    # Handle empty or invalid returns
    if len(returns) < 2:
        return {
            'annualized_return': np.nan,
            'annualized_vol': np.nan,
            'sharpe_ratio': np.nan,
            'max_drawdown': np.nan,
            'win_rate': np.nan
        }
    
    # Annualization factor for monthly returns
    annual_factor = 12
    
    # Calculate annualized return
    ann_return = np.mean(returns) * annual_factor
    
    # Calculate annualized volatility
    ann_vol = np.std(returns, ddof=1) * np.sqrt(annual_factor)
    
    # Calculate Sharpe ratio
    sharpe = ann_return / ann_vol if ann_vol > 1e-8 else 0.0
    
    # Calculate win rate (% of positive months)
    win_rate = np.mean(returns > 0) * 100
    
    # Calculate maximum drawdown
    cumulative_returns = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cumulative_returns)
    drawdowns = (cumulative_returns / running_max) - 1
    max_drawdown = np.min(drawdowns) * 100  # Convert to percentage
    
    return {
        'annualized_return': ann_return * 100,  # Convert to percentage
        'annualized_vol': ann_vol * 100,  # Convert to percentage
        'sharpe_ratio': sharpe,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate
    }

def calculate_portfolio_turnover(weights_series):
    """
    Calculate the average portfolio turnover
    
    Parameters:
    -----------
    weights_series : list of ndarrays
        List of weight arrays over time
        
    Returns:
    --------
    turnover : float
        Average one-way turnover
    """
    if len(weights_series) < 2:
        return np.nan
        
    turnovers = []
    for i in range(1, len(weights_series)):
        prev_weights = weights_series[i-1]
        curr_weights = weights_series[i]
        
        # Calculate turnover as the sum of absolute weight changes divided by 2
        turnover = np.sum(np.abs(curr_weights - prev_weights)) / 2
        turnovers.append(turnover)
    
    return np.mean(turnovers) * 100  # Convert to percentage

def create_equal_weight_benchmark(shrinkage_results, rolling_data, window_indices):
    """
    Create equal-weight benchmark results
    
    Parameters:
    -----------
    shrinkage_results : dict
        Results from shrinkage optimization
    rolling_data : dict
        Rolling windows data
    window_indices : list
        List of window indices
        
    Returns:
    --------
    benchmark_results : dict
        Results of equal-weight benchmark
    """
    print("Creating equal-weight benchmark...")
    
    # Initialize results
    benchmark_results = {
        'portfolio_weights': {},
        'test_returns': {}
    }
    
    for i in window_indices:
        # Skip if window not in shrinkage results
        if i not in shrinkage_results['selected_portfolios']:
            continue
        
        # Get selected portfolios
        selected_columns = shrinkage_results['selected_portfolios'][i]
        
        # Create equal weights
        num_portfolios = len(selected_columns)
        eq_weights = np.ones(num_portfolios) / num_portfolios
        
        # Store weights
        benchmark_results['portfolio_weights'][i] = eq_weights
        
        # Get test returns
        # Check if i is a valid index for the testing_windows list
        if 0 <= i < len(rolling_data['testing_windows']):
            testing_window = rolling_data['testing_windows'][i][selected_columns]
            eq_returns = np.dot(eq_weights, testing_window.values.T)
            benchmark_results['test_returns'][i] = eq_returns
    
    return benchmark_results

def run_test_pipeline(rolling_data, window_indices, max_portfolios=2000):
    """
    Run the entire test pipeline for specified windows
    
    Parameters:
    -----------
    rolling_data : dict
        Rolling windows data
    window_indices : list
        List of window indices to process
    max_portfolios : int
        Maximum number of portfolios to include
        
    Returns:
    --------
    test_results : dict
        Complete test results
    """
    print(f"Running complete test pipeline for {len(window_indices)} windows...")
    start_time = time.time()
    
    # 1. Run shrinkage analysis
    print("\n===== PHASE 1: SHRINKAGE OPTIMIZATION =====")
    shrinkage_results = run_factor_timing_optimization(
        rolling_data, 
        window_indices=window_indices, 
        max_portfolios=max_portfolios
    )
    
    # 2. Create equal-weight benchmark
    print("\n===== PHASE 2: CREATING BENCHMARK =====")
    benchmark_results = create_equal_weight_benchmark(
        shrinkage_results,
        rolling_data,
        window_indices
    )
    
    # 3. Perform factor rotation
    print("\n===== PHASE 3: FACTOR ROTATION =====")
    factor_results = process_factor_rotation(shrinkage_results)
    
    # 4. Combine results
    test_results = {
        'window_indices': window_indices,
        'window_dates': {i: rolling_data['window_dates'][i] for i in window_indices},
        'optimal_lambdas': shrinkage_results['optimal_lambdas'],
        'training_sharpe': shrinkage_results['training_sharpe'],
        'validation_sharpe': shrinkage_results['validation_sharpe'],
        'test_sharpe': shrinkage_results['test_sharpe'],
        'test_returns': shrinkage_results['test_returns'],
        'factor_weights': factor_results['factor_weights'],
        'long_only_weights': factor_results['long_only_weights'],
        'factor_names': factor_results['factor_names'],
        'benchmark_results': benchmark_results
    }
    
    # 5. Create time series of returns for cumulative performance plots
    print("\n===== PHASE 4: PREPARING RETURN SERIES =====")
    strategy_returns_series = {}
    benchmark_returns_series = {}
    
    # Sort window indices by date to ensure chronological order
    ordered_indices = sorted(window_indices, 
                           key=lambda idx: pd.to_datetime(rolling_data['window_dates'][idx]['test_start']))
    
    for idx in ordered_indices:
        if idx in test_results['test_returns'] and idx in benchmark_results['test_returns']:
            try:
                date_str = rolling_data['window_dates'][idx]['test_start']
                date = pd.to_datetime(date_str)
                
                # Get returns for this window
                strat_ret = test_results['test_returns'][idx]
                bench_ret = benchmark_results['test_returns'][idx]
                
                # Store in time series dictionary (ensuring we have scalar values)
                if isinstance(strat_ret, np.ndarray) and len(strat_ret) == 1:
                    strat_ret = float(strat_ret[0])
                elif isinstance(strat_ret, list) and len(strat_ret) == 1:
                    strat_ret = float(strat_ret[0])
                
                if isinstance(bench_ret, np.ndarray) and len(bench_ret) == 1:
                    bench_ret = float(bench_ret[0])
                elif isinstance(bench_ret, list) and len(bench_ret) == 1:
                    bench_ret = float(bench_ret[0])
                
                strategy_returns_series[date] = strat_ret
                benchmark_returns_series[date] = bench_ret
                
                print(f"Added return for {date}: Strategy={strat_ret:.4f}, Benchmark={bench_ret:.4f}")
            except Exception as e:
                print(f"Error processing returns for window {idx}: {e}")
    
    # Create DataFrame of returns and store in results
    if strategy_returns_series:
        returns_df = pd.DataFrame({
            'Strategy': pd.Series(strategy_returns_series),
            'Equal-Weight': pd.Series(benchmark_returns_series)
        })
        
        # Ensure the index is sorted chronologically
        returns_df = returns_df.sort_index()
        
        # Store in results dictionary
        test_results['strategy_returns'] = returns_df['Strategy']
        test_results['benchmark_returns'] = returns_df['Equal-Weight']
        
        # Print some statistics
        print(f"Created return series with {len(returns_df)} data points")
        print(f"Date range: {returns_df.index.min()} to {returns_df.index.max()}")
        print(f"Strategy mean monthly return: {returns_df['Strategy'].mean():.4f}")
        print(f"Benchmark mean monthly return: {returns_df['Equal-Weight'].mean():.4f}")
    else:
        print("Warning: Could not create return series - no valid returns data found")
    
    elapsed_time = time.time() - start_time
    print(f"\nTest pipeline completed in {elapsed_time:.2f} seconds")
    
    return test_results

def process_factor_rotation(shrinkage_results):
    """
    Process factor rotation (simplified from factor_rotation.py)
    
    Parameters:
    -----------
    shrinkage_results : dict
        Results from shrinkage optimization
        
    Returns:
    --------
    results : dict
        Results of factor rotation analysis
    """
    print("Processing factor rotation...")
    
    # Get window indices
    window_indices = list(shrinkage_results['portfolio_weights'].keys())
    
    # Initialize results
    results = {
        'window_indices': window_indices,
        'factor_weights': {},
        'long_only_weights': {},
        'factor_names': []
    }
    
    # Collect all unique portfolio names across all windows for consistent mapping
    all_portfolio_names = set()
    for i in window_indices:
        if i in shrinkage_results['selected_portfolios']:
            all_portfolio_names.update(shrinkage_results['selected_portfolios'][i])
    
    print(f"Collected {len(all_portfolio_names)} unique portfolios across all windows")
    
    # Extract factor names from all portfolios to ensure comprehensive mapping
    try:
        portfolio_names_list = list(all_portfolio_names)
        # Debug output to analyze portfolio names
        print(f"Analyzing {len(portfolio_names_list)} portfolio names for factor extraction")
        if portfolio_names_list:
            print(f"Sample portfolio names: {portfolio_names_list[:min(5, len(portfolio_names_list))]}")
            
        factor_names, factor_map = extract_factor_names(portfolio_names_list)
        if factor_names:
            results['factor_names'] = factor_names
            print(f"Successfully extracted {len(factor_names)} factor names: {factor_names[:min(10, len(factor_names))]}")
        else:
            # Fallback: create simple factor names if extraction fails
            print("Using fallback factor name extraction")
            unique_factors = set()
            factor_map = {}
            
            for portfolio in portfolio_names_list:
                # Simple extraction logic - use first component of the name
                parts = portfolio.split('_')
                factor_name = parts[0] if parts else portfolio
                factor_map[portfolio] = factor_name
                unique_factors.add(factor_name)
            
            factor_names = list(unique_factors)
            results['factor_names'] = factor_names
            print(f"Created {len(factor_names)} factor names using fallback method")
    except Exception as e:
        print(f"Error during factor name extraction: {e}")
        # Extremely simple fallback - just number the factors
        factor_names = [f"Factor_{i}" for i in range(len(all_portfolio_names))]
        factor_map = {portfolio: f"Factor_{i}" for i, portfolio in enumerate(all_portfolio_names)}
        results['factor_names'] = factor_names
        print(f"Created {len(factor_names)} generic factor names due to extraction error")

    print(f"Created comprehensive factor map with {len(factor_names)} unique factors")
    
    # Process each window
    for i in window_indices:
        # Skip if no portfolio weights or selected portfolios
        if i not in shrinkage_results['portfolio_weights'] or i not in shrinkage_results['selected_portfolios']:
            continue
        
        # Get portfolio weights for this window
        portfolio_weights = shrinkage_results['portfolio_weights'][i]
        
        # Get selected portfolio names for this window
        portfolio_names = list(shrinkage_results['selected_portfolios'][i])
        
        # Ensure lengths match
        if len(portfolio_weights) != len(portfolio_names):
            portfolio_names = portfolio_names[:len(portfolio_weights)]
        
        # Rotate portfolio weights to factor weights
        factor_weights_dict = rotate_portfolio_weights(factor_map, portfolio_weights, portfolio_names)
        
        # Convert to array
        factor_weights_array = np.zeros(len(factor_names))
        for j, factor in enumerate(factor_names):
            factor_weights_array[j] = factor_weights_dict.get(factor, 0)
        
        # Store factor weights
        results['factor_weights'][i] = factor_weights_array
        
        # Apply long-only constraint
        long_only_weights = apply_long_only_constraint(factor_weights_dict, factor_names)
        results['long_only_weights'][i] = long_only_weights
    
    return results

def create_summary_dataframe(test_results):
    """
    Create summary DataFrame with all key metrics
    
    Parameters:
    -----------
    test_results : dict
        Complete test results
        
    Returns:
    --------
    summary_df : DataFrame
        Summary DataFrame with key metrics
    """
    print("Creating summary DataFrame...")
    
    # Initialize rows
    rows = []
    
    # For each window, create a row with metrics
    for idx in sorted(test_results['window_indices']):
        # Get window dates
        if idx in test_results['window_dates']:
            # Get date information
            dates = test_results['window_dates'][idx]
            train_start = pd.to_datetime(dates['train_start'])
            train_end = pd.to_datetime(dates['train_end'])
            val_start = pd.to_datetime(dates['val_start']) if 'val_start' in dates else None
            val_end = pd.to_datetime(dates['val_end']) if 'val_end' in dates else None
            test_date = pd.to_datetime(dates['test_start'])
            
            # Get performance metrics
            lambda_val = test_results['optimal_lambdas'].get(idx, np.nan)
            train_sharpe = test_results['training_sharpe'].get(idx, np.nan)
            val_sharpe = test_results['validation_sharpe'].get(idx, np.nan)
            test_sharpe = test_results['test_sharpe'].get(idx, np.nan)
            
            # Create row
            row = {
                'Window': idx,
                'Test Date': test_date,
                'Training Period': f"{train_start.strftime('%Y-%m')} to {train_end.strftime('%Y-%m')}",
                'Lambda': lambda_val,
                'Training Sharpe': train_sharpe,
                'Validation Sharpe': val_sharpe,
                'Test Sharpe': test_sharpe
            }
            
            # Add top factor weights
            if idx in test_results['long_only_weights']:
                weights = test_results['long_only_weights'][idx]
                factor_names = test_results['factor_names']
                
                # Get top 5 factors by weight
                factor_weights = [(factor_names[i], weights[i]) for i in range(len(weights))]
                sorted_weights = sorted(factor_weights, key=lambda x: x[1], reverse=True)
                
                for i, (factor, weight) in enumerate(sorted_weights[:5]):
                    row[f"Top Factor {i+1}"] = factor
                    row[f"Weight {i+1}"] = weight
            
            rows.append(row)
    
    # Create DataFrame
    if rows:
        summary_df = pd.DataFrame(rows)
        print(f"Created summary DataFrame with {len(summary_df)} rows")
        return summary_df
    else:
        print("No data available for summary DataFrame")
        return pd.DataFrame()

def plot_metrics(test_results, benchmark_results=None):
    """
    Create plots of key metrics
    
    Parameters:
    -----------
    test_results : dict
        Complete test results
    benchmark_results : dict, optional
        Equal-weight benchmark results
        
    Returns:
    --------
    figures : dict
        Dictionary of matplotlib figures
    """
    print("Creating metric plots...")
    figures = {}
    
    try:
        # Get dates and lambda values
        dates = []
        lambdas = []
        train_sharpes = []
        val_sharpes = []
        test_sharpes = []
        
        for idx in sorted(test_results['window_indices']):
            if idx in test_results['window_dates']:
                try:
                    date_str = test_results['window_dates'][idx]['test_start']
                    date = pd.to_datetime(date_str)
                    
                    lambda_val = test_results['optimal_lambdas'].get(idx, np.nan)
                    train_sharpe = test_results['training_sharpe'].get(idx, np.nan)
                    val_sharpe = test_results['validation_sharpe'].get(idx, np.nan)
                    test_sharpe = test_results['test_sharpe'].get(idx, np.nan)
                    
                    # Only add valid numeric values
                    if not (np.isnan(lambda_val) and np.isnan(train_sharpe) and 
                            np.isnan(val_sharpe) and np.isnan(test_sharpe)):
                        dates.append(date)
                        lambdas.append(lambda_val)
                        train_sharpes.append(train_sharpe)
                        val_sharpes.append(val_sharpe)
                        test_sharpes.append(test_sharpe)
                except Exception as e:
                    print(f"Error processing window {idx}: {e}")
        
        if not dates:
            print("No valid dates available for plots. Creating empty plots with explanatory text.")
            
            # Create empty lambda plot with text
            fig_lambda, ax_lambda = plt.subplots(figsize=(10, 6))
            ax_lambda.text(0.5, 0.5, "Insufficient data for lambda evolution plot", 
                          horizontalalignment='center', verticalalignment='center',
                          transform=ax_lambda.transAxes, fontsize=14)
            ax_lambda.set_title('Optimal Lambda Evolution (No Data)')
            figures['lambda'] = fig_lambda
            fig_lambda.savefig('lambda_plot.pdf')
            plt.close(fig_lambda)
            
            # Create empty Sharpe plot with text
            fig_sharpe, ax_sharpe = plt.subplots(figsize=(10, 6))
            ax_sharpe.text(0.5, 0.5, "Insufficient data for Sharpe ratio plot", 
                          horizontalalignment='center', verticalalignment='center',
                          transform=ax_sharpe.transAxes, fontsize=14)
            ax_sharpe.set_title('Sharpe Ratio Evolution (No Data)')
            figures['sharpe'] = fig_sharpe
            fig_sharpe.savefig('sharpe_plot.pdf')
            plt.close(fig_sharpe)
            
            return figures
        
        # Plot lambda evolution - handle case with one data point
        fig_lambda, ax_lambda = plt.subplots(figsize=(10, 6))
        if len(dates) == 1:
            # For single point, use bar chart instead of line
            ax_lambda.bar([dates[0]], [lambdas[0]], width=30, color='blue')
            ax_lambda.text(dates[0], lambdas[0]*1.05, f"{lambdas[0]:.2f}", 
                          horizontalalignment='center', fontsize=12)
        else:
            ax_lambda.plot(dates, lambdas, marker='o', linestyle='-', color='blue')
        
        ax_lambda.set_title('Optimal Lambda Evolution')
        ax_lambda.set_xlabel('Date')
        ax_lambda.set_ylabel('Lambda')
        ax_lambda.set_yscale('log')
        ax_lambda.grid(True, alpha=0.3)
        figures['lambda'] = fig_lambda
        fig_lambda.savefig('lambda_plot.pdf')
        plt.close(fig_lambda)
        
        # Plot Sharpe ratios - handle case with one data point
        fig_sharpe, ax_sharpe = plt.subplots(figsize=(10, 6))
        if len(dates) == 1:
            # For single point, use bar chart
            bar_width = 30
            ind = np.arange(1)
            bars = ax_sharpe.bar(ind, [train_sharpes[0], val_sharpes[0], test_sharpes[0]], width=0.25)
            ax_sharpe.set_xticks(ind)
            ax_sharpe.set_xticklabels(['Single Window'])
            ax_sharpe.legend(['Training', 'Validation', 'Test'])
            # Add text labels
            for i, bar in enumerate(bars):
                height = bar.get_height()
                if not np.isnan(height):
                    ax_sharpe.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                                  f"{height:.2f}", ha='center', va='bottom')
        else:
            # Filter out NaN values for each line
            valid_train = [(d, s) for d, s in zip(dates, train_sharpes) if not np.isnan(s)]
            valid_val = [(d, s) for d, s in zip(dates, val_sharpes) if not np.isnan(s)]
            valid_test = [(d, s) for d, s in zip(dates, test_sharpes) if not np.isnan(s)]
            
            if valid_train:
                train_x, train_y = zip(*valid_train)
                ax_sharpe.plot(train_x, train_y, marker='o', linestyle='-', label='Training', color='blue')
            if valid_val:
                val_x, val_y = zip(*valid_val)
                ax_sharpe.plot(val_x, val_y, marker='s', linestyle='-', label='Validation', color='green')
            if valid_test:
                test_x, test_y = zip(*valid_test)
                ax_sharpe.plot(test_x, test_y, marker='^', linestyle='-', label='Test', color='red')
        
        ax_sharpe.set_title('Sharpe Ratio Evolution')
        ax_sharpe.set_xlabel('Date')
        ax_sharpe.set_ylabel('Sharpe Ratio')
        ax_sharpe.legend()
        ax_sharpe.grid(True, alpha=0.3)
        figures['sharpe'] = fig_sharpe
        fig_sharpe.savefig('sharpe_plot.pdf')
        plt.close(fig_sharpe)
        
        # Plot cumulative returns - using the directly stored return series
        if 'strategy_returns' in test_results and 'benchmark_returns' in test_results:
            strategy_returns = test_results['strategy_returns']
            benchmark_returns = test_results['benchmark_returns']
            
            if len(strategy_returns) > 0 and len(benchmark_returns) > 0:
                # Create DataFrame of returns
                returns_df = pd.DataFrame({
                    'Strategy': strategy_returns,
                    'Equal-Weight': benchmark_returns
                })
                
                # Check if we have valid numeric data
                if returns_df.shape[0] > 0 and not returns_df.isna().all().all():
                    # Sort index to ensure chronological order
                    returns_df = returns_df.sort_index()
                    
                    # Print data to help with debugging
                    print(f"Return series shape: {returns_df.shape}")
                    print(f"Return series head: \n{returns_df.head()}")
                    
                    # Calculate cumulative returns
                    cum_returns = (1 + returns_df).cumprod()
                    
                    # Plot cumulative returns
                    fig_cumulative, ax_cumulative = plt.subplots(figsize=(12, 6))
                    cum_returns.plot(ax=ax_cumulative)
                    ax_cumulative.set_title('Cumulative Returns')
                    ax_cumulative.set_xlabel('Date')
                    ax_cumulative.set_ylabel('Growth of $1')
                    ax_cumulative.grid(True, alpha=0.3)
                    ax_cumulative.legend()
                    figures['cumulative'] = fig_cumulative
                    fig_cumulative.savefig('cumulative_plot.pdf')
                    plt.close(fig_cumulative)
                    
                    # Calculate drawdowns
                    drawdowns = pd.DataFrame()
                    for col in cum_returns.columns:
                        cum_ret = cum_returns[col]
                        running_max = np.maximum.accumulate(cum_ret)
                        drawdown = (cum_ret / running_max) - 1
                        drawdowns[col] = drawdown
                    
                    # Plot drawdowns
                    fig_drawdown, ax_drawdown = plt.subplots(figsize=(12, 6))
                    drawdowns.plot(ax=ax_drawdown)
                    ax_drawdown.set_title('Drawdowns')
                    ax_drawdown.set_xlabel('Date')
                    ax_drawdown.set_ylabel('Drawdown (%)')
                    ax_drawdown.grid(True, alpha=0.3)
                    ax_drawdown.legend()
                    figures['drawdown'] = fig_drawdown
                    fig_drawdown.savefig('drawdown_plot.pdf')
                    plt.close(fig_drawdown)
                else:
                    print("Warning: No valid numeric data available for cumulative returns plot")
                    # Create placeholder plots
                    for plot_name, title in [('cumulative', 'Cumulative Returns'), 
                                            ('drawdown', 'Drawdowns')]:
                        fig, ax = plt.subplots(figsize=(12, 6))
                        ax.text(0.5, 0.5, f"Insufficient data for {title.lower()} plot", 
                              horizontalalignment='center', verticalalignment='center',
                              transform=ax.transAxes, fontsize=14)
                        ax.set_title(f'{title} (No Data)')
                        figures[plot_name] = fig
                        fig.savefig(f'{plot_name}_plot.pdf')
                        plt.close(fig)
            else:
                print("Warning: Empty return series for strategy or benchmark")
                for plot_name, title in [('cumulative', 'Cumulative Returns'), 
                                        ('drawdown', 'Drawdowns')]:
                    fig, ax = plt.subplots(figsize=(12, 6))
                    ax.text(0.5, 0.5, f"Empty return series for {title.lower()} plot", 
                          horizontalalignment='center', verticalalignment='center',
                          transform=ax.transAxes, fontsize=14)
                    ax.set_title(f'{title} (No Data)')
                    figures[plot_name] = fig
                    fig.savefig(f'{plot_name}_plot.pdf')
                    plt.close(fig)
        else:
            print("Warning: 'strategy_returns' or 'benchmark_returns' not found in test_results")
            for plot_name, title in [('cumulative', 'Cumulative Returns'), 
                                    ('drawdown', 'Drawdowns')]:
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.text(0.5, 0.5, f"Missing return series for {title.lower()} plot", 
                      horizontalalignment='center', verticalalignment='center',
                      transform=ax.transAxes, fontsize=14)
                ax.set_title(f'{title} (No Data)')
                figures[plot_name] = fig
                fig.savefig(f'{plot_name}_plot.pdf')
                plt.close(fig)
        
        print(f"Successfully created {len(figures)} plots")
    except Exception as e:
        print(f"Error generating plots: {e}")
        import traceback
        traceback.print_exc()
        
        # Create basic placeholder plots if an error occurred
        for plot_name, title in [('lambda', 'Optimal Lambda Evolution'), 
                                ('sharpe', 'Sharpe Ratio Evolution'),
                                ('cumulative', 'Cumulative Returns'), 
                                ('drawdown', 'Drawdowns')]:
            try:
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.text(0.5, 0.5, f"Error generating {title.lower()} plot:\n{str(e)}", 
                       horizontalalignment='center', verticalalignment='center',
                       transform=ax.transAxes, fontsize=12)
                ax.set_title(f'{title} (Error)')
                fig.savefig(f'{plot_name}_plot.pdf')
                plt.close(fig)
                figures[plot_name] = fig
            except:
                print(f"Could not even create error placeholder for {plot_name}")
    
    # Return figures for later use
    return figures

def export_test_results(test_results, rolling_data, benchmark_results=None):
    """
    Export test results to Excel and pickle files
    
    Parameters:
    -----------
    test_results : dict
        Complete test results
    rolling_data : dict
        Rolling windows data
    benchmark_results : dict, optional
        Equal-weight benchmark results
    """
    print(f"Exporting test results to {OUTPUT_EXCEL_FILE} and {TEST_RESULTS_FILE}...")
    
    # Save results as pickle
    with open(TEST_RESULTS_FILE, 'wb') as f:
        pickle.dump(test_results, f)
    print(f"Saved complete results to {TEST_RESULTS_FILE}")
    
    # Create summary DataFrame
    print("Creating summary DataFrame...")
    summary_df = create_summary_dataframe(test_results)
    
    # Create factor weights DataFrames
    print("Creating factor weights DataFrames...")
    long_only_rows = []
    long_short_rows = []
    
    for idx in sorted(test_results['window_indices']):
        if idx in test_results['window_dates'] and idx in test_results['long_only_weights']:
            date_str = test_results['window_dates'][idx]['test_start']
            date = pd.to_datetime(date_str)
            
            # Long-only weights
            if idx in test_results['long_only_weights']:
                weights = test_results['long_only_weights'][idx]
                row_dict = {'Date': date}
                for i, factor in enumerate(test_results['factor_names']):
                    row_dict[factor] = weights[i] if i < len(weights) else 0.0
                long_only_rows.append(row_dict)
            
            # Long-short weights
            if idx in test_results['factor_weights']:
                weights = test_results['factor_weights'][idx]
                row_dict = {'Date': date}
                for i, factor in enumerate(test_results['factor_names']):
                    row_dict[factor] = weights[i] if i < len(weights) else 0.0
                long_short_rows.append(row_dict)
    
    # Create DataFrames
    print("Converting weight data to DataFrames...")
    long_only_df = pd.DataFrame(long_only_rows).set_index('Date').sort_index() if long_only_rows else pd.DataFrame()
    long_short_df = pd.DataFrame(long_short_rows).set_index('Date').sort_index() if long_short_rows else pd.DataFrame()
    
    # Calculate comprehensive performance metrics
    print("Calculating performance metrics...")
    # Collect all returns
    strategy_returns = []
    windows = sorted(test_results['test_returns'].keys())
    for idx in windows:
        if idx in test_results['test_returns']:
            strategy_returns.append(test_results['test_returns'][idx])
    
    all_strategy_returns = np.concatenate(strategy_returns) if strategy_returns else np.array([])
    
    # Calculate strategy performance metrics
    print(f"Found {len(all_strategy_returns)} strategy return data points")
    strategy_metrics = calculate_performance_metrics(all_strategy_returns)
    
    # Calculate portfolio turnover
    print("Calculating portfolio turnover...")
    portfolio_weights_series = [test_results['long_only_weights'][idx] for idx in windows 
                               if idx in test_results['long_only_weights']]
    strategy_turnover = calculate_portfolio_turnover(portfolio_weights_series)
    strategy_metrics['turnover'] = strategy_turnover
    
    # Calculate benchmark metrics if available
    print("Calculating benchmark metrics...")
    benchmark_metrics = None
    if benchmark_results:
        benchmark_returns = []
        for idx in windows:
            if idx in benchmark_results['test_returns']:
                benchmark_returns.append(benchmark_results['test_returns'][idx])
        
        all_benchmark_returns = np.concatenate(benchmark_returns) if benchmark_returns else np.array([])
        print(f"Found {len(all_benchmark_returns)} benchmark return data points")
        
        benchmark_metrics = calculate_performance_metrics(all_benchmark_returns)
        
        # Calculate benchmark turnover
        benchmark_weights_series = [benchmark_results['portfolio_weights'][idx] for idx in windows 
                                  if idx in benchmark_results['portfolio_weights']]
        benchmark_turnover = calculate_portfolio_turnover(benchmark_weights_series)
        benchmark_metrics['turnover'] = benchmark_turnover
    
    # Create performance summary DataFrame
    print("Creating performance summary DataFrame...")
    performance_data = {
        'Metric': ['Annualized Return (%)', 'Annualized Volatility (%)', 'Sharpe Ratio', 
                  'Maximum Drawdown (%)', 'Win Rate (%)', 'Turnover (%)'],
        'Strategy': [
            f"{strategy_metrics['annualized_return']:.2f}",
            f"{strategy_metrics['annualized_vol']:.2f}",
            f"{strategy_metrics['sharpe_ratio']:.2f}",
            f"{strategy_metrics['max_drawdown']:.2f}",
            f"{strategy_metrics['win_rate']:.2f}",
            f"{strategy_metrics['turnover']:.2f}"
        ]
    }
    
    if benchmark_metrics:
        performance_data['Equal-Weight'] = [
            f"{benchmark_metrics['annualized_return']:.2f}",
            f"{benchmark_metrics['annualized_vol']:.2f}",
            f"{benchmark_metrics['sharpe_ratio']:.2f}",
            f"{benchmark_metrics['max_drawdown']:.2f}",
            f"{benchmark_metrics['win_rate']:.2f}",
            f"{benchmark_metrics['turnover']:.2f}"
        ]
    
    performance_df = pd.DataFrame(performance_data)
    
    # Create figures for saving separately (not embedding in Excel)
    print("Creating visualizations and saving as separate files...")
    try:
        figures = plot_metrics(test_results, benchmark_results)
        print(f"Created {len(figures)} figures")
        
        # Save image files separately
        for name, fig in figures.items():
            try:
                plot_file = f'{name}_plot.pdf'  # Use PDF format as per requirements
                print(f"Saving figure '{name}' to {plot_file}...")
                fig.savefig(plot_file, dpi=300)
                plt.close(fig)
                print(f"Successfully saved {plot_file}")
            except Exception as e:
                print(f"Error saving figure '{name}': {e}")
    except Exception as e:
        print(f"Error generating plots: {e}")
    
    # Export to Excel - without images
    print("Starting Excel export (without images)...")
    try:
        print(f"Writing to Excel file: {OUTPUT_EXCEL_FILE}")
        with pd.ExcelWriter(OUTPUT_EXCEL_FILE, engine='openpyxl') as writer:
            # Write performance summary sheet
            print("Writing Performance Summary sheet...")
            performance_df.to_excel(writer, sheet_name='Performance Summary', index=False)
            print(f"Successfully wrote Performance Summary sheet with {len(performance_df)} rows")
            
            # Write window summary sheet
            if not summary_df.empty:
                print("Writing Window Details sheet...")
                summary_df.to_excel(writer, sheet_name='Window Details')
                print(f"Successfully wrote Window Details sheet with {len(summary_df)} rows")
            
            # Write factor weights sheets
            if not long_only_df.empty:
                print("Writing Long-Only Weights sheet...")
                long_only_df.to_excel(writer, sheet_name='Long-Only Weights')
                print(f"Successfully wrote Long-Only Weights sheet with {len(long_only_df)} rows")
            
            if not long_short_df.empty:
                print("Writing Long-Short Weights sheet...")
                long_short_df.to_excel(writer, sheet_name='Long-Short Weights')
                print(f"Successfully wrote Long-Short Weights sheet with {len(long_short_df)} rows")
            
            # Add a notes sheet with information about plots
            notes_data = {
                'Plot': ['Lambda Evolution', 'Sharpe Ratio Evolution', 'Cumulative Returns', 'Drawdowns'],
                'File': ['lambda_plot.pdf', 'sharpe_plot.pdf', 'cumulative_plot.pdf', 'drawdown_plot.pdf'],
                'Description': [
                    'Shows the evolution of the optimal lambda parameter over time',
                    'Shows the evolution of training, validation, and test Sharpe ratios over time',
                    'Shows the cumulative returns of the strategy vs. equal-weight benchmark',
                    'Shows the drawdowns of the strategy vs. equal-weight benchmark'
                ]
            }
            notes_df = pd.DataFrame(notes_data)
            notes_df.to_excel(writer, sheet_name='Plots Guide', index=False)
            print("Successfully wrote Plots Guide sheet")
        
        print(f"Successfully saved Excel file to {OUTPUT_EXCEL_FILE}")
    except Exception as e:
        print(f"Error during Excel export: {e}")
        traceback.print_exc()
    
    print("Export process completed.")

def run_performance_analysis(window_indices=None, max_windows=None, force=False):
    """
    Run the performance analysis step (Phase 6)
    
    Args:
        window_indices (str): Comma-separated window indices to process
        max_windows (int): Maximum number of windows to process
        force (bool): Whether to regenerate output even if it exists
        
    Returns:
        bool: True if successful, False otherwise
    """
    if not os.path.exists(ROLLING_WINDOWS_FILE) or not os.path.exists(SHRINKAGE_RESULTS_FILE) or not os.path.exists(ROTATED_WEIGHTS_FILE):
        print("Error: Missing required input files. Run previous steps first.")
        return False
    
    if os.path.exists(TEST_RESULTS_FILE) and os.path.exists(OUTPUT_EXCEL_FILE) and not force:
        print(f"Test results already exist. Use --force to regenerate.")
        return False
    
    print("\n===== PHASE 6: PERFORMANCE ANALYSIS =====")
    
    try:
        # Load rolling windows data
        print("Loading rolling windows data...")
        with open(ROLLING_WINDOWS_FILE, 'rb') as f:
            rolling_data = pickle.load(f)
        
        # Load shrinkage results
        print("Loading shrinkage results...")
        with open(SHRINKAGE_RESULTS_FILE, 'rb') as f:
            shrinkage_results = pickle.load(f)
        
        # Parse window indices if provided as string
        if isinstance(window_indices, str):
            window_indices = [int(idx.strip()) for idx in window_indices.split(',')]
        
        # Determine window indices to process
        if window_indices is None and max_windows is not None:
            window_indices = list(range(min(len(rolling_data['window_dates']), max_windows)))
        elif window_indices is None:
            window_indices = list(range(len(rolling_data['window_dates'])))
        
        print(f"Analyzing performance for {len(window_indices)} windows")
        
        # Create benchmark results
        benchmark_results = create_equal_weight_benchmark(
            shrinkage_results,
            rolling_data,
            window_indices
        )
        
        # Process factor rotation results
        factor_results = process_factor_rotation(shrinkage_results)
        
        # Combine all results
        test_results = {
            'window_indices': window_indices,
            'window_dates': {i: rolling_data['window_dates'][i] for i in window_indices},
            'optimal_lambdas': shrinkage_results['optimal_lambdas'],
            'training_sharpe': shrinkage_results['training_sharpe'],
            'validation_sharpe': shrinkage_results['validation_sharpe'],
            'test_sharpe': shrinkage_results['test_sharpe'],
            'test_returns': shrinkage_results['test_returns'],
            'factor_weights': factor_results['factor_weights'],
            'long_only_weights': factor_results['long_only_weights'],
            'factor_names': factor_results['factor_names'],
            'benchmark_results': benchmark_results
        }
        
        # Create time series of returns for cumulative performance plots
        print("\n===== CREATING RETURN SERIES =====")
        strategy_returns_series = {}
        benchmark_returns_series = {}
        
        # Sort window indices by date to ensure chronological order
        ordered_indices = sorted(window_indices, 
                              key=lambda idx: pd.to_datetime(rolling_data['window_dates'][idx]['test_start']))
        
        for idx in ordered_indices:
            if idx in test_results['test_returns'] and idx in benchmark_results['test_returns']:
                try:
                    date_str = rolling_data['window_dates'][idx]['test_start']
                    date = pd.to_datetime(date_str)
                    
                    # Get returns for this window
                    strat_ret = test_results['test_returns'][idx]
                    bench_ret = benchmark_results['test_returns'][idx]
                    
                    # Store in time series dictionary (ensuring we have scalar values)
                    if isinstance(strat_ret, np.ndarray) and len(strat_ret) == 1:
                        strat_ret = float(strat_ret[0])
                    elif isinstance(strat_ret, list) and len(strat_ret) == 1:
                        strat_ret = float(strat_ret[0])
                    
                    if isinstance(bench_ret, np.ndarray) and len(bench_ret) == 1:
                        bench_ret = float(bench_ret[0])
                    elif isinstance(bench_ret, list) and len(bench_ret) == 1:
                        bench_ret = float(bench_ret[0])
                    
                    strategy_returns_series[date] = strat_ret
                    benchmark_returns_series[date] = bench_ret
                except Exception as e:
                    print(f"Error processing returns for window {idx}: {e}")
        
        # Create DataFrame of returns and store in results
        if strategy_returns_series:
            returns_df = pd.DataFrame({
                'Strategy': pd.Series(strategy_returns_series),
                'Equal-Weight': pd.Series(benchmark_returns_series)
            })
            
            # Ensure the index is sorted chronologically
            returns_df = returns_df.sort_index()
            
            # Store in results dictionary
            test_results['strategy_returns'] = returns_df['Strategy']
            test_results['benchmark_returns'] = returns_df['Equal-Weight']
            
            # Print some statistics
            print(f"Created return series with {len(returns_df)} data points")
            print(f"Date range: {returns_df.index.min()} to {returns_df.index.max()}")
            print(f"Strategy mean monthly return: {returns_df['Strategy'].mean():.4f}")
            print(f"Benchmark mean monthly return: {returns_df['Equal-Weight'].mean():.4f}")
        else:
            print("Warning: Could not create return series - no valid returns data found")
        
        # Generate visualizations
        print("\n===== GENERATING VISUALIZATIONS =====")
        figures = plot_metrics(test_results, benchmark_results)
        
        # Save the figures
        print("Saving figures...")
        for name, fig in figures.items():
            filename = f"{name}.pdf"
            fig.savefig(filename, bbox_inches='tight')
            print(f"  Saved {filename}")
            plt.close(fig)
        
        # Export results to Excel and pickle
        print("\n===== EXPORTING RESULTS =====")
        export_test_results(test_results, rolling_data, benchmark_results)
        
        print("Performance analysis complete.")
        return True
    except Exception as e:
        print(f"Error running performance analysis: {e}")
        traceback.print_exc()
        return False

def parse_arguments():
    """
    Parse command-line arguments for the pipeline
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description='Run the complete factor timing pipeline')
    
    # Pipeline control arguments
    parser.add_argument('--start_phase', type=int, default=1, choices=range(1, 7),
                      help='Starting phase (1=data_prep, 2=rolling_windows, 3=shrinkage, 4=extract_weights, 5=factor_rotation, 6=performance)')
    parser.add_argument('--end_phase', type=int, default=6, choices=range(1, 7),
                      help='Ending phase (default: 6)')
    parser.add_argument('--force', action='store_true',
                      help='Force regeneration of output files even if they exist')
    
    # Add window selection arguments directly
    window_group = parser.add_mutually_exclusive_group()
    window_group.add_argument('--window_indices', type=str,
                      help='Specific window indices to process, comma-separated (e.g., "120,121,122")')
    window_group.add_argument('--find_2010_window', action='store_true',
                      help='Find and process window around 2010')
    
    # Date range options
    date_group = parser.add_argument_group('date range options')
    date_group.add_argument('--start_date', type=str,
                      help='Start date for test period (YYYY-MM-DD)')
    date_group.add_argument('--end_date', type=str,
                      help='End date for test period (YYYY-MM-DD)')
    
    # Other options
    parser.add_argument('--max_windows', type=int, default=5,
                      help='Maximum number of windows to process (default: 5, 0 for all windows)')
    parser.add_argument('--max_portfolios', type=int, default=2000,
                      help='Maximum number of portfolios to analyze (default: 2000)')
    
    return parser.parse_args()

def main():
    """
    Run the complete factor timing pipeline
    """
    args = parse_arguments()
    
    # Start timing
    start_time = time.time()
    
    print("=== FACTOR TIMING COMPLETE PIPELINE ===")
    print(f"Starting at phase {args.start_phase}, ending at phase {args.end_phase}")
    
    # Parse window indices if provided
    window_indices_str = None
    window_indices = None
    
    # Check if any window selection arguments were provided
    has_window_selection = False
    for attr in ['window_indices', 'start_date', 'end_date', 'find_2010_window']:
        if hasattr(args, attr) and getattr(args, attr):
            has_window_selection = True
            break
    
    if has_window_selection:
        # We'll need to load the rolling windows data to parse these
        if os.path.exists(ROLLING_WINDOWS_FILE):
            rolling_data = load_rolling_windows()
            if rolling_data:
                window_indices = parse_window_indices(args, rolling_data)
                if window_indices:
                    window_indices_str = ','.join(map(str, window_indices))
                    print(f"Selected {len(window_indices)} windows: {window_indices_str}")
        else:
            print("Warning: Can't parse window indices without rolling_windows.pkl")
            print("Window selection will be applied starting from Phase 3")
    
    # Phase 1: Data Preparation
    if args.start_phase <= 1 and args.end_phase >= 1:
        if not run_data_preparation(force=args.force):
            if args.start_phase == 1:
                print("Failed to complete data preparation. Exiting.")
                return
    
    # Phase 2: Rolling Windows
    if args.start_phase <= 2 and args.end_phase >= 2:
        if not run_rolling_windows(force=args.force):
            if args.start_phase == 2:
                print("Failed to complete rolling windows generation. Exiting.")
                return
    
    # Phase 3: Shrinkage Optimization
    if args.start_phase <= 3 and args.end_phase >= 3:
        if not run_shrinkage_optimization(
            window_indices=window_indices_str, 
            max_windows=args.max_windows,
            force=args.force
        ):
            if args.start_phase == 3:
                print("Failed to complete shrinkage optimization. Exiting.")
                return
    
    # Phase 4: Extract Weights
    if args.start_phase <= 4 and args.end_phase >= 4:
        if not run_extract_weights(force=args.force):
            if args.start_phase == 4:
                print("Failed to complete weight extraction. Exiting.")
                return
    
    # Phase 5: Factor Rotation
    if args.start_phase <= 5 and args.end_phase >= 5:
        if not run_factor_rotation(force=args.force):
            if args.start_phase == 5:
                print("Failed to complete factor rotation. Exiting.")
                return
    
    # Phase 6: Performance Analysis
    if args.start_phase <= 6 and args.end_phase >= 6:
        if not run_performance_analysis(
            window_indices=window_indices,
            max_windows=args.max_windows,
            force=args.force
        ):
            if args.start_phase == 6:
                print("Failed to complete performance analysis. Exiting.")
                return
    
    # Calculate total elapsed time
    elapsed_time = time.time() - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print(f"\n=== PIPELINE EXECUTION COMPLETE ===")
    if hours > 0:
        print(f"Total elapsed time: {int(hours)}h {int(minutes)}m {seconds:.1f}s")
    elif minutes > 0:
        print(f"Total elapsed time: {int(minutes)}m {seconds:.1f}s")
    else:
        print(f"Total elapsed time: {seconds:.1f}s")

if __name__ == "__main__":
    main()