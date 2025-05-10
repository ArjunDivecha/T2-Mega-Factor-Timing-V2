"""
# Run Factor Timing Test
# 
# INPUT FILES:
# - rolling_windows.pkl: Dictionary containing all rolling windows data
#
# OUTPUT FILES:
# - factor_weights.xlsx: Summary Excel file with factor weights and performance 
# - test_results.pkl: Complete results of the test
#
# This script provides an end-to-end test of the factor timing methodology
# for a specific date range, producing a comprehensive summary report.
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
from factor_timing_params import add_common_args, parse_window_indices

# Import functions from other modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from Step_3_shrinkage import run_factor_timing_optimization, calculate_sharpe_ratio
    from Step_5_factor_rotation import extract_factor_names, rotate_portfolio_weights, apply_long_only_constraint
except ImportError as e:
    print(f"Error importing functions from other modules: {e}")
    print("Make sure all required scripts are in the same directory.")
    sys.exit(1)

# Define input/output files
ROLLING_WINDOWS_FILE = "rolling_windows.pkl"
SHRINKAGE_RESULTS_FILE = "shrinkage_results.pkl"
ROTATED_WEIGHTS_FILE = "rotated_optimal_weights.xlsx"
OUTPUT_EXCEL_FILE = "factor_timing_results.xlsx"
OUTPUT_PICKLE_FILE = "test_results.pkl"

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

# Equal-weighted benchmark analysis has been removed as requested

def run_test_pipeline(rolling_data, window_indices, max_portfolios=2000, run_shrinkage=False):
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
    run_shrinkage : bool
        Whether to run shrinkage optimization even if results file exists
        
    Returns:
    --------
    test_results : dict
        Complete test results
    """
    print(f"Running complete test pipeline for {len(window_indices)} windows...")
    start_time = time.time()
    
    # 1. Check if shrinkage results exist or if we need to rerun
    print("\n===== PHASE 1: SHRINKAGE OPTIMIZATION =====")
    if os.path.exists(SHRINKAGE_RESULTS_FILE) and not run_shrinkage:
        print(f"Loading existing shrinkage results from {SHRINKAGE_RESULTS_FILE}...")
        try:
            with open(SHRINKAGE_RESULTS_FILE, 'rb') as f:
                shrinkage_results = pickle.load(f)
            print(f"Successfully loaded shrinkage results for {len(shrinkage_results.get('window_indices', []))} windows")
            
            # Filter shrinkage results to only include the requested window indices
            if 'window_indices' in shrinkage_results and set(window_indices).issubset(set(shrinkage_results['window_indices'])):
                print(f"Filtering shrinkage results to {len(window_indices)} requested windows")
                # No need to filter if all windows are requested
            else:
                print(f"Warning: Not all requested windows found in shrinkage results.")
                print(f"Running shrinkage optimization for {len(window_indices)} windows...")
                shrinkage_results = run_factor_timing_optimization(
                    rolling_data, 
                    window_indices=window_indices, 
                    max_portfolios=max_portfolios
                )
        except Exception as e:
            print(f"Error loading shrinkage results: {e}")
            print(f"Running shrinkage optimization for {len(window_indices)} windows...")
            shrinkage_results = run_factor_timing_optimization(
                rolling_data, 
                window_indices=window_indices, 
                max_portfolios=max_portfolios
            )
    else:
        if run_shrinkage:
            print(f"--run_shrinkage flag set, running shrinkage optimization...")
        else:
            print(f"Shrinkage results file not found, running shrinkage optimization...")
        
        shrinkage_results = run_factor_timing_optimization(
            rolling_data, 
            window_indices=window_indices, 
            max_portfolios=max_portfolios
        )
    
    # Equal-weighted benchmark analysis has been removed as requested
    print("\n===== PHASE 2: SKIPPING BENCHMARK (REMOVED) =====")
    benchmark_results = None
    
    # 3. Perform factor rotation
    print("\n===== PHASE 3: FACTOR ROTATION =====")
    factor_results = process_factor_rotation(shrinkage_results)
    
    # 4. Combine results
    test_results = {
        'window_indices': window_indices,
        'window_dates': {i: rolling_data['window_dates'][i] for i in window_indices},
        'optimal_lambdas': shrinkage_results['optimal_lambdas'],
        'training_sharpe': shrinkage_results['training_sharpe'],
        'test_sharpe': shrinkage_results['test_sharpe'],
        'test_returns': shrinkage_results['test_returns'],
        'factor_weights': factor_results['factor_weights'],
        'long_only_weights': factor_results['long_only_weights'],
        'factor_names': factor_results['factor_names'],
        'benchmark_results': benchmark_results
    }
    
    elapsed_time = time.time() - start_time
    print(f"\nTest pipeline completed in {elapsed_time:.2f} seconds")
    
    return test_results

def process_factor_rotation(shrinkage_results):
    """
    Process factor rotation using the rotated weights from rotated_optimal_weights.xlsx
    
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
    
    # Load rotated weights from Excel file
    print(f"Loading rotated weights from {ROTATED_WEIGHTS_FILE}...")
    if not os.path.exists(ROTATED_WEIGHTS_FILE):
        print(f"Error: {ROTATED_WEIGHTS_FILE} not found. Please run Step_5_factor_rotation.py first.")
        print("Falling back to internal rotation logic...")
        return _process_factor_rotation_fallback(shrinkage_results)
    
    try:
        # Load the rotated weights from Excel
        rotated_df = pd.read_excel(ROTATED_WEIGHTS_FILE)
        
        # Extract factor names from the 'Factor' column
        factor_names = rotated_df['Factor'].tolist()
        results['factor_names'] = factor_names
        print(f"Loaded {len(factor_names)} factors from {ROTATED_WEIGHTS_FILE}")
        
        # Create a mapping from date to window index
        date_to_window = {}
        for i in window_indices:
            if i in shrinkage_results['window_dates'] and 'test_end' in shrinkage_results['window_dates'][i]:
                test_end_date = shrinkage_results['window_dates'][i]['test_end']
                date_str = test_end_date.strftime('%Y-%m-%d')
                date_to_window[date_str] = i
        
        # Process each window
        for date_str, window_idx in date_to_window.items():
            if date_str in rotated_df.columns:
                # Extract weights for this date
                factor_weights_dict = {}
                for j, factor in enumerate(factor_names):
                    weight = rotated_df.loc[j, date_str]
                    factor_weights_dict[factor] = weight
                
                # Convert to array
                factor_weights_array = np.array([factor_weights_dict.get(factor, 0) for factor in factor_names])
                
                # Store factor weights
                results['factor_weights'][window_idx] = factor_weights_array
                
                # Apply long-only constraint
                long_only_weights = apply_long_only_constraint(factor_weights_dict, factor_names)
                results['long_only_weights'][window_idx] = long_only_weights
            else:
                print(f"Warning: Date {date_str} not found in rotated weights file. Skipping window {window_idx}.")
        
        print(f"Successfully processed {len(results['factor_weights'])} windows with rotated weights")
        return results
        
    except Exception as e:
        print(f"Error loading rotated weights: {e}")
        print("Falling back to internal rotation logic...")
        return _process_factor_rotation_fallback(shrinkage_results)


def _process_factor_rotation_fallback(shrinkage_results):
    """
    Fallback function for factor rotation if the rotated weights file is not available
    
    Parameters:
    -----------
    shrinkage_results : dict
        Results from shrinkage optimization
        
    Returns:
    --------
    results : dict
        Results of factor rotation analysis
    """
    print("Using fallback factor rotation logic...")
    
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
            # Get date info
            window_date = test_results['window_dates'].get(idx, {})
            train_start = window_date.get('train_start', pd.NaT)
            train_end = window_date.get('train_end', pd.NaT)
            test_start = window_date.get('test_start', pd.NaT)
            test_end = window_date.get('test_end', pd.NaT)
            test_date = pd.to_datetime(test_start)
            
            # Get performance metrics
            lambda_val = test_results['optimal_lambdas'].get(idx, np.nan)
            train_sharpe = test_results['training_sharpe'].get(idx, np.nan)
            test_sharpe = test_results['test_sharpe'].get(idx, np.nan)
            
            # Create row
            row = {
                'Window': idx,
                'Test Date': test_date,
                'Training Period': f"{train_start.strftime('%Y-%m')} to {train_end.strftime('%Y-%m')}",
                'Lambda': lambda_val,
                'Training Sharpe': train_sharpe,
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

def plot_metrics(test_results):
    """
    Create plots of key metrics
    
    Parameters:
    -----------
    test_results : dict
        Complete test results
        
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
                    test_sharpe = test_results['test_sharpe'].get(idx, np.nan)
                    
                    # Only add valid numeric values
                    if not (np.isnan(lambda_val) and np.isnan(train_sharpe) and np.isnan(test_sharpe)):
                        dates.append(date)
                        lambdas.append(lambda_val)
                        train_sharpes.append(train_sharpe)
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
            bars = ax_sharpe.bar(ind, [train_sharpes[0], test_sharpes[0]], width=0.25)
            ax_sharpe.set_xticks(ind)
            ax_sharpe.set_xticklabels(['Single Window'])
            ax_sharpe.legend(['Training', 'Test'])
            # Add text labels
            for i, bar in enumerate(bars):
                height = bar.get_height()
                if not np.isnan(height):
                    ax_sharpe.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                                  f"{height:.2f}", ha='center', va='bottom')
        else:
            # Filter out NaN values for each line
            valid_train = [(d, s) for d, s in zip(dates, train_sharpes) if not np.isnan(s)]
            valid_test = [(d, s) for d, s in zip(dates, test_sharpes) if not np.isnan(s)]
            
            if valid_train:
                train_x, train_y = zip(*valid_train)
                ax_sharpe.plot(train_x, train_y, marker='o', linestyle='-', label='Training', color='blue')
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
        
        print(f"Successfully created {len(figures)} plots")
    except Exception as e:
        print(f"Error generating plots: {e}")
        import traceback
        traceback.print_exc()
        
        # Create basic placeholder plots if an error occurred
        for plot_name, title in [('lambda', 'Optimal Lambda Evolution'), 
                                ('sharpe', 'Sharpe Ratio Evolution')]:
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

def export_test_results(test_results, rolling_data):
    """
    Export test results to Excel and pickle files
    
    Parameters:
    -----------
    test_results : dict
        Complete test results
    rolling_data : dict
        Rolling windows data
    """
    print(f"Exporting test results to {OUTPUT_EXCEL_FILE} and {OUTPUT_PICKLE_FILE}...")
    
    # Save results as pickle
    with open(OUTPUT_PICKLE_FILE, 'wb') as f:
        pickle.dump(test_results, f)
    print(f"Saved complete results to {OUTPUT_PICKLE_FILE}")
    
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
    
    performance_df = pd.DataFrame(performance_data)
    
    # Create figures for saving separately (not embedding in Excel)
    print("Creating visualizations and saving as separate files...")
    try:
        figures = plot_metrics(test_results)
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
                    'Shows the evolution of training and test Sharpe ratios over time',
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

def parse_arguments():
    """Parse command-line arguments"""
    parser = add_common_args('Run factor timing test pipeline')
    
    # Additional script-specific arguments
    parser.add_argument('--run_shrinkage', action='store_true',
                        help='Run shrinkage optimization even if results file exists')
    parser.add_argument('--export_excel', action='store_true',
                        help='Export results to Excel (default: True if not already exists)')
    parser.add_argument('--export_plots', action='store_true',
                        help='Export plots to PDF files (default: True)')
    
    # Override the default max_windows to 0 (process all windows)
    parser.set_defaults(max_windows=0)
    
    return parser.parse_args()

def main():
    """Run factor timing test pipeline"""
    # Parse command-line arguments
    args = parse_arguments()
    
    # Start timing
    start_time = time.time()
    print("===== PHASE 1: INITIALIZATION =====")
    
    # Load rolling windows data
    print("Loading rolling windows data...")
    with open(ROLLING_WINDOWS_FILE, 'rb') as f:
        rolling_data = pickle.load(f)
    
    # Get window indices to process
    window_indices = parse_window_indices(args, rolling_data)
    if not window_indices:
        print("No windows to process. Exiting.")
        return
    
    # Run test pipeline
    test_results = run_test_pipeline(
        rolling_data, 
        window_indices,
        max_portfolios=args.max_portfolios,
        run_shrinkage=args.run_shrinkage
    )
    
    # Export results
    export_test_results(test_results, rolling_data)

if __name__ == "__main__":
    main()