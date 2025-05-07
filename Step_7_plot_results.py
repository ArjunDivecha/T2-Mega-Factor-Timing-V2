"""
# Plot Factor Results
#
# INPUT FILES:
# - test_results.pkl: Pre-computed factor timing results from run_factor_timing_test.py
# - rolling_windows.pkl: Rolling window data for date mapping (optional)
#
# OUTPUT FILES:
# - lambda_plot.pdf: Evolution of lambda values over time
# - sharpe_plot.pdf: Evolution of Sharpe ratios over time
# - cumulative_plot.pdf: Cumulative returns comparison
# - drawdown_plot.pdf: Drawdown comparison
# - factor_weights_long_short.pdf: Long-short factor weights over time
# - factor_weights_with_long_only_constraint.pdf: Long-only factor weights over time
#
# This script focuses solely on visualization of pre-computed results,
# allowing for rapid iteration of plots without rerunning the entire pipeline.
"""

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import time
import datetime
from collections import defaultdict
from factor_timing_params import add_common_args, parse_window_indices, load_rolling_windows

# Constants
TEST_RESULTS_FILE = "test_results.pkl"

def parse_arguments():
    """Parse command-line arguments"""
    parser = add_common_args('Generate plots from factor timing results')
    
    # Additional script-specific arguments
    parser.add_argument('--input_file', type=str, default=TEST_RESULTS_FILE,
                       help=f'Input results file path (default: {TEST_RESULTS_FILE})')
    parser.add_argument('--plot_types', type=str, default="all",
                       help='Types of plots to generate, comma-separated (lambda,sharpe,cumulative,drawdown,factor_weights)')
    parser.add_argument('--output_dir', type=str, default=".",
                       help='Directory to save plots (default: current directory)')
    
    return parser.parse_args()

def plot_lambda_evolution(test_results, output_dir='.'):
    """Plot evolution of lambda values over time"""
    print("Plotting lambda evolution...")
    
    # Check if we have lambda values to plot
    if 'optimal_lambdas' not in test_results or not test_results['optimal_lambdas']:
        print("No lambda values found in results. Skipping plot.")
        return
    
    # Get dates and lambda values
    dates = []
    lambdas = []
    
    for idx in sorted(test_results['window_indices']):
        if idx in test_results['window_dates'] and idx in test_results['optimal_lambdas']:
            date = pd.to_datetime(test_results['window_dates'][idx]['test_start'])
            dates.append(date)
            lambdas.append(test_results['optimal_lambdas'][idx])
    
    if not dates:
        print("No valid dates and lambda values found. Skipping plot.")
        return
    
    # Create plot
    plt.figure(figsize=(12, 6))
    plt.plot(dates, lambdas, 'o-', linewidth=2, markersize=8)
    plt.xlabel('Test Start Date')
    plt.ylabel('Optimal Lambda Value')
    plt.title('Evolution of Optimal Lambda Values Over Time')
    plt.grid(True)
    
    # Format x-axis as dates
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(mdates.YearLocator())
    plt.gcf().autofmt_xdate()
    
    # Save plot
    output_file = os.path.join(output_dir, 'lambda_plot.pdf')
    plt.savefig(output_file, bbox_inches='tight')
    plt.close()
    print(f"Saved lambda plot to {output_file}")

def plot_sharpe_ratios(test_results, output_dir='.'):
    """Plot Sharpe ratios over time"""
    print("Plotting Sharpe ratios...")
    
    # Check several potential keys for Sharpe ratios
    sharpe_data = None
    if 'sharpe_ratios' in test_results and test_results['sharpe_ratios']:
        sharpe_data = test_results['sharpe_ratios']
    elif all(key in test_results for key in ['training_sharpe', 'validation_sharpe', 'test_sharpe']):
        # Create compatible structure from individual Sharpe ratio dictionaries
        sharpe_data = {}
        for idx in test_results['window_indices']:
            if idx in test_results['training_sharpe'] and idx in test_results['validation_sharpe'] and idx in test_results['test_sharpe']:
                sharpe_data[idx] = {
                    'train': test_results['training_sharpe'].get(idx, np.nan),
                    'val': test_results['validation_sharpe'].get(idx, np.nan),
                    'test': test_results['test_sharpe'].get(idx, np.nan)
                }
    
    if not sharpe_data:
        print("No Sharpe ratios found in results. Skipping plot.")
        return
    
    # Get dates and Sharpe ratios
    dates = []
    train_sharpes = []
    val_sharpes = []
    test_sharpes = []
    
    for idx in sorted(test_results['window_indices']):
        if idx in test_results['window_dates'] and idx in sharpe_data:
            date = pd.to_datetime(test_results['window_dates'][idx]['test_start'])
            dates.append(date)
            
            if isinstance(sharpe_data[idx], dict):
                # If sharpe_data[idx] is a dictionary with 'train', 'val', 'test' keys
                train_sharpes.append(sharpe_data[idx].get('train', np.nan))
                val_sharpes.append(sharpe_data[idx].get('val', np.nan))
                test_sharpes.append(sharpe_data[idx].get('test', np.nan))
            else:
                # If sharpe_data[idx] is a scalar value (assume it's the test Sharpe)
                train_sharpes.append(np.nan)  # We don't have training Sharpe
                val_sharpes.append(np.nan)    # We don't have validation Sharpe
                test_sharpes.append(sharpe_data[idx])
    
    if not dates:
        print("No valid dates and Sharpe ratios found. Skipping plot.")
        return
    
    # Create plot
    plt.figure(figsize=(12, 6))
    
    if any(not np.isnan(s) for s in train_sharpes):
        plt.plot(dates, train_sharpes, 'o-', label='Training', linewidth=2, markersize=6)
    
    if any(not np.isnan(s) for s in val_sharpes):
        plt.plot(dates, val_sharpes, 's-', label='Validation', linewidth=2, markersize=6)
    
    if any(not np.isnan(s) for s in test_sharpes):
        plt.plot(dates, test_sharpes, '^-', label='Test', linewidth=2, markersize=8)
    
    plt.xlabel('Test Start Date')
    plt.ylabel('Sharpe Ratio')
    plt.title('Evolution of Sharpe Ratios Over Time')
    plt.grid(True)
    plt.legend()
    
    # Format x-axis as dates
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(mdates.YearLocator())
    plt.gcf().autofmt_xdate()
    
    # Add zero line
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    # Save plot
    output_file = os.path.join(output_dir, 'sharpe_plot.pdf')
    plt.savefig(output_file, bbox_inches='tight')
    plt.close()
    print(f"Saved Sharpe ratio plot to {output_file}")

def plot_cumulative_returns(test_results, output_dir='.'):
    """Plot cumulative returns"""
    print("Plotting cumulative returns...")
    
    # Check for returns data in several possible formats
    strategy_data = []
    
    # Check for 'strategy_returns' dictionary with 'date' and 'return' keys
    if 'strategy_returns' in test_results and test_results['strategy_returns']:
        for idx in sorted(test_results['strategy_returns'].keys()):
            try:
                date_str = test_results['strategy_returns'][idx]['date']
                value = test_results['strategy_returns'][idx]['return']
                strategy_data.append({'date': pd.to_datetime(date_str), 'return': value})
            except (KeyError, TypeError):
                pass
    
    # Check for 'test_returns' dictionary with return arrays for each window
    if not strategy_data and 'test_returns' in test_results and test_results['test_returns']:
        for idx in sorted(test_results['test_returns'].keys()):
            if idx in test_results['window_dates']:
                try:
                    date_str = test_results['window_dates'][idx]['test_start']
                    date = pd.to_datetime(date_str)
                    ret_val = test_results['test_returns'][idx]
                    
                    # Handle different return formats (single value or array)
                    if isinstance(ret_val, (list, np.ndarray)):
                        if len(ret_val) > 0:
                            ret_val = float(np.mean(ret_val))  # Use mean if it's an array
                        else:
                            continue  # Skip if empty array
                    
                    strategy_data.append({'date': date, 'return': ret_val})
                except (KeyError, TypeError, ValueError):
                    pass
    
    if not strategy_data:
        print("No valid strategy return data found. Skipping plot.")
        return
    
    strategy_df = pd.DataFrame(strategy_data)
    strategy_df = strategy_df.sort_values('date')
    
    # Calculate cumulative returns for strategy
    strategy_df['cum_return'] = (1 + strategy_df['return']).cumprod() - 1
    
    # Create plot
    plt.figure(figsize=(12, 6))
    plt.plot(strategy_df['date'], strategy_df['cum_return'] * 100, 'b-', 
             linewidth=2, label='Factor Timing Strategy')
    
    # Extract benchmark returns if available
    benchmark_data = []
    
    # Check for 'benchmark_returns' dictionary with 'date' and 'return' keys
    if 'benchmark_returns' in test_results and test_results['benchmark_returns']:
        for idx in sorted(test_results['benchmark_returns'].keys()):
            try:
                date_str = test_results['benchmark_returns'][idx]['date']
                value = test_results['benchmark_returns'][idx]['return']
                benchmark_data.append({'date': pd.to_datetime(date_str), 'return': value})
            except (KeyError, TypeError):
                pass
    
    # Check for 'benchmark_results' dictionary with 'test_returns' key
    if not benchmark_data and 'benchmark_results' in test_results and test_results["benchmark_results"] is not None and "test_returns" in test_results["benchmark_results"]:
        bench_returns = test_results['benchmark_results']['test_returns']
        for idx in sorted(bench_returns.keys()):
            if idx in test_results['window_dates']:
                try:
                    date_str = test_results['window_dates'][idx]['test_start']
                    date = pd.to_datetime(date_str)
                    ret_val = bench_returns[idx]
                    
                    # Handle different return formats (single value or array)
                    if isinstance(ret_val, (list, np.ndarray)):
                        if len(ret_val) > 0:
                            ret_val = float(np.mean(ret_val))  # Use mean if it's an array
                        else:
                            continue  # Skip if empty array
                    
                    benchmark_data.append({'date': date, 'return': ret_val})
                except (KeyError, TypeError, ValueError):
                    pass
    
    # Add benchmark if available
    if benchmark_data:
        benchmark_df = pd.DataFrame(benchmark_data)
        benchmark_df = benchmark_df.sort_values('date')
        benchmark_df['cum_return'] = (1 + benchmark_df['return']).cumprod() - 1
        
        plt.plot(benchmark_df['date'], benchmark_df['cum_return'] * 100, 'r--', 
                 linewidth=2, label='Equal-Weight Benchmark')
    
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return (%)')
    plt.title('Cumulative Returns')
    plt.grid(True)
    plt.legend()
    
    # Format x-axis as dates
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(mdates.YearLocator())
    plt.gcf().autofmt_xdate()
    
    # Add zero line
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    # Save plot
    output_file = os.path.join(output_dir, 'cumulative_plot.pdf')
    plt.savefig(output_file, bbox_inches='tight')
    plt.close()
    print(f"Saved cumulative returns plot to {output_file}")

def plot_drawdowns(test_results, output_dir='.'):
    """Plot drawdowns over time"""
    print("Plotting drawdowns...")
    
    # Check for returns data in several possible formats
    strategy_data = []
    
    # Check for 'strategy_returns' dictionary with 'date' and 'return' keys
    if 'strategy_returns' in test_results and test_results['strategy_returns']:
        for idx in sorted(test_results['strategy_returns'].keys()):
            try:
                date_str = test_results['strategy_returns'][idx]['date']
                value = test_results['strategy_returns'][idx]['return']
                strategy_data.append({'date': pd.to_datetime(date_str), 'return': value})
            except (KeyError, TypeError):
                pass
    
    # Check for 'test_returns' dictionary with return arrays for each window
    if not strategy_data and 'test_returns' in test_results and test_results['test_returns']:
        for idx in sorted(test_results['test_returns'].keys()):
            if idx in test_results['window_dates']:
                try:
                    date_str = test_results['window_dates'][idx]['test_start']
                    date = pd.to_datetime(date_str)
                    ret_val = test_results['test_returns'][idx]
                    
                    # Handle different return formats (single value or array)
                    if isinstance(ret_val, (list, np.ndarray)):
                        if len(ret_val) > 0:
                            ret_val = float(np.mean(ret_val))  # Use mean if it's an array
                        else:
                            continue  # Skip if empty array
                    
                    strategy_data.append({'date': date, 'return': ret_val})
                except (KeyError, TypeError, ValueError):
                    pass
    
    if not strategy_data:
        print("No valid strategy return data found. Skipping plot.")
        return
    
    strategy_df = pd.DataFrame(strategy_data)
    strategy_df = strategy_df.sort_values('date')
    
    # Calculate cumulative returns and drawdowns for strategy
    strategy_df['cum_return'] = (1 + strategy_df['return']).cumprod()
    strategy_df['peak'] = strategy_df['cum_return'].cummax()
    strategy_df['drawdown'] = (strategy_df['cum_return'] / strategy_df['peak'] - 1) * 100
    
    # Create plot
    plt.figure(figsize=(12, 6))
    plt.plot(strategy_df['date'], strategy_df['drawdown'], 'b-', 
             linewidth=2, label='Factor Timing Strategy')
    
    # Extract benchmark returns if available
    benchmark_data = []
    
    # Check for 'benchmark_returns' dictionary with 'date' and 'return' keys
    if 'benchmark_returns' in test_results and test_results['benchmark_returns']:
        for idx in sorted(test_results['benchmark_returns'].keys()):
            try:
                date_str = test_results['benchmark_returns'][idx]['date']
                value = test_results['benchmark_returns'][idx]['return']
                benchmark_data.append({'date': pd.to_datetime(date_str), 'return': value})
            except (KeyError, TypeError):
                pass
    
    # Check for 'benchmark_results' dictionary with 'test_returns' key
    if not benchmark_data and 'benchmark_results' in test_results and test_results["benchmark_results"] is not None and "test_returns" in test_results["benchmark_results"]:
        bench_returns = test_results['benchmark_results']['test_returns']
        for idx in sorted(bench_returns.keys()):
            if idx in test_results['window_dates']:
                try:
                    date_str = test_results['window_dates'][idx]['test_start']
                    date = pd.to_datetime(date_str)
                    ret_val = bench_returns[idx]
                    
                    # Handle different return formats (single value or array)
                    if isinstance(ret_val, (list, np.ndarray)):
                        if len(ret_val) > 0:
                            ret_val = float(np.mean(ret_val))  # Use mean if it's an array
                        else:
                            continue  # Skip if empty array
                    
                    benchmark_data.append({'date': date, 'return': ret_val})
                except (KeyError, TypeError, ValueError):
                    pass
    
    # Add benchmark if available
    if benchmark_data:
        benchmark_df = pd.DataFrame(benchmark_data)
        benchmark_df = benchmark_df.sort_values('date')
        benchmark_df['cum_return'] = (1 + benchmark_df['return']).cumprod()
        benchmark_df['peak'] = benchmark_df['cum_return'].cummax()
        benchmark_df['drawdown'] = (benchmark_df['cum_return'] / benchmark_df['peak'] - 1) * 100
        
        plt.plot(benchmark_df['date'], benchmark_df['drawdown'], 'r--', 
                 linewidth=2, label='Equal-Weight Benchmark')
    
    plt.xlabel('Date')
    plt.ylabel('Drawdown (%)')
    plt.title('Drawdowns Over Time')
    plt.grid(True)
    plt.legend()
    
    # Format x-axis as dates
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(mdates.YearLocator())
    plt.gcf().autofmt_xdate()
    
    # Save plot
    output_file = os.path.join(output_dir, 'drawdown_plot.pdf')
    plt.savefig(output_file, bbox_inches='tight')
    plt.close()
    print(f"Saved drawdown plot to {output_file}")

def plot_factor_weights(test_results, long_only=True, output_dir='.'):
    """Plot factor weights over time"""
    weight_type = 'long_only_weights' if long_only else 'factor_weights'
    print(f"Plotting {weight_type} over time...")
    
    # Check if we have factor weights to plot
    if weight_type not in test_results or not test_results[weight_type]:
        print(f"No {weight_type} were calculated. Skipping plot.")
        return
    
    if 'factor_names' not in test_results or not test_results['factor_names']:
        print("No factor names found in results. Skipping plot.")
        return
    
    # Set up plot
    plt.figure(figsize=(12, 8))
    
    # Get window dates and convert to datetime
    dates = []
    factor_weights_over_time = {}
    
    # Initialize weights for each factor
    factor_names = test_results['factor_names']
    for factor in factor_names:
        factor_weights_over_time[factor] = []
    
    # Collect weights over time
    for idx in sorted(test_results['window_dates'].keys()):
        if idx in test_results[weight_type]:
            dates.append(pd.to_datetime(test_results['window_dates'][idx]['test_start']))
            
            # Get weights for this window
            weights = test_results[weight_type][idx]
            
            # Check if weights is a numpy array or a dictionary
            if isinstance(weights, np.ndarray):
                # If it's a numpy array, assume the indices match factor_names
                for i, factor in enumerate(factor_names):
                    if i < len(weights):
                        factor_weights_over_time[factor].append(weights[i])
                    else:
                        factor_weights_over_time[factor].append(0)
            else:
                # If it's a dictionary, use the get method
                for factor in factor_names:
                    factor_weights_over_time[factor].append(weights.get(factor, 0))
    
    if not dates:
        print("No valid dates and factor weights found. Skipping plot.")
        return
    
    # Plot stacked weights over time
    plt.stackplot(dates, 
                  *[factor_weights_over_time[factor] for factor in factor_names],
                  labels=factor_names,
                  alpha=0.7)
    
    plt.xlabel('Date')
    plt.ylabel('Factor Weight')
    plt.title('Factor Weights Over Time' + (' (Long-Only Constraint)' if long_only else ''))
    plt.grid(True)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
    # Format x-axis as dates
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(mdates.YearLocator())
    plt.gcf().autofmt_xdate()
    
    # Add zero line
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    # Save plot
    filename = 'factor_weights_with_long_only_constraint.pdf' if long_only else 'factor_weights_long_short.pdf'
    output_file = os.path.join(output_dir, filename)
    plt.savefig(output_file, bbox_inches='tight')
    plt.close()
    print(f"Saved factor weights plot to {output_file}")

def main():
    """Generate plots from factor timing results"""
    args = parse_arguments()
    
    # Determine which plots to generate
    if args.plot_types.lower() == 'all':
        plot_types = ['lambda', 'sharpe', 'cumulative', 'drawdown', 'factor_weights']
    else:
        plot_types = [p.strip().lower() for p in args.plot_types.split(',')]
    
    print(f"Loading test results from {args.input_file}...")
    try:
        with open(args.input_file, 'rb') as f:
            test_results = pickle.load(f)
    except FileNotFoundError:
        print(f"Error: {args.input_file} not found. Run run_factor_timing_test.py first.")
        return
    
    # Create output directory if it doesn't exist
    if args.output_dir != '.' and not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Generate requested plots
    for plot_type in plot_types:
        if plot_type == 'lambda':
            plot_lambda_evolution(test_results, args.output_dir)
        elif plot_type == 'sharpe':
            plot_sharpe_ratios(test_results, args.output_dir)
        elif plot_type == 'cumulative':
            plot_cumulative_returns(test_results, args.output_dir)
        elif plot_type == 'drawdown':
            plot_drawdowns(test_results, args.output_dir)
        elif plot_type == 'factor_weights':
            plot_factor_weights(test_results, long_only=False, output_dir=args.output_dir)
            plot_factor_weights(test_results, long_only=True, output_dir=args.output_dir)
    
    print("Plot generation completed successfully.")

if __name__ == "__main__":
    main() 