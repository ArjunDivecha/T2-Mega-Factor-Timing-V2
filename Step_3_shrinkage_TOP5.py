"""
# Factor Timing Top 5 Implementation
# 
# INPUT FILES:
# - rolling_windows.pkl: Dictionary containing training and testing windows
#   created in the previous phase
#
# OUTPUT FILES:
# - shrinkage_results.pkl: Dictionary containing:
#   - portfolio_weights: Equal weights (20%) for top 5 portfolios
#   - performance_metrics: Out-of-sample performance results
#
# This script implements a simplified factor timing methodology:
# 1. Selects the top 5 best performing portfolios based on training data
# 2. Assigns equal weights (20%) for each portfolio
# 3. Evaluates performance directly on the test window (no validation)
#
# Author: Claude
# Last Updated: May 2025
"""

import pandas as pd
import numpy as np
import pickle
import os
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.covariance import LedoitWolf
from sklearn.impute import SimpleImputer
import argparse
import time
import gc
import concurrent.futures
from functools import partial
from factor_timing_params import add_common_args, parse_window_indices, load_rolling_windows

# Define input/output files
ROLLING_WINDOWS_FILE = "rolling_windows.pkl"
SHRINKAGE_RESULTS_FILE = "shrinkage_results.pkl"

# Cache for cleaned returns data to avoid redundant cleaning
_cleaned_returns_cache = {}

def load_data():
    """Load the rolling windows data"""
    print(f"Loading rolling windows data from {ROLLING_WINDOWS_FILE}...")
    
    if not os.path.exists(ROLLING_WINDOWS_FILE):
        print(f"Error: {ROLLING_WINDOWS_FILE} not found.")
        return None
        
    with open(ROLLING_WINDOWS_FILE, 'rb') as f:
        data = pickle.load(f)
    
    return data

def clean_returns_data(returns):
    """
    Handle missing values in returns data
    
    Parameters:
    -----------
    returns : DataFrame
        Returns data with potential NaN values
        
    Returns:
    --------
    cleaned_returns : DataFrame
        Returns data with NaN values handled
    """
    # Check if we already cleaned this data frame (using its memory address as key)
    cache_key = id(returns)
    if cache_key in _cleaned_returns_cache:
        print(f"Using cached cleaned returns data (shape: {returns.shape})")
        return _cleaned_returns_cache[cache_key]
    
    print(f"Cleaning returns data (shape: {returns.shape})...")
    print(f"Missing values before cleaning: {returns.isna().sum().sum()}")
    
    # Create a copy to avoid modifying the original
    cleaned_returns = returns.copy()
    
    # For each column, fill NaNs with the column mean
    for col in cleaned_returns.columns:
        # Get column mean
        col_mean = cleaned_returns[col].mean()
        # Use proper pandas method without chained assignment
        cleaned_returns.loc[:, col] = cleaned_returns[col].fillna(col_mean)
    
    # If any NaNs remain (columns with all NaNs), fill with zeros
    cleaned_returns = cleaned_returns.fillna(0)
    
    print(f"Missing values after cleaning: {cleaned_returns.isna().sum().sum()}")
    
    # Cache the cleaned dataframe
    _cleaned_returns_cache[cache_key] = cleaned_returns
    
    return cleaned_returns

def select_top_portfolios(returns, n=5):
    """
    Select top N portfolios based on trailing returns
    
    Parameters:
    -----------
    returns : DataFrame
        Returns data for all portfolios
    n : int
        Number of top portfolios to select (default: 5)
    
    Returns:
    --------
    top_returns : DataFrame
        Returns data for top N portfolios
    """
    print(f"Selecting top {n} portfolios from {returns.shape[1]} total portfolios...")
    
    # Calculate mean returns for each portfolio
    mean_returns = returns.mean()
    
    # Select top N portfolios by mean return
    top_n_indices = mean_returns.nlargest(n).index  # Removed .abs() to focus on highest positive returns
    top_returns = returns[top_n_indices]
    
    print(f"Selected {top_returns.shape[1]} portfolios: {list(top_n_indices)}")
    
    return top_returns

def equal_weight_portfolios(returns, all_portfolio_names):
    """
    Assign equal weights to selected portfolios, zeros to all others
    
    Parameters:
    -----------
    returns : DataFrame
        Returns data for selected portfolios
    all_portfolio_names : Index or array-like
        Names of all portfolios in the original dataset
        
    Returns:
    --------
    weights : ndarray
        Weights for all portfolios (20% each for selected portfolios, 0 for others)
    """
    selected_portfolios = returns.columns
    num_selected = len(selected_portfolios)
    equal_weight = 1.0 / len(selected_portfolios) if len(selected_portfolios) > 0 else 0.0
    print(f"Assigning equal weights ({equal_weight:.2f} each) to {len(selected_portfolios)} selected portfolios out of {len(all_portfolio_names)} total portfolios")
    
    # Create a weights array with zeros for all portfolios
    weights = np.zeros(len(all_portfolio_names))
    
    # Get the indices of the selected portfolios in the all_portfolio_names array
    for i, portfolio in enumerate(all_portfolio_names):
        if portfolio in selected_portfolios:
            # Assign equal weight to this selected portfolio
            weights[i] = 1.0 / num_selected
    
    # Verify that weights sum to 1.0
    weight_sum = np.sum(weights)
    print(f"Sum of all weights: {weight_sum:.4f}")
    print(f"Number of non-zero weights: {np.count_nonzero(weights)}")
    
    return weights

def ledoit_wolf_shrinkage(returns):
    """
    Apply Ledoit-Wolf shrinkage to estimate the covariance matrix
    
    Parameters:
    -----------
    returns : DataFrame
        Returns of factor timing portfolios
        
    Returns:
    --------
    cov_lw : ndarray
        Shrinkage estimate of the covariance matrix
    shrinkage : float
        The shrinkage intensity (delta*) determined by Ledoit-Wolf.
    """
    print("Applying Ledoit-Wolf shrinkage to covariance matrix...")
    start_time = time.time()
    
    # Clean returns data to handle NaN values
    cleaned_returns = clean_returns_data(returns)
    
    # Initialize Ledoit-Wolf estimator
    lw = LedoitWolf()
    
    # Fit to the cleaned returns data
    print("Fitting Ledoit-Wolf estimator...")
    lw.fit(cleaned_returns)
    
    # Get the shrunk covariance matrix
    cov_lw = lw.covariance_
    shrinkage = lw.shrinkage_
    
    elapsed_time = time.time() - start_time
    print(f"Ledoit-Wolf shrinkage completed in {elapsed_time:.2f} seconds")
    print(f"  Shrinkage Intensity (delta*): {shrinkage:.6f}")
    
    return cov_lw, shrinkage

def optimal_portfolio_weights(returns, cov_matrix, lambda_val=0):
    """
    Calculate optimal portfolio weights with regularization
    
    Following the method from Kozak, Nagel, and Santosh (2020)
    w = [Σ + (λ/T) * D]^(-1) * μ̂
    
    Parameters:
    -----------
    returns : DataFrame
        Returns of factor timing portfolios
    cov_matrix : ndarray
        Shrinkage estimate of the covariance matrix
    lambda_val : float
        Regularization parameter
        
    Returns:
    --------
    weights : ndarray
        Optimal portfolio weights
    """
    print(f"Calculating optimal portfolio weights with lambda = {lambda_val:.4f}...")
    start_time = time.time()
    
    # Clean returns to handle NaN values for mean calculation
    cleaned_returns = clean_returns_data(returns)
    
    # Calculate sample means
    mu_hat = cleaned_returns.mean().values
    
    # Adjust the impact of the mean returns vector
    return_multiplier = 10.0  # Adjust this value to control emphasis
    mu_hat = mu_hat * return_multiplier
    
    # Ensure mu_hat is a 2D column vector for solve
    mu_hat = mu_hat.reshape(-1, 1)
    
    # Ensure cov_matrix is a proper 2D array
    if not isinstance(cov_matrix, np.ndarray) or cov_matrix.ndim != 2:
        print("Converting covariance matrix to 2D NumPy array")
        cov_matrix = np.array(cov_matrix, dtype=float)
        if cov_matrix.ndim == 1:
            cov_matrix = cov_matrix.reshape(-1, 1)
    
    # Create diagonal matrix D with variances of timing portfolios
    D = np.diag(np.diag(cov_matrix))
    
    # Number of observations
    T = len(cleaned_returns)
    
    # Calculate regularized inverse
    lambda_term = (lambda_val / T) * D
    
    # For numerical stability, only apply regularization to timing portfolios, not base factors
    # Base factors are the first ones in the portfolio list (constant predictor)
    num_base_factors = len(returns.columns.levels[0]) if isinstance(returns.columns, pd.MultiIndex) else 1
    
    # Create a mask for the diagonal to only apply regularization to timing portfolios
    mask = np.ones(cov_matrix.shape[0])
    mask[:num_base_factors] = 0  # No shrinkage for base factors
    
    # Apply mask to diagonal matrix - ensure proper 2D matrix shapes
    mask_matrix = np.diag(mask)
    lambda_term = lambda_term @ mask_matrix @ mask_matrix.T
    
    # Calculate inverse of regularized covariance matrix
    try:
        print("Computing inverse of regularized covariance matrix...")
        # For memory efficiency, compute (Σ + λD)^(-1)μ directly without forming the inverse
        # Use scipy.linalg.solve which is more memory-efficient than np.linalg.inv
        from scipy.linalg import solve
        
        # Ensure we have a proper 2D NumPy array for the matrix and right-hand side
        matrix_sum = cov_matrix + lambda_term
        
        # Debug information about matrix shapes
        print(f"Matrix shape: {matrix_sum.shape}, Vector shape: {mu_hat.shape}")
        
        # Ensure matrix is 2D
        if matrix_sum.ndim != 2:
            raise ValueError(f"Matrix must be 2D, got shape {matrix_sum.shape}")
        
        weights = solve(matrix_sum, mu_hat)
        # Convert back to 1D array for easier manipulation
        weights = weights.flatten()
    except np.linalg.LinAlgError:
        # If matrix is singular, add small diagonal element
        print("Warning: Singular matrix encountered, adding small diagonal element")
        cov_matrix = cov_matrix + np.eye(cov_matrix.shape[0]) * 1e-6
        matrix_sum = cov_matrix + lambda_term
        weights = solve(matrix_sum, mu_hat)
        weights = weights.flatten()
    except Exception as e:
        print(f"Error solving linear system: {e}")
        print(f"Matrix type: {type(matrix_sum)}, shape: {matrix_sum.shape}")
        print(f"Vector type: {type(mu_hat)}, shape: {mu_hat.shape}")
        raise
    
    # Scale weights to have absolute sum of 1 (focus on factor rotation)
    weights = weights / np.sum(np.abs(weights))
    
    elapsed_time = time.time() - start_time
    print(f"Optimal weights calculation completed in {elapsed_time:.2f} seconds")
    
    return weights

def calculate_portfolio_return(weights, returns, all_portfolio_names=None):
    """
    Calculate portfolio return based on weights
    
    Parameters:
    -----------
    weights : ndarray
        Portfolio weights (can be for all portfolios or just selected ones)
    returns : DataFrame
        Returns data (typically only for selected portfolios)
    all_portfolio_names : Index or array-like, optional
        Names of all portfolios corresponding to the weights array
        
    Returns:
    --------
    portfolio_return : float
        Portfolio return
    """
    # Clean returns to handle NaN values before calculating portfolio return
    cleaned_returns = clean_returns_data(returns)
    
    # If we have weights for all portfolios but returns only for selected ones
    if all_portfolio_names is not None and len(weights) == len(all_portfolio_names) and len(weights) != len(returns.columns):
        # Extract only the weights for the portfolios in the returns DataFrame
        selected_weights = np.zeros(len(returns.columns))
        for i, portfolio in enumerate(returns.columns):
            # Find the index of this portfolio in all_portfolio_names
            if portfolio in all_portfolio_names:
                idx = np.where(all_portfolio_names == portfolio)[0][0]
                selected_weights[i] = weights[idx]
        
        # Use the selected weights for the dot product
        return np.dot(selected_weights, cleaned_returns.values.T)
    else:
        # Standard case: weights and returns have matching dimensions
        return np.dot(weights, cleaned_returns.values.T)

def calculate_sharpe_ratio(returns):
    """
    Calculate annualized Sharpe ratio
    
    Parameters:
    -----------
    returns : ndarray
        Array of returns
        
    Returns:
    --------
    sharpe : float
        Annualized Sharpe ratio
    """
    # Annualization factor for monthly returns
    annual_factor = 12
    
    # Check if we have enough data points for a meaningful calculation
    if len(returns) < 2:
        print("Warning: Not enough data points to calculate Sharpe ratio. Using only mean return.")
        mean_return = np.mean(returns) * annual_factor
        # Return just the mean annualized return since we can't calculate std dev
        return mean_return if not np.isnan(mean_return) else 0.0
    
    mean_return = np.mean(returns) * annual_factor
    std_return = np.std(returns, ddof=1) * np.sqrt(annual_factor)
    
    # Avoid division by zero or very small numbers
    if std_return < 1e-8:
        print("Warning: Standard deviation is close to zero. Using only mean return.")
        return mean_return if not np.isnan(mean_return) else 0.0
        
    sharpe = mean_return / std_return
    
    # Handle NaN values (can occur with 0/0 scenarios)
    if np.isnan(sharpe):
        print("Warning: Sharpe ratio calculation resulted in NaN. Defaulting to zero.")
        return 0.0
        
    return sharpe

def optimize_lambda(training_window, lambda_grid=None):
    """
    Return a fixed lambda value since we're not using validation
    
    Parameters:
    -----------
    training_window : DataFrame
        Training window returns
    lambda_grid : array-like, optional
        Grid of lambda values to try (not used)
        
    Returns:
    --------
    optimal_lambda : float
        Fixed lambda value
    shrinkage_intensity : float
        The Ledoit-Wolf shrinkage intensity used for the covariance matrix.
    """
    # Use fixed lambda value since we don't have validation data
    fixed_lambda = 0.01
    print(f"Using fixed lambda value of {fixed_lambda} (no validation data)")
    
    # Apply Ledoit-Wolf shrinkage to get covariance matrix and intensity
    _, shrinkage_intensity = ledoit_wolf_shrinkage(training_window)
    
    return fixed_lambda, shrinkage_intensity

def run_factor_timing_optimization(rolling_data, window_indices=None, max_portfolios=5):
    """
    Run the factor timing optimization for specified windows using top portfolios with equal weights
    
    Parameters:
    -----------
    rolling_data : dict
        Dictionary with training and testing windows
    window_indices : list, optional
        Specific windows to process. If None, process all windows.
    max_portfolios : int
        Number of top portfolios to select (default: 10)
        
    Returns:
    --------
    results : dict
        Results of optimization
    """
    print(f"Running top {max_portfolios} equal-weighted portfolio selection...")
    start_time = time.time()
    
    total_windows = len(rolling_data['training_windows'])
    
    if window_indices is None:
        window_indices = list(range(total_windows))
    
    print(f"Processing {len(window_indices)} windows...")
    print(f"Using top {max_portfolios} portfolios with equal weights")
    
    # Initialize results
    results = {
        'window_indices': window_indices,
        'portfolio_weights': {},
        'training_sharpe': {},
        'test_returns': {},
        'test_sharpe': {},
        'window_dates': {},
        'selected_portfolios': {},
        # Add fields expected by run_factor_timing_test.py
        'optimal_lambdas': {},
        'shrinkage_intensities': {}  # May be needed by other parts of the pipeline
    }
    
    # Copy the window dates we'll process
    for i in window_indices:
        results['window_dates'][i] = rolling_data['window_dates'][i]
    
    for i in window_indices:
        window_start_time = time.time()
        print(f"\n{'=' * 60}")
        print(f"Processing window {i+1}/{len(window_indices)}...")
        print(f"{'=' * 60}")
        
        # Get data for this window
        original_training_window = rolling_data['training_windows'][i]
        testing_window = rolling_data['testing_windows'][i]
        
        # Store all portfolio names before filtering
        all_portfolio_names = original_training_window.columns
        
        # Print window dates
        train_start = rolling_data['window_dates'][i]['train_start']
        train_end = rolling_data['window_dates'][i]['train_end']
        test_start = rolling_data['window_dates'][i]['test_start']
        test_end = rolling_data['window_dates'][i]['test_end']
        
        print(f"Training period:   {train_start} to {train_end}")
        print(f"Test period:       {test_start} to {test_end}")
        
        # Select top portfolios based on training returns
        training_window = select_top_portfolios(original_training_window, n=max_portfolios)
        # Select same portfolios for testing
        selected_columns = training_window.columns
        testing_window = testing_window[selected_columns]
        
        # Store selected portfolio columns
        results['selected_portfolios'][i] = selected_columns
        
        # Assign equal weights to the top 5 portfolios (20% each), zeros to all others
        weights = equal_weight_portfolios(training_window, all_portfolio_names)
        results['portfolio_weights'][i] = weights
        
        # Add dummy lambda value (0.0) since we're not using regularization
        results['optimal_lambdas'][i] = 0.0
        
        # Add dummy shrinkage intensity (0.0) since we're not using Ledoit-Wolf shrinkage
        results['shrinkage_intensities'][i] = 0.0
        
        # Calculate performance metrics
        print("Calculating performance metrics...")
        train_returns = calculate_portfolio_return(weights, training_window, all_portfolio_names)
        test_returns = calculate_portfolio_return(weights, testing_window, all_portfolio_names)
        
        # Calculate Sharpe ratios
        train_sharpe = calculate_sharpe_ratio(train_returns)
        test_sharpe = calculate_sharpe_ratio(test_returns)
        
        results['training_sharpe'][i] = train_sharpe
        results['test_returns'][i] = test_returns
        results['test_sharpe'][i] = test_sharpe
        
        window_elapsed_time = time.time() - window_start_time
        print(f"\nWindow {i+1} Results (completed in {window_elapsed_time:.2f} seconds):")
        print(f"  Training Sharpe:   {train_sharpe:.4f}")
        print(f"  Test Sharpe:       {test_sharpe:.4f}")
    
    # Calculate overall out-of-sample Sharpe ratio
    all_test_returns = []
    for i in window_indices:
        if i in results['test_returns']:
            all_test_returns.extend(results['test_returns'][i])
    
    overall_sharpe = calculate_sharpe_ratio(all_test_returns)
    print(f"\nOverall out-of-sample Sharpe ratio: {overall_sharpe:.4f}")
    
    elapsed_time = time.time() - start_time
    print(f"Optimization completed in {elapsed_time:.2f} seconds")
    
    return results

def plot_lambdas(results):
    """
    Plot optimal lambda values over time
    
    Parameters:
    -----------
    results : dict
        Results dictionary containing window_dates and optimal_lambdas
    """
    plt.figure(figsize=(10, 6))
    
    # Extract window indices and validation end dates
    indices = results['window_indices']
    dates = [results['window_dates'][i]['val_end'] for i in indices]
    lambdas = [results['optimal_lambdas'][i] for i in indices]
    
    plt.plot(dates, lambdas, marker='o')
    plt.title('Optimal Lambda Values Over Time')
    plt.xlabel('End of Validation Window')
    plt.ylabel('Optimal Lambda')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save figure
    plt.savefig('optimal_lambdas.pdf')
    print("Saved optimal_lambdas.pdf")
    plt.close()

def plot_performance(results):
    """
    Plot performance metrics
    
    Parameters:
    -----------
    results : dict
        Results dictionary containing window_dates and test_sharpe
    """
    # Extract window indices and test end dates
    indices = results['window_indices']
    dates = [results['window_dates'][i]['test_end'] for i in indices]
    sharpes = [results['test_sharpe'][i] for i in indices]
    
    # Plot Sharpe ratios
    plt.figure(figsize=(12, 6))
    plt.plot(dates, sharpes, marker='o', label='Test Window Sharpe')
    
    # Calculate and plot moving average if we have enough data
    if len(sharpes) >= 3:
        ma_periods = min(len(sharpes) // 2, 12)
        test_ma = pd.Series(sharpes).rolling(ma_periods).mean()
        plt.plot(dates, test_ma, label=f'{ma_periods}-Month Moving Avg', linewidth=2)
    
    plt.title('Out-of-Sample Sharpe Ratios')
    plt.xlabel('End of Test Window')
    plt.ylabel('Sharpe Ratio')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    # Save figure
    plt.savefig('test_sharpe_ratios.pdf')
    print("Saved test_sharpe_ratios.pdf")
    plt.close()

def save_results(results, file_path=SHRINKAGE_RESULTS_FILE):
    """
    Save results to pickle file
    
    Parameters:
    -----------
    results : dict
        Dictionary of results to save
    file_path : str
        Path to save results
    """
    print(f"Saving results to {file_path}...")
    
    # Make sure directory exists
    os.makedirs(os.path.dirname(file_path) if os.path.dirname(file_path) else '.', exist_ok=True)
    
    # Save results as pickle
    with open(file_path, 'wb') as f:
        pickle.dump(results, f)
    
    print("Results saved successfully")

def parse_arguments():
    """Parse command-line arguments"""
    parser = add_common_args('Run top 10 equal-weighted portfolio selection')
    
    # Additional script-specific arguments
    parser.add_argument('--output_file', type=str, default=SHRINKAGE_RESULTS_FILE,
                       help=f'Output file path (default: {SHRINKAGE_RESULTS_FILE})')
    
    # Override the default max_windows to 0 (process all windows)
    parser.set_defaults(max_windows=0)
    
    return parser.parse_args()

def find_window_by_year(window_dates, target_year=2010):
    """Find window index where validation period includes the target year"""
    for idx, dates in window_dates.items():
        val_end_year = dates['val_end'].year
        if val_end_year == target_year:
            return idx
    
    # If exact year not found, find closest
    closest_idx = None
    min_diff = float('inf')
    for idx, dates in window_dates.items():
        diff = abs(dates['val_end'].year - target_year)
        if diff < min_diff:
            min_diff = diff
            closest_idx = idx
    
    return closest_idx

def main():
    """Main function to run top 5 equal-weighted portfolio selection"""
    args = parse_arguments()
    
    # Load rolling windows data
    rolling_data = load_rolling_windows()
    if rolling_data is None:
        return
    
    # Get window indices to process
    window_indices = parse_window_indices(args, rolling_data)
    if not window_indices:
        print("No windows to process. Exiting.")
        return
    
    print(f"Processing {len(window_indices)} windows using top 5 equal-weighted portfolios...")
    
    # Process top 5 equal-weighted portfolio selection
    results = run_factor_timing_optimization(
        rolling_data,
        window_indices=window_indices,
        max_portfolios=5  # Fixed to always use top 5 portfolios
    )
    
    # Save results
    print(f"Saving results to {args.output_file}...")
    with open(args.output_file, 'wb') as f:
        pickle.dump(results, f)
    
    print("Top 5 equal-weighted portfolio selection completed successfully.")

if __name__ == "__main__":
    main()