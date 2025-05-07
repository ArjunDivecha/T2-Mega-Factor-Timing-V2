"""
# Factor Rotation Implementation
#
# INPUT FILES:
# - shrinkage_results.pkl: Contains optimal weights for filtered factor timing portfolios
# - rolling_windows.pkl: Contains list of filtered factor timing portfolio names
# - prepared_data.pkl: Contains original factor names
#
# OUTPUT FILES:
# - rotated_optimal_weights.xlsx: Excel file with original factor names as rows
#   and dates (test end dates) as columns, containing the aggregated weights.
#
# This script implements Phase 3 of the factor timing methodology:
# 1. Loads optimal weights for the filtered factor timing portfolios.
# 2. Extracts the original factor name from each filtered portfolio name.
# 3. Aggregates (sums) the weights for all portfolios belonging to the same original factor
#    for each time period (rolling window).
# 4. Saves the time series of these aggregated weights for the original factors.
#
# Author: Claude
# Last Updated: May 5, 2025
"""

import pandas as pd
import numpy as np
import pickle
import os
from collections import defaultdict
import scipy.optimize as spo

# Define input and output files
SHRINKAGE_RESULTS_FILE = "shrinkage_results.pkl"
ROLLING_WINDOWS_FILE = "rolling_windows.pkl"
PREPARED_DATA_FILE = "prepared_data.pkl"
OUTPUT_FILE = "rotated_optimal_weights.xlsx"

def load_pickle(file_path, description):
    """Loads data from a pickle file."""
    print(f"Loading {description} from {file_path}...")
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return None
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        print(f"Successfully loaded {description}.")
        return data
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def extract_original_factor(portfolio_name):
    """Extracts the original factor name from the timing portfolio name."""
    # For portfolio names like "10Yr Bond 12_CS_US 10Yr" we want to extract "10Yr Bond 12_CS"
    
    # Simple approach: Look for _CS_ or _TS_ pattern and keep everything before it and the CS/TS part
    if '_CS_' in portfolio_name:
        base_name, rest = portfolio_name.split('_CS_', 1)
        return f"{base_name}_CS"
    elif '_TS_' in portfolio_name:
        base_name, rest = portfolio_name.split('_TS_', 1)
        return f"{base_name}_TS"
    
    # Alternative pattern: Factor_CS or Factor_TS at the beginning
    parts = portfolio_name.split('_')
    if len(parts) >= 2 and parts[1] in ['CS', 'TS']:
        return f"{parts[0]}_{parts[1]}"
    
    # Check for CS or TS in any position
    for i, part in enumerate(parts[:-1]):
        if part in ['CS', 'TS']:
            return '_'.join(parts[:i+1])
    
    # If we have a name ending with _CS or _TS, use that
    if portfolio_name.endswith('_CS') or portfolio_name.endswith('_TS'):
        return portfolio_name
    
    # Last resort: try to identify if the name contains CS or TS anywhere
    if '_CS' in portfolio_name:
        index = portfolio_name.find('_CS')
        return portfolio_name[:index+3]  # Include the _CS
    elif '_TS' in portfolio_name:
        index = portfolio_name.find('_TS')
        return portfolio_name[:index+3]  # Include the _TS
    
    # If all else fails, return the portfolio name as is
    print(f"Warning: Could not parse CS/TS suffix from '{portfolio_name}'. Returning name without suffix.")
    return portfolio_name

def extract_factor_names(portfolio_names):
    """
    Extract factor names from portfolio name strings
    
    Parameters:
    -----------
    portfolio_names : list
        List of portfolio names in the format 'Factor_CS_Predictor' or similar
        
    Returns:
    --------
    factor_names : list
        List of unique factor names
    factor_map : dict
        Mapping from portfolio name to factor name
    """
    print(f"Extracting factor names from {len(portfolio_names)} portfolio names...")
    
    # Initialize mapping
    factor_map = {}
    
    # Extract factor names from portfolio names
    factor_names = set()
    
    # Print a sample of portfolio names for debugging
    if portfolio_names:
        sample_names = portfolio_names[:min(5, len(portfolio_names))]
        print(f"Sample portfolio names: {sample_names}")
    
    for portfolio in portfolio_names:
        # Skip empty strings
        if not portfolio:
            continue
            
        # Most portfolio names are Factor_CS_Predictor or Factor_TS_Predictor
        parts = portfolio.split('_')
        
        # Determine which part contains the factor name based on common patterns
        factor_name = None
        
        # Approach 1: Check if CS/TS is present to determine format
        if len(parts) >= 3:
            if parts[1] in ['CS', 'TS']:
                # Format is likely Factor_CS_Predictor
                # Keep CS/TS suffix with the factor name
                factor_name = f"{parts[0]}_{parts[1]}"
            elif parts[-2] in ['CS', 'TS']:
                # Format might be something like Predictor_CS_Factor
                # Keep CS/TS suffix with the factor name
                factor_name = f"{parts[-1]}_{parts[-2]}"
        
        # Approach 2: If no CS/TS pattern identified, use more generic approach
        if not factor_name:
            if len(parts) >= 2:
                # Try to identify a meaningful factor name
                # Common formats include Factor_Suffix, Prefix_Factor, etc.
                factor_name = parts[0]  # Default to first part
                
                # Check for known indicator words that suggest this isn't the factor
                if parts[0].lower() in ['long', 'short', 'beta', 'market', 'industry', 'sector']:
                    factor_name = parts[1] if len(parts) > 1 else parts[0]
            else:
                # Simple case, just use the whole string
                factor_name = portfolio
        
        # If all approaches failed, use the first part or the whole string
        if not factor_name:
            factor_name = parts[0] if parts else portfolio
            
        # Store in mapping and add to set of unique factors
        factor_map[portfolio] = factor_name
        factor_names.add(factor_name)
    
    # If we couldn't extract any factor names, create generic ones
    if not factor_names:
        print("Warning: Could not extract meaningful factor names, creating generic ones")
        factor_map = {portfolio: f"Factor_{i}" for i, portfolio in enumerate(portfolio_names)}
        factor_names = set(factor_map.values())
    
    factor_names_list = list(factor_names)
    print(f"Extracted {len(factor_names_list)} unique factor names")
    
    # Print some of the extracted factor names for verification
    if factor_names_list:
        print(f"Sample factor names: {factor_names_list[:min(10, len(factor_names_list))]}")
        
    return factor_names_list, factor_map

def rotate_portfolio_weights(factor_map, portfolio_weights, portfolio_names):
    """
    Rotate portfolio weights to factor weights
    
    Parameters:
    -----------
    factor_map : dict
        Mapping from portfolio name to factor name
    portfolio_weights : ndarray
        Weights for each portfolio
    portfolio_names : list
        List of portfolio names
        
    Returns:
    --------
    factor_weights : dict
        Weights for each factor
    """
    print("Rotating portfolio weights to factor weights...")
    
    # Initialize factor weights dictionary
    factor_weights = {}
    
    # Count missing mappings for reporting
    missing_count = 0
    total_count = len(portfolio_names)
    
    # Sum weights for each factor
    for i, portfolio in enumerate(portfolio_names):
        if i >= len(portfolio_weights):
            print(f"Warning: Portfolio index {i} exceeds weight array length {len(portfolio_weights)}")
            continue
            
        weight = portfolio_weights[i]
        factor = factor_map.get(portfolio)
        
        if factor is None:
            # Create a factor mapping on-the-fly for missing portfolios
            parts = portfolio.split('_')
            if len(parts) > 0:
                # Use the first part as the factor name
                factor = parts[0]
                # Add to the factor map for future use
                factor_map[portfolio] = factor
                missing_count += 1
            else:
                # Very unlikely case with empty portfolio name
                print(f"Warning: Cannot create mapping for empty portfolio name")
                continue
            
        if factor in factor_weights:
            factor_weights[factor] += weight
        else:
            factor_weights[factor] = weight
    
    # Normalize factor weights
    total_weight = sum(abs(val) for val in factor_weights.values())
    if total_weight > 0:
        for factor in factor_weights:
            factor_weights[factor] /= total_weight
    
    if missing_count > 0:
        print(f"Created on-the-fly mappings for {missing_count} out of {total_count} portfolios")
    
    print(f"Rotated weights for {len(factor_weights)} factors")
    return factor_weights

def apply_long_only_constraint(factor_weights, factor_names):
    """
    Apply long-only constraint to factor weights
    
    Parameters:
    -----------
    factor_weights : dict
        Dictionary of factor weights
    factor_names : list
        List of all factor names
        
    Returns:
    --------
    long_only_weights : ndarray
        Array of long-only weights
    """
    print("Applying long-only constraint to factor weights...")
    
    # Create initial weight array from factor weights dict
    weights_array = np.zeros(len(factor_names))
    for i, factor in enumerate(factor_names):
        weights_array[i] = factor_weights.get(factor, 0)
    
    # Define objective function to minimize tracking error
    def objective(w):
        return np.sum((w - weights_array)**2)
    
    # Define constraints
    constraints = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}  # Weights sum to 1
    ]
    
    # Define bounds (long-only)
    bounds = [(0, None) for _ in range(len(factor_names))]
    
    # Solve optimization problem
    initial_guess = np.ones(len(factor_names)) / len(factor_names)
    result = spo.minimize(objective, initial_guess, method='SLSQP', 
                     constraints=constraints, bounds=bounds)
    
    if result.success:
        print("Successfully applied long-only constraint")
        return result.x
    else:
        print(f"Warning: Optimization failed: {result.message}")
        # Return normalized positive weights as fallback
        positive_weights = np.maximum(weights_array, 0)
        if np.sum(positive_weights) > 0:
            return positive_weights / np.sum(positive_weights)
        else:
            return np.ones(len(factor_names)) / len(factor_names)

def main():
    """Main function to perform factor rotation."""
    print("=== FACTOR ROTATION ===")

    # Load necessary data
    shrinkage_results = load_pickle(SHRINKAGE_RESULTS_FILE, "shrinkage results")
    rolling_data = load_pickle(ROLLING_WINDOWS_FILE, "rolling window data")
    prepared_data = load_pickle(PREPARED_DATA_FILE, "prepared data")

    if not all([shrinkage_results, rolling_data, prepared_data]):
        print("Error loading input files. Exiting.")
        return

    # --- Extract required data ---
    # Get optimal weights for each window (filtered portfolios)
    optimal_weights_timing = shrinkage_results.get('portfolio_weights')
    if optimal_weights_timing is None:
        print("Error: 'portfolio_weights' not found in shrinkage results.")
        return

    # Get the list of filtered portfolio names used in the optimization
    # Use 'all_portfolios' instead of 'filtered_portfolios' after Step 2 changes
    filtered_portfolio_names = rolling_data.get('all_portfolios')
    if filtered_portfolio_names is None:
        print("Error: 'all_portfolios' not found in rolling window data.")
        return
        
    # Create a mapping from filtered portfolio names to their index
    portfolio_name_to_index = {name: idx for idx, name in enumerate(filtered_portfolio_names)}

    # Get original factor names (from the data prep stage)
    original_factors = prepared_data['factor_returns'].columns.tolist()
    if not original_factors:
        print("Error: Could not retrieve original factor names from prepared data.")
        return

    # Get window dates for column headers
    window_dates = shrinkage_results.get('window_dates')
    if window_dates is None:
        print("Error: 'window_dates' not found in shrinkage results.")
        return

    print(f"Rotating weights for {len(filtered_portfolio_names)} filtered portfolios back to {len(original_factors)} original factors.")

    # --- Perform Rotation ---
    rotated_weights_data = defaultdict(dict) # {original_factor: {date_str: weight}}

    num_windows = len(shrinkage_results['window_indices'])
    print(f"Processing {num_windows} windows...")
    
    # Print a few sample portfolio names for debugging
    print("Sample portfolio names:")
    for name in filtered_portfolio_names[:5]:
        print(f"  {name} -> {extract_original_factor(name)}")

    for i in shrinkage_results['window_indices']:
        if i not in optimal_weights_timing:
            print(f"Warning: Weights for window {i} not found. Skipping.")
            continue
            
        # Ensure the number of weights matches the number of filtered portfolios
        current_weights = optimal_weights_timing[i]
        if len(current_weights) != len(filtered_portfolio_names):
             print(f"Warning: Weight count mismatch for window {i}. Expected {len(filtered_portfolio_names)}, got {len(current_weights)}. Skipping.")
             continue

        if i not in window_dates or 'test_end' not in window_dates[i]:
            print(f"Warning: Test end date not found for window {i}. Skipping.")
            continue

        # Get test end date for column header
        test_end_date = window_dates[i]['test_end']
        date_str = test_end_date.strftime('%Y-%m-%d')

        # Aggregate weights for each original factor
        temp_rotated_weights = defaultdict(float)
        for portfolio_name in filtered_portfolio_names:
            original_factor = extract_original_factor(portfolio_name)
            portfolio_idx = portfolio_name_to_index.get(portfolio_name)
            if portfolio_idx is not None:
                temp_rotated_weights[original_factor] += current_weights[portfolio_idx]
            else:
                print(f"Warning: Index for portfolio '{portfolio_name}' not found in window {i}. Skipping.")
        
        # Add validation to check if we're capturing all weights
        total_original_weight = sum(abs(w) for w in temp_rotated_weights.values())
        total_portfolio_weight = sum(abs(w) for w in current_weights)
        if abs(total_original_weight - total_portfolio_weight) > 0.01:
            print(f"Warning: Weight mismatch in window {i}. Original: {total_original_weight:.4f}, Portfolio: {total_portfolio_weight:.4f}")

        # Store aggregated weights for the date
        for factor, weight in temp_rotated_weights.items():
            rotated_weights_data[factor][date_str] = weight
            
        if (i + 1) % 50 == 0:
             print(f"  Processed {i+1}/{num_windows} windows...")

    print("Rotation complete. Creating DataFrame...")

    # Convert defaultdict to DataFrame
    rotated_weights_df = pd.DataFrame.from_dict(rotated_weights_data, orient='index')

    # Ensure columns are sorted by date
    rotated_weights_df = rotated_weights_df.reindex(sorted(rotated_weights_df.columns, key=pd.to_datetime), axis=1)
    
    # Check for any extreme values
    if not rotated_weights_df.empty:
        max_weight = rotated_weights_df.max().max()
        min_weight = rotated_weights_df.min().min()
        print(f"Weight range: {min_weight:.4f} to {max_weight:.4f}")
        
        # Check if weights sum to approximately 1 for each date
        weight_sums = rotated_weights_df.sum()
        print(f"Weight sum range: {weight_sums.min():.4f} to {weight_sums.max():.4f}")
        
        # Normalize weights if they don't sum to 1
        if abs(weight_sums.mean() - 1.0) > 0.1:
            print("Normalizing weights to sum to 1 for each date...")
            for col in rotated_weights_df.columns:
                col_sum = rotated_weights_df[col].sum()
                if col_sum != 0:
                    rotated_weights_df[col] = rotated_weights_df[col] / col_sum
            
            # Verify normalization
            weight_sums_after = rotated_weights_df.sum()
            print(f"Weight sum range after normalization: {weight_sums_after.min():.4f} to {weight_sums_after.max():.4f}")

    # Add factor names as a column from the index
    rotated_weights_df.index.name = 'Factor'
    rotated_weights_df.reset_index(inplace=True)

    print(f"Rotated weights DataFrame shape: {rotated_weights_df.shape}")

    # --- Save to Excel ---
    print(f"Saving rotated weights to {OUTPUT_FILE}...")
    try:
        rotated_weights_df.to_excel(OUTPUT_FILE, index=False, sheet_name='Rotated_Weights')
        print(f"Successfully saved rotated weights to {OUTPUT_FILE}")
    except Exception as e:
        print(f"Error saving Excel file: {e}")

    print("\nPhase 3 (Factor Rotation) completed.")

if __name__ == "__main__":
    main()