"""
# Simplified Factor Timing Data Preparation
# 
# INPUT FILES:
# - T2 Mega Factor Bloom.xlsx: Contains factor returns and conditioning variables
#   - Target sheet: Factor returns 
#   - Data from bloomberg sheet: US macroeconomic and market variables
#
# OUTPUT FILES:
# - prepared_data.pkl: Processed and cleaned data ready for factor timing model
#   - Contains factor returns, standardized conditioning variables, and factor timing portfolios
#
# This script implements a simplified version of Phase 1 of the factor timing methodology:
# 1. Cleans and standardizes dates across all data
# 2. Handles missing values in factor returns and conditioning variables
# 3. Properly scales factor returns (converts from percentage to decimal form)
# 4. Standardizes conditioning variables to z-scores
# 5. Creates factor timing portfolios (cross-products of factors and predictors)
# 6. Properly lags predictors to avoid look-ahead bias
#
# Author: Claude
# Last Updated: May 6, 2025
"""

import pandas as pd
import numpy as np
import os
import pickle
from scipy import stats

# Define input and output files
INPUT_FILE = "T2 Mega Factor Bloom.xlsx"
OUTPUT_FILE = "prepared_data.pkl"

def load_data(excel_file):
    """
    Load required sheets from the Excel file
    
    Args:
        excel_file (str): Path to the Excel file
        
    Returns:
        dict: Dictionary containing DataFrames for each sheet
    """
    print(f"Loading data from {excel_file}...")
    
    # Initialize dictionary to store data
    data = {}
    
    # Load target sheet (factor returns)
    data['factor_returns'] = pd.read_excel(excel_file, sheet_name="Target")
    
    # Load macro variables - only reading this sheet as specified
    data['macro_vars'] = pd.read_excel(excel_file, sheet_name="Data from bloomberg")
    
    print(f"Loaded {len(data)} sheets from the Excel file")
    return data

def clean_and_standardize_dates(data):
    """
    Standardize dates across all DataFrames to first-of-month
    
    Args:
        data (dict): Dictionary containing DataFrames for each sheet
        
    Returns:
        dict: Dictionary with cleaned date indices
    """
    print("Standardizing dates across all data...")
    
    for key, df in data.items():
        # Skip the first row which contains column headers
        df = df.iloc[1:].copy()
        
        # Convert the date column to datetime
        df['Date'] = pd.to_datetime(df['Unnamed: 0'])
        
        # Convert to period and back to timestamp to get first of month
        df['Date'] = df['Date'].dt.to_period('M').dt.to_timestamp()
        
        # Set the date as index and drop the original column
        df = df.set_index('Date')
        df = df.drop(columns=['Unnamed: 0'])
        
        # Store the cleaned DataFrame back
        data[key] = df
        
    print("Date standardization complete")
    return data

def handle_missing_values(data):
    """
    Handle missing values in factor returns and conditioning variables
    - For factor returns: Fill with mean of available values for that factor
    - For conditioning variables: Forward fill, then backward fill
    
    Args:
        data (dict): Dictionary containing DataFrames with standardized dates
        
    Returns:
        dict: Dictionary with missing values handled
    """
    print("Handling missing values...")
    
    # Handle missing values in factor returns
    factor_returns = data['factor_returns']
    factor_returns = factor_returns.apply(lambda x: x.fillna(x.mean()), axis=0)
    data['factor_returns'] = factor_returns
    print(f"  Factor returns missing after filling: {factor_returns.isna().sum().sum()}")
    
    # Handle missing values in macro variables
    macro_vars = data['macro_vars']
    macro_vars = macro_vars.ffill().bfill()  # Forward then backward fill
    data['macro_vars'] = macro_vars
    print(f"  Macro variables missing after filling: {macro_vars.isna().sum().sum()}")
    
    return data

def scale_factor_returns(data):
    """
    Scale factor returns from percentage to decimal form (divide by 100)
    
    Args:
        data (dict): Dictionary containing DataFrames with handled missing values
        
    Returns:
        dict: Dictionary with properly scaled factor returns
    """
    print("Scaling factor returns from percentage to decimal form...")
    
    # Scale factor returns (divide by 100 to convert from percentage to decimal)
    factor_returns = data['factor_returns']
    factor_returns = factor_returns / 100.0
    data['factor_returns'] = factor_returns
    
    # Check the scale after conversion
    mean_return = factor_returns.mean().mean()
    std_return = factor_returns.std().mean()
    print(f"  Mean monthly return after scaling: {mean_return:.4f} ({mean_return*100:.2f}%)")
    print(f"  Mean monthly volatility after scaling: {std_return:.4f} ({std_return*100:.2f}%)")
    
    return data

def standardize_conditioning_variables(data):
    """
    Standardize all conditioning variables to z-scores
    
    Args:
        data (dict): Dictionary containing DataFrames with handled missing values and scaled returns
        
    Returns:
        dict: Dictionary with standardized conditioning variables
    """
    print("Standardizing conditioning variables to z-scores...")
    
    macro_vars = data['macro_vars']
    
    # Apply z-score standardization to each column
    macro_vars_std = pd.DataFrame(index=macro_vars.index)
    for col in macro_vars.columns:
        macro_vars_std[col] = stats.zscore(macro_vars[col])
    
    data['macro_vars_std'] = macro_vars_std
    print(f"Standardized {len(macro_vars_std.columns)} conditioning variables")
    
    return data

def create_factor_timing_portfolios(data):
    """
    Create factor timing portfolios by multiplying factor returns with lagged predictors
    
    Args:
        data (dict): Dictionary containing DataFrames with standardized conditioning variables
        
    Returns:
        dict: Dictionary with factor timing portfolios added
    """
    print("Creating factor timing portfolios...")
    
    factor_returns = data['factor_returns']
    macro_vars_std = data['macro_vars_std']
    
    # Get the list of factors and predictors
    factors = factor_returns.columns.tolist()
    predictors = macro_vars_std.columns.tolist()
    
    print(f"Using {len(factors)} factors and {len(predictors)} predictors")
    print(f"This will generate {len(factors) * len(predictors)} factor timing portfolios")
    
    # Initialize dictionary to store all factor timing portfolio data
    portfolio_data = {}
    
    # Create factor timing portfolios
    count = 0
    for factor in factors:
        # Get factor returns
        factor_data = factor_returns[factor]
        
        for predictor in predictors:
            # Get lagged predictor (shift by 1 month to avoid look-ahead bias)
            predictor_data = macro_vars_std[predictor].shift(1)
            
            # Create timing portfolio (factor return * lagged predictor)
            portfolio_name = f"{factor}_{predictor}"
            portfolio_data[portfolio_name] = factor_data * predictor_data
            count += 1
            
            # Print progress every 500 portfolios
            if count % 500 == 0:
                print(f"  Created {count} factor timing portfolios")
    
    # Create the factor timing DataFrame all at once
    factor_timing = pd.DataFrame(portfolio_data, index=factor_returns.index)
    
    print(f"Created {count} factor timing portfolios")
    data['factor_timing'] = factor_timing
    
    return data

def main():
    """
    Main function to run all data preparation steps
    """
    print("=== SIMPLIFIED FACTOR TIMING DATA PREPARATION ===")
    
    # Check if the input file exists
    if not os.path.exists(INPUT_FILE):
        print(f"Error: Input file {INPUT_FILE} not found")
        return
    
    # Load data from Excel file
    data = load_data(INPUT_FILE)
    
    # Clean and standardize dates
    data = clean_and_standardize_dates(data)
    
    # Handle missing values
    data = handle_missing_values(data)
    
    # Scale factor returns
    data = scale_factor_returns(data)
    
    # Standardize conditioning variables
    data = standardize_conditioning_variables(data)
    
    # Create factor timing portfolios
    data = create_factor_timing_portfolios(data)
    
    # Save the processed data
    with open(OUTPUT_FILE, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"\nData preparation complete. Saved results to {OUTPUT_FILE}")
    print(f"Factor timing portfolios shape: {data['factor_timing'].shape}")
    
    # Update status
    print("\nSimplified Phase 1 (Data Preparation) completed successfully.")

if __name__ == "__main__":
    main()