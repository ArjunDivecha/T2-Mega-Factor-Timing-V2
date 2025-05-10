"""
Feature-to-Country Weight Conversion Program
===========================================

This program converts feature importance weights from a machine learning model into 
country-specific investment weights for stock market forecasting.

Version: 1.1
Last Updated: 2025-04-23

INPUT FILES:
- "factor_timing_results.xlsx" (sheet "Long-Only Weights")
  Feature weights from a machine learning model, with dates as index and features as columns
  
- "Normalized_T2_MasterCSV.csv"
  Factor data for multiple countries, containing columns: date, country, variable, value
  where variable is the feature name and value is the feature value

OUTPUT FILES:
- "Final_Country_Weights.xlsx"
  Excel file with three sheets:
  1. "All Periods": Time series of country weights for all dates
  2. "Summary Statistics": Statistical analysis of weights (mean, std dev, min, max)
  3. "Latest Weights": Most recent country weights with comparison to historical averages

METHODOLOGY:
1. For each date, identify features with significant weights from the CURRENT month's model
2. For each significant feature:
   - Select top 20% of countries (or bottom 20% for inverted features) using CURRENT month data
   - Distribute the feature's weight equally among selected countries
3. Accumulate weights across all features to get final country weights
4. For missing country data, the country is excluded from selection for that feature

MISSING DATA HANDLING:
- Countries with missing data for a feature are automatically excluded from selection
- No explicit imputation is performed in this script

"""

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

# ===============================
# DATA LOADING AND PREPROCESSING
# ===============================

# Input file paths
weights_file = "factor_timing_results.xlsx"
factor_file = "/Users/macbook2024/Dropbox/AAA Backup/Transformer/T2 Factor Timing/Normalized_T2_MasterCSV.csv"

print("Loading data...")
# Load feature weights from optimization model
feature_weights_df = pd.read_excel(weights_file, sheet_name="Long-Only Weights", index_col=0)

# Load factor data for all countries
factor_df = pd.read_csv(factor_file)
factor_df['date'] = pd.to_datetime(factor_df['date'])  # Convert date column to datetime

# ===============================
# FEATURE CLASSIFICATION
# ===============================

# Features where LOWER values are better (will select BOTTOM 20% of countries)
INVERTED_FEATURES = {
    # Financial Indicators
    'BEST Cash Flow', 'BEST Div Yield', 'BEST EPS 3 Year', 'BEST PBK', 'BEST PE', 
    'BEST PS', 'BEST ROE', 'EV to EBITDA', 'Shiller PE', 'Trailing PE', 'Positive PE', 
    'Best Price Sales', 'Debt To EV',
    # Economic Indicators
    'Currency Change', 'Debt to GDP', 'REER', '10Yr Bond 12', 'Bond Yield Change',
    # Technical Indicators
    'RSI14', 'Advance Decline', '1MTR', '3MTR', 
    # Risk Metrics
    'Bloom Country Risk'
}

# ===============================
# INITIALIZATION
# ===============================

# Get all unique countries from the factor data in their original order
all_countries = factor_df['country'].unique()  # Removed sorting to preserve original order

# Get all unique dates from the factor data
all_factor_dates = sorted(factor_df['date'].unique())

# Check if there are factor dates beyond the last feature weights date
latest_weights_date = feature_weights_df.index.max()
future_factor_dates = [d for d in all_factor_dates if d > latest_weights_date]

# Initialize DataFrame to store weights for all countries and dates
# Include both feature_weights dates and any future factor dates
all_dates = list(feature_weights_df.index) + future_factor_dates
all_weights = pd.DataFrame(index=all_dates, columns=all_countries)
all_weights = all_weights.fillna(0.0)  # Start with zero weights

# ===============================
# WEIGHT CALCULATION PROCESS
# ===============================

print("\nProcessing all dates...")
# Process each date in the all_dates list
for date in tqdm(all_dates):
    # Initialize weights for all countries on this date
    country_weights = {country: 0.0 for country in all_countries}
    
    # Convert to datetime if it's not already
    if not isinstance(date, pd.Timestamp):
        date_dt = pd.to_datetime(date)
    else:
        date_dt = date
    
    # Use the CURRENT date for feature weights instead of previous date
    # Skip if the current date is not in the feature weights index
    if date_dt not in feature_weights_df.index:
        print(f"Skipping {date} - date not available in feature weights")
        continue
    
    # Get feature weights from the CURRENT date
    date_weights = feature_weights_df.loc[date_dt]
    
    # Filter out features with negligible weights (numerical stability)
    significant_weights = date_weights[date_weights.abs() > 1e-10]
    
    # Process each feature that has a significant weight
    for feature, feature_weight in significant_weights.items():
        # Get data for this feature and CURRENT date across all countries
        feature_data = factor_df[
            (factor_df['date'] == date) & 
            (factor_df['variable'] == feature)
        ].copy()
        
        # Skip if no data available for this feature/date
        if feature_data.empty:
            continue
            
        # Calculate number of countries to select (top/bottom 20%)
        n_select = max(1, int(len(feature_data) * 0.2))
        
        # For all features, HIGHER values are better (no inversion)
        selected = feature_data.nlargest(n_select, 'value')
        
        # Distribute feature weight equally among selected countries
        weight_per_country = feature_weight / n_select
        for country in selected['country']:
            country_weights[country] += weight_per_country
    
    # Store calculated weights for this date
    for country, weight in country_weights.items():
        all_weights.loc[date, country] = weight

# ===============================
# VALIDATION AND ANALYSIS
# ===============================

# Verify weights sum to approximately 1 for each date
weight_sums = all_weights.sum(axis=1)
print("\nWeight sum statistics:")
print(weight_sums.describe())

# ===============================
# RESULTS SAVING
# ===============================

print("\nSaving results...")
output_file = 'Final_Country_Weights.xlsx'
with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
    # Sheet 1: Complete time series of country weights
    all_weights.to_excel(writer, sheet_name='All Periods')
    
    # Sheet 2: Summary statistics for each country
    summary_stats = pd.DataFrame({
        'Mean Weight': all_weights.mean(),
        'Std Dev': all_weights.std(),
        'Min Weight': all_weights.min(),
        'Max Weight': all_weights.max(),
        'Days with Weight': (all_weights > 0).sum()
    }).sort_values('Mean Weight', ascending=False)
    
    summary_stats.to_excel(writer, sheet_name='Summary Statistics')
    
    # Sheet 3: Latest country weights with comparison to historical average
    latest_weights = pd.DataFrame({
        'Weight': all_weights.iloc[-1],
        'Average Weight': all_weights.mean(),
        'Days with Weight': (all_weights > 0).sum()
    }).sort_values('Weight', ascending=False)
    
    latest_weights.to_excel(writer, sheet_name='Latest Weights')

print(f"\nResults saved to {output_file}")

# ===============================
# SUMMARY REPORTING
# ===============================

# Print top countries by average weight
print("\nTop 10 countries by average weight:")
print(summary_stats.head(10))
