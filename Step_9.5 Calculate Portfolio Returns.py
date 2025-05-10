"""
Portfolio Performance Analysis Program
=====================================

This program calculates and analyzes the performance of a country-weighted investment portfolio
by applying country weights to historical returns data. It compares the performance against
an equal-weight benchmark and generates comprehensive performance metrics and visualizations.

Version: 1.1
Last Updated: 2025-04-23

INPUT FILES:
- "Final_Country_Weights.xlsx"
  Country weights file with "All Periods" sheet containing dates as index and countries as columns.
  Each value represents the weight allocated to a country on a specific date.
  
- "Portfolio_Data.xlsx"
  Contains two sheets:
  1. "Returns": Historical returns for each country, with dates as index and countries as columns
  2. "Benchmarks": Benchmark returns including equal-weight portfolio returns

OUTPUT FILES:
- "Final_Portfolio_Returns.xlsx"
  Excel file with four sheets:
  1. "Monthly Returns": Monthly returns for portfolio, equal-weight benchmark, and net returns
  2. "Cumulative Returns": Cumulative returns over time
  3. "Statistics": Performance statistics (returns, volatility, Sharpe ratio, etc.)
  4. "Net Returns": Net (active) returns and cumulative net returns
  
- "Final_Portfolio_Returns.pdf"
  Visualization with three plots:
  1. Cumulative Total Returns: Portfolio vs Equal Weight
  2. Cumulative Net Returns: Portfolio minus Equal Weight
  3. Monthly Net Returns: Bar chart of monthly active returns

METHODOLOGY:
1. Load country weights and historical returns data
2. For each month's weights, apply them to the NEXT month's returns to avoid look-ahead bias
3. Calculate portfolio returns by applying weights to country returns
4. Compare against equal-weight benchmark
5. Calculate performance metrics (returns, volatility, Sharpe ratio, drawdowns, etc.)
6. Generate visualizations and detailed performance reports

MISSING DATA HANDLING:
- Analysis is restricted to dates common to both weights and returns datasets
- Last month is excluded due to not having future returns available
- No explicit imputation is performed in this script
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# ===============================
# VISUALIZATION SETUP
# ===============================

# Set plot style for consistent, professional visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = [12, 6]

# ===============================
# DATA LOADING
# ===============================

print("Loading data...")
# Input file paths
weights_file = 'Final_Country_Weights.xlsx'
portfolio_data_file = '/Users/macbook2024/Dropbox/AAA Backup/Transformer/T2 Factor Timing/Portfolio_Data.xlsx'

# Load country weights from the T2_Final_Country_Weights.xlsx file
weights_df = pd.read_excel(weights_file, sheet_name='All Periods', index_col=0)

# Load historical returns data and benchmark data
returns_df = pd.read_excel(portfolio_data_file, sheet_name='Returns', index_col=0)
benchmark_df = pd.read_excel(portfolio_data_file, sheet_name='Benchmarks', index_col=0)

# ===============================
# DATA PREPROCESSING
# ===============================

# Convert all index dates to datetime and standardize to month-end timestamps
weights_df.index = pd.to_datetime(weights_df.index)
weights_df.index = weights_df.index.to_period('M').to_timestamp()  # Standardize to month-end

returns_df.index = pd.to_datetime(returns_df.index)
returns_df.index = returns_df.index.to_period('M').to_timestamp()  # Standardize to month-end

benchmark_df.index = pd.to_datetime(benchmark_df.index)
benchmark_df.index = benchmark_df.index.to_period('M').to_timestamp()  # Standardize to month-end

# Shift returns forward by one month to use future month returns
# This ensures we're using weights from current month to predict next month's returns
returns_df_shifted = returns_df.copy()  # No shift, align weights and returns to same month
benchmark_df_shifted = benchmark_df.copy()  # No shift

# Find common dates between weights and returns datasets
# We need dates that exist in both weights and returns datasets
weights_dates = set(weights_df.index)
returns_dates = set(returns_df.index)  # Use unshifted returns for date alignment
common_dates = sorted(list(weights_dates.intersection(returns_dates)))

# Remove the last date since it will have NaN returns after shifting forward
if len(common_dates) > 0:
    common_dates = common_dates[:-1]

# Use all available dates with weights and returns - don't filter based on a fixed date
first_available_date = common_dates[0] if common_dates else None

print(f"\nAnalysis period: {common_dates[0]} to {common_dates[-1]}")
print(f"Number of months: {len(common_dates)}")

# ===============================
# PORTFOLIO RETURN CALCULATION
# ===============================

print("\nCalculating portfolio returns...")
# Initialize array to store portfolio returns
portfolio_returns = np.zeros(len(common_dates))

# Calculate portfolio returns for each date in the common dates
for i, date in enumerate(common_dates):
    # Get weights for this date
    weights = weights_df.loc[date]
    
    # Get returns for this same date (which are already next month's returns in the original data)
    next_returns = returns_df.loc[date]
    
    # Calculate weighted return for this date
    # Only use countries that have both weights and returns data
    common_countries = set(weights.index).intersection(next_returns.index)
    
    # Skip if no common countries
    if len(common_countries) == 0:
        portfolio_returns[i] = np.nan
        continue
    
    # Calculate weighted return
    weighted_return = 0
    total_weight = 0
    
    for country in common_countries:
        if not np.isnan(next_returns[country]) and weights[country] > 0:
            weighted_return += weights[country] * next_returns[country]
            total_weight += weights[country]
    
    # Normalize by total weight to account for missing data
    if total_weight > 0:
        portfolio_returns[i] = weighted_return / total_weight
    else:
        portfolio_returns[i] = np.nan

# Create results DataFrame with portfolio and benchmark returns
results = pd.DataFrame({
    'Portfolio': portfolio_returns,
    'Equal Weight': benchmark_df.loc[common_dates, 'equal_weight']
})

# Calculate net returns (active returns = portfolio minus benchmark)
results['Net Return'] = results['Portfolio'] - results['Equal Weight']

# Verify that net return calculation is correct
portfolio_mean = results['Portfolio'].mean() * 12 * 100
benchmark_mean = results['Equal Weight'].mean() * 12 * 100
expected_net_mean = portfolio_mean - benchmark_mean

# Calculate cumulative returns (growth of $1 invested)
cumulative_returns = (1 + results).cumprod()
cumulative_net = (1 + results['Net Return']).cumprod()

# ===============================
# PERFORMANCE STATISTICS
# ===============================

# Function to calculate key performance metrics
def calculate_stats(returns):
    stats = {}
    stats['Annual Return'] = returns.mean() * 12 * 100  # Annualized return in %
    stats['Annual Vol'] = returns.std() * np.sqrt(12) * 100  # Annualized volatility in %
    stats['Sharpe Ratio'] = (returns.mean() * 12) / (returns.std() * np.sqrt(12))  # Annualized Sharpe
    stats['Max Drawdown'] = ((1 + returns).cumprod() / (1 + returns).cumprod().cummax() - 1).min() * 100  # Max drawdown in %
    stats['Hit Rate'] = (returns > 0).mean() * 100  # Percentage of positive months
    stats['Skewness'] = returns.skew()  # Distribution skewness (asymmetry)
    stats['Kurtosis'] = returns.kurtosis()  # Distribution kurtosis (tail thickness)
    return pd.Series(stats)

# Calculate statistics for portfolio, benchmark, and net returns
stats = pd.DataFrame({
    'Portfolio': calculate_stats(results['Portfolio']),
    'Equal Weight': calculate_stats(results['Equal Weight']),
    'Net Return': calculate_stats(results['Net Return'])
})

# Verify that the net return annual return matches the difference between portfolio and benchmark
print(f"\nVerification of Net Return calculation:")
print(f"Portfolio Annual Return: {portfolio_mean:.6f}%")
print(f"Equal Weight Annual Return: {benchmark_mean:.6f}%")
print(f"Expected Net Annual Return: {expected_net_mean:.6f}%")
print(f"Calculated Net Annual Return: {stats.loc['Annual Return', 'Net Return']:.6f}%")
print(f"Difference: {abs(expected_net_mean - stats.loc['Annual Return', 'Net Return']):.6f}%")

# ===============================
# VISUALIZATION
# ===============================

# Create multi-panel figure for performance visualization
plt.figure(figsize=(15, 15))

# Plot 1: Cumulative Total Returns - Compare portfolio to benchmark
plt.subplot(3, 1, 1)
cumulative_returns['Portfolio'].plot(label='Portfolio', color='blue')
cumulative_returns['Equal Weight'].plot(label='Equal Weight', color='red', alpha=0.7)
plt.title('Cumulative Total Returns')
plt.ylabel('Cumulative Return')
plt.grid(True)
plt.legend()

# Plot 2: Cumulative Net Returns - Show excess return over benchmark
plt.subplot(3, 1, 2)
cumulative_net.plot(label='Cumulative Net Return', color='green')
plt.axhline(y=1, color='r', linestyle='--', alpha=0.3)  # Reference line at 1.0
plt.title('Cumulative Net Returns (Portfolio - Equal Weight)')
plt.ylabel('Cumulative Return')
plt.grid(True)
plt.legend()

# Plot 3: Monthly Net Returns - Show monthly active returns
plt.subplot(3, 1, 3)
results['Net Return'].plot(kind='bar', color='blue', alpha=0.6)
plt.axhline(y=0, color='r', linestyle='--', alpha=0.3)  # Reference line at 0
plt.title('Monthly Net Returns')
plt.ylabel('Monthly Return')
plt.grid(True)
plt.xticks(rotation=45)

# Save the figure with high resolution
plt.tight_layout()
plt.savefig('Final_Portfolio_Returns.pdf', format='pdf', dpi=300, bbox_inches='tight')
plt.close()

# ===============================
# RESULTS SAVING
# ===============================

print("\nSaving results...")
output_file = 'Final_Portfolio_Returns.xlsx'
with pd.ExcelWriter(output_file) as writer:
    # Sheet 1: Monthly returns for all strategies
    results.to_excel(writer, sheet_name='Monthly Returns')
    
    # Sheet 2: Cumulative returns over time
    cumulative_returns.to_excel(writer, sheet_name='Cumulative Returns')
    
    # Sheet 3: Performance statistics
    stats.to_excel(writer, sheet_name='Statistics')
    
    # Sheet 4: Net returns analysis
    pd.DataFrame({
        'Net Return': results['Net Return'],
        'Cumulative Net': cumulative_net
    }).to_excel(writer, sheet_name='Net Returns')

print(f"\nResults saved to {output_file}")

# ===============================
# DETAILED PERFORMANCE REPORTING
# ===============================

# Print performance statistics table
print("\nPortfolio Statistics:")
print("--------------------")
print(stats)

# Calculate additional net return statistics
net_returns = results['Net Return']
positive_months = (net_returns > 0).sum()
negative_months = (net_returns < 0).sum()
avg_positive = net_returns[net_returns > 0].mean() * 100  # Convert to percentage
avg_negative = net_returns[net_returns < 0].mean() * 100  # Convert to percentage

# Print detailed net return analysis
print("\nNet Return Analysis:")
print("-------------------")
print(f"Positive Months: {positive_months} ({positive_months/len(net_returns)*100:.1f}%)")
print(f"Negative Months: {negative_months} ({negative_months/len(net_returns)*100:.1f}%)")
print(f"Average Positive Return: {avg_positive:.2f}%")
print(f"Average Negative Return: {avg_negative:.2f}%")
print(f"Win/Loss Ratio: {abs(avg_positive/avg_negative):.2f}")  # Ratio of avg win to avg loss

# Print most recent net returns
print("\nMost Recent Net Returns:")
print(results['Net Return'].tail())

# Calculate and print correlation with benchmark
correlation = results.corr().iloc[0,1]
print(f"\nCorrelation with Equal Weight: {correlation:.2f}")

# Calculate tracking error (annualized standard deviation of active returns)
tracking_error = (results['Portfolio'] - results['Equal Weight']).std() * np.sqrt(12) * 100
print(f"Tracking Error: {tracking_error:.2f}%")

# Calculate information ratio (active return / tracking error)
active_return = results['Portfolio'].mean() - results['Equal Weight'].mean()
info_ratio = (active_return * 12) / (tracking_error / 100)
print(f"Information Ratio: {info_ratio:.2f}")

# Print most recent performance
print("\nMost Recent Returns:")
print(results.tail())
