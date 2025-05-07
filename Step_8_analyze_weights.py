#!/usr/bin/env python
"""
# Analyze Factor Weights
#
# INPUT FILES:
# - factor_timing_results.xlsx: Excel output file with factor weights
#
# Analyzes the factor weights from the factor timing results, providing:
# - Top factors by average weight
# - Distribution of CS vs TS factors in the portfolio
# - Performance differences between CS and TS factors
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# Load the long-only weights
print("Loading long-only weights...")
df_long = pd.read_excel('factor_timing_results.xlsx', sheet_name='Long-Only Weights')

# Set the date column as index
if 'Date' in df_long.columns:
    df_long.set_index('Date', inplace=True)
elif 'Unnamed: 0' in df_long.columns:
    df_long.set_index('Unnamed: 0', inplace=True)
    df_long.index.name = 'Date'

# Calculate average weights across all periods
avg_weights = df_long.mean().sort_values(ascending=False)

# Define known CS and TS factors based on their names
# Cross-sectional factors are typically stock-specific metrics
known_cs_factors = [
    'Best PE', 'Best PE ', 'Positive PE', 'Positive PE ', 'Best PBK', 'Best Price Sales',
    'Best Div Yield', 'Best Cash Flow', 'Best ROE', 'Operating Margin', 'Debt to EV',
    'EV to EBITDA', 'Earnings Yield', 'Trailing PE', 'Shiller PE', 'P2P', 'LT Growth'
]

# Time-series factors are typically macro or market-wide indicators
known_ts_factors = [
    'REER', 'Inflation', 'Debt to GDP', 'GDP', 'Budget Def', 'Current Account',
    '10Yr Bond', '10Yr Bond 12', '1MTR', '3MTR', '12MTR', '12-1MTR',
    'Advance Decline', 'RSI14', '20 Day Vol', '360 Day Vol', '120MA Signal',
    'Currency Vol', 'Currency 12', 'Oil 12', 'Gold 12', 'Copper 12', 'Agriculture 12',
    'Bloom Country Risk', 'Trailing EPS 36'
]

# Categorize factors based on known lists
cs_factors = [factor for factor in avg_weights.index if any(cs_name in factor for cs_name in known_cs_factors)]
ts_factors = [factor for factor in avg_weights.index if any(ts_name in factor for ts_name in known_ts_factors)]

# Handle any uncategorized factors
uncategorized = [factor for factor in avg_weights.index if factor not in cs_factors and factor not in ts_factors]
if uncategorized:
    print(f"\nUncategorized factors: {uncategorized}")
    # Add uncategorized factors to TS by default
    ts_factors.extend(uncategorized)

print(f"\nFound {len(cs_factors)} Cross-Sectional (CS) factors and {len(ts_factors)} Time-Series (TS) factors")

# Calculate the total weight for CS vs TS
cs_total_weight = sum(avg_weights[cs_factors]) if cs_factors else 0
ts_total_weight = sum(avg_weights[ts_factors]) if ts_factors else 0

print(f"\nTotal portfolio allocation:")
print(f"CS factors: {cs_total_weight:.2%}")
print(f"TS factors: {ts_total_weight:.2%}")

# Show top 10 factors overall
print("\nTop 10 factors by average weight:")
for i, (factor, weight) in enumerate(avg_weights.head(10).items()):
    factor_type = 'CS' if '_CS' in factor else 'TS'
    print(f"{i+1}. {factor}: {weight:.4f} ({factor_type})")

# Top 5 CS factors
print("\nTop 5 CS factors:")
if cs_factors:
    cs_weights = avg_weights[cs_factors].head(5)
    for i, (factor, weight) in enumerate(cs_weights.items()):
        print(f"{i+1}. {factor}: {weight:.4f}")
else:
    print("No CS factors found")

# Top 5 TS factors
print("\nTop 5 TS factors:")
if ts_factors:
    ts_weights = avg_weights[ts_factors].head(5)
    for i, (factor, weight) in enumerate(ts_weights.items()):
        print(f"{i+1}. {factor}: {weight:.4f}")
else:
    print("No TS factors found")

# Count how many of each factor type are in the top 20
top20 = avg_weights.head(20)
cs_in_top20 = sum(1 for factor in top20.index if '_CS' in factor)
ts_in_top20 = sum(1 for factor in top20.index if '_TS' in factor)

# If no CS/TS pattern found, all factors in top20 are considered TS
if len(cs_factors) == 0 and len(ts_factors) == 0:
    ts_in_top20 = len(top20)

print(f"\nIn the top 20 factors:")
print(f"CS factors: {cs_in_top20}")
print(f"TS factors: {ts_in_top20}")

# Strip the _CS and _TS suffixes to group by base factor name
def strip_suffix(factor_name):
    if '_CS' in factor_name:
        return factor_name.replace('_CS', '')
    elif '_TS' in factor_name:
        return factor_name.replace('_TS', '')
    return factor_name

# Create a map of base factors to their CS and TS variants
base_factors = {}
for factor in avg_weights.index:
    base = strip_suffix(factor)
    if base not in base_factors:
        base_factors[base] = {"CS": None, "TS": None}
    
    if '_CS' in factor:
        base_factors[base]["CS"] = factor
    elif '_TS' in factor:
        base_factors[base]["TS"] = factor

# Find factors that have both CS and TS variants
paired_factors = {base: variants for base, variants in base_factors.items() 
                 if variants["CS"] is not None and variants["TS"] is not None}

print(f"\nFound {len(paired_factors)} factors with both CS and TS variants")

# Compare weights of CS vs TS for the same base factor
print("\nCS vs TS weight comparison for paired factors:")
for base, variants in sorted(paired_factors.items(), 
                            key=lambda x: abs(avg_weights.get(x[1]["CS"], 0) - 
                                             avg_weights.get(x[1]["TS"], 0)), 
                            reverse=True)[:10]:
    cs_weight = avg_weights.get(variants["CS"], 0)
    ts_weight = avg_weights.get(variants["TS"], 0)
    diff = cs_weight - ts_weight
    stronger = "CS" if diff > 0 else "TS"
    print(f"{base}: CS={cs_weight:.4f}, TS={ts_weight:.4f}, Diff={abs(diff):.4f} ({stronger} stronger)")

# Plot distribution of CS vs TS weights only if we have both types
if cs_total_weight > 0 or ts_total_weight > 0:
    plt.figure(figsize=(12, 6))
    labels = []
    sizes = []
    
    if cs_total_weight > 0:
        labels.append('CS Factors')
        sizes.append(cs_total_weight)
        
    if ts_total_weight > 0:
        labels.append('TS Factors')
        sizes.append(ts_total_weight)
    
    if len(sizes) > 0:
        plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
        plt.axis('equal')
        plt.title('Distribution of Portfolio Allocation Between Factor Types')
        plt.savefig('factor_allocation.pdf')
        print("\nSaved factor allocation pie chart to factor_allocation.pdf")
    else:
        print("\nNo weights to plot in pie chart")

if __name__ == "__main__":
    print("\nAnalysis complete.") 