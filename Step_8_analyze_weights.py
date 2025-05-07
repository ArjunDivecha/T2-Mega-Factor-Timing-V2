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

# Separate CS and TS factors
cs_factors = [factor for factor in avg_weights.index if '_CS' in factor]
ts_factors = [factor for factor in avg_weights.index if '_TS' in factor]

print(f"\nFound {len(cs_factors)} Cross-Sectional (CS) factors and {len(ts_factors)} Time-Series (TS) factors")

# Calculate the total weight for CS vs TS
cs_total_weight = sum(avg_weights[cs_factors])
ts_total_weight = sum(avg_weights[ts_factors])

print(f"\nTotal portfolio allocation:")
print(f"CS factors: {cs_total_weight:.2%}")
print(f"TS factors: {ts_total_weight:.2%}")

# Show top 10 factors overall
print("\nTop 10 factors by average weight:")
for i, (factor, weight) in enumerate(avg_weights.head(10).items()):
    print(f"{i+1}. {factor}: {weight:.4f} ({'CS' if '_CS' in factor else 'TS'})")

# Top 5 CS factors
print("\nTop 5 CS factors:")
cs_weights = avg_weights[cs_factors].head(5)
for i, (factor, weight) in enumerate(cs_weights.items()):
    print(f"{i+1}. {factor}: {weight:.4f}")

# Top 5 TS factors
print("\nTop 5 TS factors:")
ts_weights = avg_weights[ts_factors].head(5)
for i, (factor, weight) in enumerate(ts_weights.items()):
    print(f"{i+1}. {factor}: {weight:.4f}")

# Count how many of each factor type are in the top 20
top20 = avg_weights.head(20)
cs_in_top20 = sum(1 for factor in top20.index if '_CS' in factor)
ts_in_top20 = sum(1 for factor in top20.index if '_TS' in factor)

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

# Plot distribution of CS vs TS weights
plt.figure(figsize=(12, 6))
labels = ['CS Factors', 'TS Factors']
sizes = [cs_total_weight, ts_total_weight]
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
plt.axis('equal')
plt.title('Distribution of Portfolio Allocation Between CS and TS Factors')
plt.savefig('cs_vs_ts_allocation.pdf')
print("\nSaved CS vs TS allocation pie chart to cs_vs_ts_allocation.pdf")

if __name__ == "__main__":
    print("\nAnalysis complete.") 