# print_intensities.py
import pickle
import pandas as pd
import os

file_path = 'shrinkage_results.pkl'

if not os.path.exists(file_path):
    print(f"Error: File not found at {file_path}")
    exit(1)

print(f"Loading results from {file_path}...")
try:
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
except Exception as e:
    print(f"Error loading pickle file: {e}")
    exit(1)

intensities = data.get('shrinkage_intensities')
window_dates = data.get('window_dates')

if intensities is None:
    print("Error: 'shrinkage_intensities' key not found in the results file.")
    exit(1)

if window_dates is None:
    print("Warning: 'window_dates' key not found. Cannot display test end dates.")
    window_dates = {} # Create empty dict to avoid errors later

print("\nShrinkage Intensities (delta*) per Window:")
print("-" * 50)
print(f"{'Window Index':<12} | {'Test End Date':<13} | {'Shrinkage Intensity':<20}")
print("-" * 50)

# Sort by window index before printing
sorted_indices = sorted(intensities.keys())

for idx in sorted_indices:
    intensity = intensities[idx]
    date_info = window_dates.get(idx, {})
    test_end_date = date_info.get('test_end', None)
    
    date_str = 'N/A'
    if isinstance(test_end_date, pd.Timestamp):
        try:
            date_str = test_end_date.strftime('%Y-%m-%d')
        except ValueError: # Handle potential NaT values
             pass 
        
    print(f"{idx:<12} | {date_str:<13} | {intensity:<20.6f}")

print("-" * 50)
print(f"\nTotal windows processed: {len(intensities)}")
