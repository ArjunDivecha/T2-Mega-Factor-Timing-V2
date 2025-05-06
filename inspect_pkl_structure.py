# inspect_pkl_structure.py
import pickle
import os

SHRINKAGE_RESULTS_FILE = "shrinkage_results.pkl"

print(f"Loading {SHRINKAGE_RESULTS_FILE} to inspect structure...")
if not os.path.exists(SHRINKAGE_RESULTS_FILE):
    print(f"Error: {SHRINKAGE_RESULTS_FILE} not found.")
    exit(1)

try:
    with open(SHRINKAGE_RESULTS_FILE, 'rb') as f:
        data = pickle.load(f)
except Exception as e:
    print(f"Error loading pickle file: {e}")
    exit(1)

print(f"Type of loaded data: {type(data)}")

if isinstance(data, dict):
    print("\nTop-level keys:")
    for key in data.keys():
        print(f"- {key} (Type: {type(data[key])})")
        # If the value is another dictionary, print its keys too (limited check)
        if isinstance(data[key], dict) and data[key]:
             first_inner_key = next(iter(data[key]))
             print(f"  - Example inner key: {first_inner_key} (Type: {type(data[key][first_inner_key])})")
             # If the inner value is a DataFrame, print columns
             if hasattr(data[key][first_inner_key], 'columns'):
                  print(f"    - Columns: {list(data[key][first_inner_key].columns[:5])}...") # Show first 5 columns

elif isinstance(data, list):
    print(f"\nData is a list with {len(data)} elements.")
    if data:
        print(f"Type of first element: {type(data[0])})")
        if isinstance(data[0], dict):
             print("Keys in the first list element (dictionary):")
             for key in data[0].keys():
                 print(f"- {key} (Type: {type(data[0][key])})")

else:
    print("\nData is not a dictionary or list.")
