import sys
print(f"Python version: {sys.version}")
print(f"Python path: {sys.path}")

try:
    import anthropic
    print(f"Anthropic package found and imported successfully!")
    print(f"Anthropic version: {anthropic.__version__}")
except ImportError as e:
    print(f"Error importing anthropic: {e}")
    
    # Try to find the package
    import subprocess
    result = subprocess.run(["pip", "show", "anthropic"], capture_output=True, text=True)
    print("\nPip show anthropic:")
    print(result.stdout)
    
    # Check if the package is in site-packages
    import os
    for path in sys.path:
        if "site-packages" in path:
            anthropic_path = os.path.join(path, "anthropic")
            print(f"Checking {anthropic_path}: {os.path.exists(anthropic_path)}")
