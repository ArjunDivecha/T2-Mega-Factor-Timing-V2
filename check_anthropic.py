import sys
import os

print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")
print(f"Python path: {sys.path}")

# Try to find anthropic in site-packages
for path in sys.path:
    if "site-packages" in path:
        anthropic_path = os.path.join(path, "anthropic")
        print(f"Checking {anthropic_path}: {os.path.exists(anthropic_path)}")

# Try to import anthropic
try:
    import anthropic
    print(f"Successfully imported anthropic version {anthropic.__version__}")
except ImportError as e:
    print(f"Failed to import anthropic: {e}")

# Try to import with a direct path
try:
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "anthropic", 
        "/opt/anaconda3/lib/python3.12/site-packages/anthropic/__init__.py"
    )
    if spec:
        anthropic_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(anthropic_module)
        print(f"Successfully imported anthropic directly: {anthropic_module.__version__}")
    else:
        print("Spec not found for anthropic")
except Exception as e:
    print(f"Failed to import anthropic directly: {e}")
