import os
import dotenv

# Print the current working directory
print(f"Current working directory: {os.getcwd()}")

# Check if .env file exists
env_path = os.path.join(os.getcwd(), '.env')
print(f"Checking for .env file at: {env_path}")
print(f"File exists: {os.path.exists(env_path)}")

# Try to load the .env file
print("\nAttempting to load .env file...")
dotenv.load_dotenv()

# Check if the API key is loaded
api_key = os.environ.get("ANTHROPIC_API_KEY")
if api_key:
    print(f"API key loaded successfully!")
    print(f"API key starts with: {api_key[:15]}...")
    print(f"API key length: {len(api_key)}")
else:
    print("API key not found in environment variables")

# Print all environment variables (excluding values for security)
print("\nAll environment variables:")
for key in os.environ:
    if "API" in key or "KEY" in key or "SECRET" in key:
        print(f"{key}: [REDACTED]")
    else:
        print(f"{key}: {os.environ[key]}")
