import os
import dotenv
from anthropic import Anthropic

# Load environment variables
dotenv.load_dotenv()

# Get API key
api_key = os.environ.get("ANTHROPIC_API_KEY")
print(f"API key loaded (first 10 chars): {api_key[:10]}...")

# Initialize Anthropic client
client = Anthropic(api_key=api_key)

# Test API call
try:
    response = client.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=100,
        messages=[{"role": "user", "content": "Hello, Claude!"}]
    )
    print("API call successful!")
    print(f"Response: {response.content[0].text}")
except Exception as e:
    print(f"API call failed: {e}")
