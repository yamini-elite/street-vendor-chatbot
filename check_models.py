import groq
import os
from dotenv import load_dotenv

load_dotenv()

# Load API key from environment
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    print("Please set GROQ_API_KEY environment variable")
    exit(1)

client = groq.Groq(api_key=api_key)

# List available models
try:
    models = client.models.list()
    print("Available Groq models:")
    for model in models.data:
        print(f"- {model.id}")
        print()
except Exception as e:
    print(f"Error listing models: {e}")