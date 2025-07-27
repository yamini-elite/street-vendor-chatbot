from dotenv import load_dotenv
import os, pathlib

env_path = pathlib.Path(__file__).parent / ".env"  # robust absolute path
load_dotenv(dotenv_path=env_path)

print("API Key loaded:", bool(os.getenv("WATSONX_APIKEY")))
print("URL loaded:", bool(os.getenv("WATSONX_URL")))
print("Project ID loaded:", bool(os.getenv("PROJECT_ID")))
