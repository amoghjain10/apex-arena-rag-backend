import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Try to get the GOOGLE_API_KEY
api_key = os.getenv("GOOGLE_API_KEY")

if api_key:
    print(f"Successfully loaded GOOGLE_API_KEY: {api_key[:5]}...{api_key[-5:]}") # Print only first/last 5 chars for security
    print(f"Full key length: {len(api_key)}")
else:
    print("Failed to load GOOGLE_API_KEY from .env file.")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Files in current directory: {os.listdir()}")