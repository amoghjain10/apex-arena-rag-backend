import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in .env file.")

genai.configure(api_key=GOOGLE_API_KEY)

try:
    for m in genai.list_models():
        if "generateContent" in m.supported_generation_methods:
            print(f"Model ID: {m.name}, Supported methods: {m.supported_generation_methods}")
        if m.name == "gemini-pro":
            print(f"Found gemini-pro! Details: {m}")
except Exception as e:
    print(f"Error listing models: {e}")