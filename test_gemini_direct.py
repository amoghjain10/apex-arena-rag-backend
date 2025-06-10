import os
from dotenv import load_dotenv
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in .env file.")

genai.configure(api_key=GOOGLE_API_KEY)

model_name = "gemini-2.5-flash-preview-05-20" # Use the specific model ID

try:
    model = genai.GenerativeModel(model_name=model_name)
    # Simple text message
    prompt = "What types of events can I host?"
    print(f"Sending prompt to {model_name}: '{prompt}'")

    response = model.generate_content(
        prompt,
        generation_config=genai.GenerationConfig(temperature=0.7),
        safety_settings={
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }
    )

    print("\n--- Direct Gemini API Response ---")
    print(f"Text: {response.text}")
    print(f"Raw Parts: {response.candidates[0].content.parts}")
    if hasattr(response.candidates[0].content, 'parts') and response.candidates[0].content.parts:
        for i, part in enumerate(response.candidates[0].content.parts):
            print(f"  Part {i}: {part.text if hasattr(part, 'text') else part}")
            # You can add more checks here if other attributes like 'thought' appear
    print("---------------------------------")


except Exception as e:
    print(f"Error making direct Gemini API call: {e}")