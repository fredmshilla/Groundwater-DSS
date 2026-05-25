import warnings
warnings.filterwarnings('ignore') # Silence the deprecation warning
import google.generativeai as genai

# Hardcode the key directly to completely bypass Windows .env blocking
API_KEY = "PASTE_YOUR_KEY_HERE"

if API_KEY == "AIzaSyA-ZVQ2jhbmVilJbHLdwE4YaKX-qBGKQ1M":
    print("🚨 You forgot to paste your real API key into the script!")
    exit()

# Configure the library
genai.configure(api_key=API_KEY)

print("📡 Connecting to Google AI Studio...")
print("✅ Found the following models supported for text generation:\n")

# Call ListModels
try:
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(f"   - {m.name}")
    print("\nScript complete. Copy the list above and send it back!")
except Exception as e:
    print(f"🚨 Failed to list models. Error: {e}")