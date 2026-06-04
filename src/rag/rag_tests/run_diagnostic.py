# run_diagnostic.py  (drop it in root, delete after)
import os
from google import genai
from dotenv import load_dotenv

load_dotenv()
client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

print("=== Models supporting embedContent ===")
for model in client.models.list():
    name = model.name
    actions = getattr(model, 'supported_actions', None) or []
    if 'embedContent' in str(actions) or 'embed' in name.lower():
        print(f"  {name}  |  actions: {actions}")