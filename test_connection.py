"""
Step 1.9 - Check that the OpenAI key works and list the models it can use.
"""
from dotenv import load_dotenv
from openai import OpenAI, AuthenticationError, APIConnectionError

# 1. Read .env and load OPENAI_API_KEY into the environment
load_dotenv()

# 2. Create the client - it finds OPENAI_API_KEY automatically
client = OpenAI()

try:
    # 3. Ask OpenAI which models this key can access (free call)
    models = client.models.list()
    names = sorted(m.id for m in models)

    # 4. Keep only chat models: "gpt-..." or "o" + digit (e.g. o3, o4-mini)
    chat_models = [
        n for n in names
        if n.startswith("gpt-") or (n[0] == "o" and n[1:2].isdigit())
    ]

    print(f"Key works! It can access {len(names)} models in total.")
    print(f"Chat models ({len(chat_models)}):")
    for name in chat_models:
        print("  -", name)

except AuthenticationError:
    print("ERROR: The key was rejected. Check .env for typos, or ask your manager if the key is active.")
except APIConnectionError:
    print("ERROR: Could not reach OpenAI. Check your internet, VPN, or company firewall.")