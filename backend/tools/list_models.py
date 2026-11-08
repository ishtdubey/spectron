# list_models.py
import os
import asyncio
from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

async def get_active_models():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("❌ Error: GROQ_API_KEY is not set.")
        return

    client = AsyncOpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=api_key
    )

    try:
        models = await client.models.list()
        print("✅ Active models available on your Groq account:\n")
        for model in models.data:
            print(f" - {model.id}")
    except Exception as e:
        print(f"❌ Error fetching models: {e}")

if __name__ == "__main__":
    asyncio.run(get_active_models())