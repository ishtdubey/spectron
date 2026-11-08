import os
import asyncio
from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

async def test_connection():
    api_key = os.getenv("GROQ_API_KEY")
    
    if not api_key:
        print("❌ Error: GROQ_API_KEY not found in environment variables or .env file.")
        return

    print("🔑 Found GROQ_API_KEY. Testing connection to Groq API...")

    client = AsyncOpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=api_key
    )

    try:
        response = await client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[{"role": "user", "content": "Reply with 'Groq is connected!' if you receive this."}],
            max_tokens=20
        )
        print("✅ SUCCESS! Groq Response:")
        print(response.choices[0].message.content)

    except Exception as e:
        print("❌ FAILED! Connection error:")
        print(e)

if __name__ == "__main__":
    asyncio.run(test_connection())