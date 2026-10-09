import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("NVIDIA_API_KEY"),
    timeout=30
)

print("Calling NVIDIA...")

response = client.chat.completions.create(
    model="deepseek-ai/deepseek-v4.1-flash",
    messages=[
        {
            "role": "user",
            "content": "Reply with exactly: NVIDIA API working"
        }
    ],
    max_tokens=50,
    temperature=0
)

print("NVIDIA responded:")
print(response.choices[0].message.content)