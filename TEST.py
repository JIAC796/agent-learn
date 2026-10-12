from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

response = client.chat.completions.create(
    model="deepseek-v4-flash",
    messages=[
        {"role": "system", "content": "你是一个猫娘ai agent编程导师"},
        {"role": "user", "content": "Function Calling函数调用 框架是什么？"}

    ]
)
print(response.choices[0].message.content)