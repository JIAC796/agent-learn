from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

# 1. 把 messages 提到外面，system prompt 只设一次
messages = [
    {"role": "system", "content": "你是一个温柔且专业的猫娘AI Agent编程导师"}
]

# 2. 套一个 while True，让用户一直聊
print("🐱 猫娘导师已上线！输入 quit 退出\n")

while True:
    user_input = input("👤 你：").strip()
    if user_input.lower() in ("quit", "exit", "q"):
        print("🐱 再见喵~")
        break
    if not user_input:
        continue

    # 3. 用户消息追加到 messages
    messages.append({"role": "user", "content": user_input})

    # 4. 把整个 messages 发过去
    response = client.chat.completions.create(
        model="deepseek-v4-flash",
        messages=messages
    )

    reply = response.choices[0].message.content
    print(f"🐱 {reply}")

    # 5. AI 回复也追加到 messages
    messages.append({"role": "assistant", "content": reply})