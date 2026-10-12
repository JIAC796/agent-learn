import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv()

llm = ChatOpenAI(
    model="deepseek-v4-flash",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)

messages = [SystemMessage(content="你是猫娘导师，回答温柔但专业。")]

print("猫娘导师上线啦～ 输入 q 退出")
while True:
    user_input = input("你：")
    if user_input.lower() in ("q", "quit", "退出"):
        print("猫娘导师：拜拜～ 记得复习今天的多轮对话哦 ✨")
        break

    messages.append(HumanMessage(content=user_input))
    result = llm.invoke(messages)
    messages.append(AIMessage(content=result.content))
    print(f"猫娘导师：{result.content}")