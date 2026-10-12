import os
from dotenv import load_dotenv
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage

load_dotenv()

llm = ChatOpenAI(
    model="deepseek-v4-flash",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)

@tool
def calculate(表达式: str) -> str:
    """计算数学表达式，例如 '3*7+2' 返回 23"""
    return str(eval(表达式))

print( calculate)

llm_with_tools = llm.bind_tools([calculate])






# 3. 打印结果
print(llm_with_tools)
# 组装完整对话：问题 → 模型说要调工具 → 工具结果
# 全局对话记录，所有历史都存这里
messages = []

def ask(question):
    # 1. 把你的问题加入对话
    messages.append(HumanMessage(question))

    # 2. Agent 循环：模型说要工具就给，直到它不再要
    while True:
        result = llm_with_tools.invoke(messages)
        messages.append(result)  # 模型的话也记入对话

        if result.tool_calls:    # 模型说：我要用工具
            for tool_call in result.tool_calls:
                tool_result = calculate.invoke(tool_call["args"])
                messages.append(ToolMessage(
                    content=tool_result,
                    tool_call_id=tool_call["id"],
                ))
            # 不 break，继续循环，把工具结果喂回去再问
        else:                    # 模型不再要工具，直接给答案
            print(result.content)
            break

# 问第一次
ask("318*62-5+89+77+65+25=?")

# 问第二次（现在可以了！）
ask("那再算一下 123*456=?")