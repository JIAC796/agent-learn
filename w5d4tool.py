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

# ① 定义工具：函数变"外挂"，docstring 是给 AI 看的使用说明
@tool
def calculate(表达式: str) -> str:
    """计算数学表达式，例如 '3*7+2' 返回 23"""
    return str(eval(表达式))

@tool
def current_time() -> str:
    """查询当前日期和时间"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
@tool
def translator(text:str) -> str:
    """将中文翻译为英文，例如，‘中国’ 返回 china"""
    return llm.invoke([HumanMessage(content=f"把 '{text}' 翻译成英文，只输出结果")]).content
# ② 绑定工具：让模型知道能用这俩
llm_tools = llm.bind_tools([calculate, current_time,translator])

# ③ 问一个需要"动手"的问题
messages = [HumanMessage(content="现在几点了？顺便帮我算 15*4+8 等于多少,苹果的英文是什么？")]

# ④ 第一轮：模型"决定"要不要调day工具
result = llm_tools.invoke(messages)
print("模型决定调用：", result.tool_calls if result.tool_calls else "直接回答", "\n")

# ⑤ 有调用就执行，结果塞回对话
if result.tool_calls:
    messages.append(result)          # 带 tool_calls 的 AI 消息放回历史
    for tc in result.tool_calls:
        if tc["name"] == "calculate":
            out = calculate.invoke(tc["args"])
        elif tc["name"] == "current_time":
            out = current_time.invoke(tc["args"])
        elif tc["name"] == "translator":
            out = translator.invoke(tc["args"])
        else:
            out = "未知工具"
        messages.append(ToolMessage(content=str(out), tool_call_id=tc["id"]))
    # ⑥ 让模型结合工具结果做最终回答
    final = llm_tools.invoke(messages)
    print("最终回答：", final.content)