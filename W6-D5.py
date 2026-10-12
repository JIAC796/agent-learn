import os
from dotenv import load_dotenv
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage, SystemMessage
from pydantic import BaseModel
from typing import List

load_dotenv()

llm = ChatOpenAI(
    model="deepseek-v4-flash",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
    model_kwargs={"extra_body": {"thinking": {"type": "disabled"}}},
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
# ② 绑定工具：让模型知道能用这俩

llm_tools = llm.bind_tools([calculate, current_time])

# 3. messages 累积
messages = [SystemMessage("你是一个猫娘编程导师，用通俗语言回答")]

# 4. 主循环（Day4 ask() 升级版）
while True:
    user_input = input("你: ")
    if user_input == "exit": break
    messages.append(HumanMessage(user_input))

    # ⑤ Agent循环（这里用流式）
    while True:
        final_msg = None
        for chunk in llm_tools.stream(messages):
            if chunk.content:
                print(chunk.content, end="", flush=True)
            final_msg = chunk if final_msg is None else final_msg + chunk
        print()  # 换行

        messages.append(final_msg)

        if final_msg.tool_calls:
            for tc in final_msg.tool_calls:
                # 根据name分发到对应工具
                tool_map = {"calculate": calculate, "current_time": current_time}
                result = tool_map[tc["name"]].invoke(tc["args"])
                messages.append(ToolMessage(content=result, tool_call_id=tc["id"]))
        else:
            break  # 本轮回答结束，回到外层 while 等下一次提问


# 6. 退出时结构化总结
class SessionSummary(BaseModel):
    问题数量: int
    涉及主题: List[str]
    是否用到工具: bool

llm_struct = llm.with_structured_output(SessionSummary, method="function_calling")
SessionSummary = llm_struct.invoke(messages)
print(SessionSummary)