import json
import os
import requests
from dotenv import load_dotenv

load_dotenv()   # 从 .env 读 key，源码里不留密钥

# ============================================================
# 第一步：API 配置
# ============================================================
DEEPSEEK_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"


# 1. 工具函数

def get_weather(city: str) -> str:
    """通过 wttr.in 获取天气，完全免费，无需 Key"""
    try:
        url = f"https://wttr.in/{city}?format=j1"
        resp = requests.get(url, timeout=10)
        data = resp.json()

        current = data["current_condition"][0]
        return (
            f"{city}：{current['weatherDesc'][0]['value']}，"
            f"{current['temp_C']}°C，"
            f"体感 {current['FeelsLikeC']}°C，"
            f"湿度 {current['humidity']}%，"
            f"风速 {current['windspeedKmph']}km/h"
        )
    except Exception as e:
        return f"❌ 天气查询失败：{e}"


def calculate(expression: str) -> str:
    """安全计算器"""
    try:
        # 安全限制：只允许数字和基本运算符
        allowed = set("0123456789+-*/().% ")
        if not all(c in allowed for c in expression):
            return "表达式包含不允许的字符"
        result = eval(expression)
        return f"{expression} = {result}"
    except Exception as e:
        return f"计算出错：{e}"

    # 工具名 → 函数 映射表


available_functions = {
    "get_weather": get_weather,
    "calculate": calculate,
}
# 2. 工具描述文本
TOOLS_DESC = """ 
可用工具：
1. get_weather: 查询指定城市的实时天气，参数 city（城市名，如 北京、上海）
2. calculate: 执行数学计算，参数 expression（算式字符串，如 "35*1.5"）
"""

# 3. Prompt 模板
REACT_PROMPT = """ 你是一个能使用工具的智能助手。请按以下格式回复：

Thought: 你的思考过程
Action: 工具名称
Action Input: 参数

当你得到足够信息后，用以下格式给出最终答案：
Final Answer: 你的回答

可用工具：
{TOOLS_DESC}

现在开始。用户的问题是：{question}
"""


def call_deepseek(messages):
    """调用 DeepSeek API，返回响应 JSON"""
    body = {
        "model": "deepseek-v4-flash",
        "messages": messages,
    }

    resp = requests.post(
        DEEPSEEK_URL,
        headers={
            "Authorization": f"Bearer {DEEPSEEK_KEY}",
            "Content-Type": "application/json; charset=utf-8",
        },
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        timeout=30,
    )
    data = resp.json()
    return data["choices"][0]["message"]["content"]


# 4. 解析函数
def parse_react(response):
    """
    解析 LLM 回复，返回：
    - ("final", answer)       → 出了最终答案
    - ("action", tool, inp)   → 要调工具
    - ("thought", text)       → 纯思考，继续
    """
    # 先找 Final Answer
    if "Final Answer:" in response:
        answer = response.split("Final Answer:")[-1].strip()
        return ("final", answer)

    # 再找 Action + Action Input
    if "Action:" in response and "Action Input:" in response:
        lines = response.strip().split("\n")
        tool = ""
        inp = ""
        for line in lines:
            if line.startswith("Action:"):
                tool = line.replace("Action:", "").strip()
            if line.startswith("Action Input:"):
                inp = line.replace("Action Input:", "").strip()
        if tool and inp:
            return ("action", tool, inp)

    # 都不是 → 纯思考
    return ("thought", response)


# 5. 主循环
def react_agent(question):
    """ReAct 主循环"""
    # 构造初始 Prompt
    prompt = REACT_PROMPT.format(TOOLS_DESC=TOOLS_DESC, question=question)
    messages = [{"role": "user", "content": prompt}]

    for i in range(10):  # 最多 10 轮
        print(f"\n--- 第 {i + 1} 轮 ---")

        # 1. 调 LLM
        response = call_deepseek(messages)
        print(f"LLM 回复：\n{response}")

        # 2. 解析
        result = parse_react(response)

        # 3. 判断类型
        if result[0] == "final":
            return result[1]

        elif result[0] == "action":
            tool_name, tool_input = result[1], result[2]
            print(f"🔧 执行 {tool_name}({tool_input})")
            obs = available_functions[tool_name](tool_input)
            print(f"📋 结果：{obs}")

            # 把 LLM 回复 + 工具结果追加到历史
            messages.append({"role": "assistant", "content": response})
            messages.append({"role": "user", "content": f"Observation: {obs}"})

        else:
            # 纯思考，把回复追加进去继续
            messages.append({"role": "assistant", "content": response})

    return "超过最大轮数，请重试"


# 跑起来
if __name__ == "__main__":
    print("🤖 Day 4: ReAct Agent\n")
    while True:
        q = input("👤 你：").strip()
        if q.lower() in ("quit", "exit", "q"):
            print("👋 再见！")
            break
        if not q:
            continue
        answer = react_agent(q)
        print(f"\n✅ 最终回答：{answer}\n")
