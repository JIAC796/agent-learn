"""
Day 3: Function Calling 实操
目标：让 LLM 学会调用外部工具
核心理念：LLM 不动手，只动嘴——它告诉你"我想调哪个工具"，
         你来执行，然后把结果喂给它，它再生成最终回复。
"""

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

# ============================================================
# 第二步：定义工具（Tool Schema）
# LLM 通过这个描述知道"我能用什么工具"
# ============================================================
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取指定城市的实时天气信息，包括天气状况、温度和湿度",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名称，如 北京、上海、广州"
                    }
                },
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "执行数学计算，支持加减乘除和括号",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "数学表达式，如 '(15+23)*8'"
                    }
                },
                "required": ["expression"]
            }
        }
    }
]

# ============================================================
# 第三步：工具的实际实现（你自己写的函数）
# ============================================================
# ============================================================



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

# ============================================================
# 第四步：核心对话函数
# 这就是一个最简 Agent 的雏形：
#   用户输入 → LLM 判断是否需要工具 → 执行工具 → 喂回结果 → LLM 生成回复
# ============================================================
def call_deepseek(messages, with_tools=False):
    """调用 DeepSeek API，返回响应 JSON"""
    body = {
        "model": "deepseek-v4-flash",
        "messages": messages,
    }
    if with_tools:
        body["tools"] = tools

    resp = requests.post(
        DEEPSEEK_URL,
        headers={
            "Authorization": f"Bearer {DEEPSEEK_KEY}",
            "Content-Type": "application/json; charset=utf-8",
        },
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        timeout=30,
    )
    return resp.json()


def chat(user_input: str) -> str:
    """核心：用户输入 → LLM 判断 → 执行工具 → 喂回结果 → 最终回答"""
    messages = [{"role": "user", "content": user_input}]

    # 第 1 轮：问 LLM "你需要调工具吗？"
    data = call_deepseek(messages, with_tools=True)
    msg = data["choices"][0]["message"]

    # 情况 A：LLM 不需要工具，直接回答
    if "tool_calls" not in msg or not msg["tool_calls"]:
        return msg["content"]

    # 情况 B：LLM 要调工具，我们帮它执行
    print(f"\n🔧 LLM 想调用 {len(msg['tool_calls'])} 个工具：")
    messages.append(msg)

    for tc in msg["tool_calls"]:
        fn_name = tc["function"]["name"]
        fn_args = json.loads(tc["function"]["arguments"])
        print(f"   → {fn_name}({fn_args})")

        result = available_functions[fn_name](**fn_args)
        print(f"   ← 结果：{result}")

        messages.append({
            "role": "tool",
            "tool_call_id": tc["id"],
            "content": result,
        })

    # 第 2 轮：把工具结果喂给 LLM，让它生成最终回答
    data2 = call_deepseek(messages, with_tools=False)
    return data2["choices"][0]["message"]["content"]


# ============================================================
# 第五步：跑起来
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("🧪 Day 3: Function Calling Agent 交互式对话")
    print("   输入 quit 或 exit 退出")
    print("=" * 60)

    while True:
        try:
            user_input = input("\n👤 你：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 再见！")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("👋 再见！")
            break

        answer = chat(user_input)
        print(f"🤖 {answer}")