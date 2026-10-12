---
AIGC:
    Label: "1"
    ContentProducer: 001191110102MACQD9K64018705
    ProduceID: 7627154608607002915-data_volume/files/所有对话/主对话/复习笔记_AI_Agent_Day1-4.md
    ReservedCode1: ""
    ContentPropagator: 001191110102MACQD9K64028705
    PropagateID: 2490864430229212#1789819211624
    ReservedCode2: ""
---
# AI Agent 开发复习笔记（Day 1-4）

> 涵盖 W1-4 筑基+分水岭阶段全部核心内容。代码只写关键骨架，完整版在本地。

---

## Day 1：跑通第一个 LLM 调用

### 核心概念
- **LLM API**：通过 HTTP 请求把消息发给模型，模型返回文本
- **messages**：对话的组织方式，按 `system → user → assistant → user → ...` 顺序排列
- **system prompt**：设定 AI 的角色、语气、行为规则
- **temperature**：控制输出随机性，0=严谨确定，1.5=天马行空

### 关键代码骨架
```python
from openai import OpenAI

client = OpenAI(
    api_key="sk-xxx",
    base_url="https://api.deepseek.com"
)

response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": "你是一个毒舌但有用的编程导师"},
        {"role": "user", "content": "Python列表推导式怎么写？"}
    ]
)
print(response.choices[0].message.content)
```

### 踩过的坑
- 装 `openai` 和 `python-dotenv` 两个包
- `.env` 文件放项目根目录
- DeepSeek 的 base_url 是 `https://api.deepseek.com`

---

## Day 2：Prompt 工程三大核心技巧

### 核心认知
> **Prompt 不是聊天，是编程。** 指令越精确、越有结构，输出就越可控。

### 技巧1：结构化输出（让 AI 返回 JSON）
```python
{"role": "system", "content": """
你是一个商品分析师。返回JSON格式：
{"name": "商品名", "score": "推荐指数1-10"}
"""}
```
**作用**：让 AI 输出固定格式，方便代码解析。

### 技巧2：分隔符（区分指令和数据）
```python
{"role": "user", "content": '''总结三引号内的文本：
\"\"\" 这里是要处理的原文 \"\"\"'''}
```
**作用**：防止用户输入的内容被误当成指令执行（防注入）。

### 技巧3：Few-shot 示例（给 AI 看例子）
```python
{"role": "user", "content": "这个产品太好用了！"},
{"role": "assistant", "content": "正面"},
{"role": "user", "content": "浪费钱，千万别买"},
{"role": "assistant", "content": "负面"},
{"role": "user", "content": "包装一般但东西还行"}   # ← 让AI判断
```
**作用**：用 2-3 个例子教会 AI 你想要的输出模式，比写规则更管用。

---

## Day 3：Function Calling（让 LLM 调用工具）

### 核心认知
> **LLM 不动手，只动嘴。** 它告诉你"我想调哪个工具、传什么参数"，你来执行，然后把结果喂回去。

### 执行流程
```
用户输入 → LLM 判断是否需要工具
  ├─ 不需要 → 直接回答
  └─ 需要 → 返回 tool_calls（工具名+参数）
            → 你执行工具函数
            → 把结果喂回 LLM
            → LLM 生成最终回复
```

### 关键代码骨架
```python
# 1. 定义工具描述（JSON Schema，告诉LLM你能做什么）
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取指定城市的实时天气",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市名"}
                },
                "required": ["city"]
            }
        }
    }
]

# 2. 实现工具函数（你自己写的真实逻辑）
def get_weather(city: str) -> str:
    # 实际调API...
    return "北京：晴，25°C"

# 3. 工具名→函数 映射表
available_functions = {
    "get_weather": get_weather,
}

# 4. 核心对话：两轮API调用
# 第1轮：带 tools 参数，让LLM决定要不要调工具
# 第2轮：把工具结果喂回去，不带 tools，让LLM生成最终回答
```

### 踩过的坑
- 和风天气 404 是因为 API 订阅问题，不影响核心逻辑
- 模型已切换至 `deepseek-v4-flash`（更快更便宜）
- 最终天气 API 方案见 Day 4

---

## Day 4：手写原生 ReAct Agent ⭐

> 这是 W3-4 分水岭的核心，从零用 4 个函数组装出完整 Agent。

### 核心认知
> **ReAct = Reasoning（推理）+ Acting（行动）**。AI 在"思考→行动→观察→再思考"的循环中自主完成任务，比 Function Calling 更进一步：AI 自己决定何时停、何时继续。

### 四步循环
```
Thought（思考）：我该做什么？
  ↓
Action（行动）：调用工具
  ↓
Observation（观察）：拿到结果
  ↓
... 循环，直到 ...
  ↓
Final Answer（最终答案）
```

### 五个台阶

**台阶1：定义工具函数 + 映射表**
```python
def get_weather(city: str) -> str:
    # Open-Meteo API（免费，无需Key）
    # 第1步：城市名 → 经纬度（geocoding API）
    # 第2步：经纬度 → 天气（forecast API）
    ...

def calculate(expression: str) -> str:
    return str(eval(expression))

available_functions = {
    "get_weather": get_weather,
    "calculate": calculate,
}
```

**台阶2：TOOLS_DESC（工具描述文本）**
```python
TOOLS_DESC = """
- get_weather(city): 获取指定城市实时天气
- calculate(expression): 执行数学计算
"""
```
LLM 通过这段文本知道"我能用什么工具"，而不是通过 JSON Schema（与 Day 3 的区别）。

**台阶3：REACT_PROMPT + call_deepseek**
```python
REACT_PROMPT = """你是一个助手，用以下格式回答：

Thought: 你的思考过程
Action: 工具名(参数)
Observation: 工具返回结果
...（可重复）
Final Answer: 最终答案

可用工具：
{tools}

用户问题：{question}
"""
```
关键在于用 Prompt 模板把"工具列表"和"用户问题"注入进去。

**台阶4：parse_react（解析 AI 输出）**
```python
def parse_react(text: str):
    """从AI输出中提取 Thought/Action/Final Answer"""
    # 解析 Thought: 后面的内容
    # 解析 Action: 工具名(参数)
    # 解析 Final Answer: 后面的内容
    ...
```

**台阶5：主循环 react_agent**
```python
def react_agent(question: str):
    prompt = REACT_PROMPT.format(tools=TOOLS_DESC, question=question)
    messages = [{"role": "user", "content": prompt}]
    
    while True:
        response = call_deepseek(messages)
        text = response["choices"][0]["message"]["content"]
        
        parsed = parse_react(text)
        
        if parsed["action"]:      # 有Action → 执行工具 → 继续循环
            result = available_functions[parsed["tool_name"]](**parsed["args"])
            messages.append({"role": "user", "content": f"Observation: {result}"})
        elif parsed["final_answer"]:  # 有Final Answer → 结束
            return parsed["final_answer"]
```

### 天气 API 迭代史（三次选型）
| 方案 | 问题 | 结果 |
|------|------|------|
| 和风天气 | 收费，需注册Key | ❌ 放弃 |
| wttr.in | 数据不准（24°C vs 实际34°C） | ❌ 放弃 |
| **Open-Meteo** | 免费无Key，欧洲气象中心数据 | ✅ 最终方案 |

### 踩过的坑
- **缩进混用**：Tab 和空格混用 → PyCharm `Ctrl+Alt+L` 自动格式化
- **with_tools 参数**：Day 4 不需要，用 Prompt 文本驱动而非 JSON Schema
- **返回值**：call_deepseek 返回文本而非 JSON 对象

---

## 关键对比

| | Day 3 Function Calling | Day 4 ReAct Agent |
|---|---|---|
| 工具描述方式 | JSON Schema | 纯文本 Prompt |
| 调用决策 | LLM 通过 tool_calls 字段返回 | LLM 在文本中输出 Action: |
| 循环控制 | 你写死两轮 | LLM 自己决定何时停止 |
| 可扩展性 | 较复杂 | 简单灵活 |

---

## 当前技术栈
- **模型**：`deepseek-v4-flash`（最新，性价比极高）
- **API URL**：`https://api.deepseek.com/chat/completions`
- **天气 API**：Open-Meteo（免费无Key）
- **天气码映射**：`0=晴天, 1-3=多云, 45-48=雾, 51-67=雨, 71-77=雪, 95-99=雷暴`

## 下一站：W5-6 LangChain 框架实战

---

> 本内容由 Coze AI 生成，请遵循相关法律法规及《人工智能生成合成内容标识办法》使用与传播。
