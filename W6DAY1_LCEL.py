# /*LCEL 是 **LangChain 表达式语言**的缩写，英文全称是 **LangChain Expression Language**。
#
# 你可以把它理解为**一种用来“组装” AI 工作流的简便写法**。就像搭积木一样，你用一些现成的组件（比如“调用大模型”、“把文本格式化”、“查数据”），通过 `|` 这个管道符号把它们连接起来，形成一个完整的流程。
#
# ### 举个例子
#
# 想象你有一个简单的任务：**先让 AI 讲个笑话，然后把笑话翻译成中文**。
#
# 用 LCEL 写出来大概是这样：
#
# ```python
# prompt = ChatPromptTemplate.from_template("讲一个关于{topic}的笑话")
# model = ChatOpenAI()
# output_parser = StrOutputParser()
#
# chain = prompt | model | output_parser
# ```
#
# 这条链（chain）的执行顺序就是：
#
# 1. `prompt`：把用户输入的话题放进模板，生成一个完整的指令。
# 2. `model`：把这个指令交给大模型处理。
# 3. `output_parser`：把模型返回的内容整理成纯文本。
#
# 最后你只要调用这个 `chain`，它就会自动按顺序执行。
#
# ### LCEL 解决了什么问题？
#
# - **直观**：用 `|` 连接，一眼就能看出数据流向。
# - **方便组合**：可以把小的模块拼成大的流程，比如再加一步“翻译成英文”，就在后面再 `|` 一下。
# - **内置了很多实用功能**：比如自动重试、流式输出、并行执行、记录日志，你不需要自己写很多额外代码。
#
# 简单说，**LCEL 就是 LangChain 提供的一种“管道式”编程方式**，让你用最少的力气把大模型和工具串联起来完成任务。
#
# 进程已结束，退出代码为 0
# */

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI              # ChatModels：管"调模型API"
from langchain_core.prompts import ChatPromptTemplate  # Prompts：管"提示词模板"
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage, AIMessage

# ① 创建模型对象 —— 替代你手写的一大堆 requests 代码
load_dotenv()   # 把 .env 里的内容读进环境变量
llm = ChatOpenAI(
    model="deepseek-v4-flash",          # 模型名（你 Day4 已经在用的那个）
    api_key=os.getenv("DEEPSEEK_API_KEY"),           # ★ 整个代码只有这里要改
    base_url="https://api.deepseek.com",
)
prompt = ChatPromptTemplate.from_template(
    "你是{role}，请用{style}的方式回答：\n{question}"
)
parser = StrOutputParser()


@tool
def add_emoji(text: str) -> str:
    """在句末加入✨表情"""
    return text + " ✨"


# ========= 关键就这一行 =========
chain = prompt | llm | parser|add_emoji | (lambda s: s.replace("你", "您"))

# ========= 用也只用一行 =========
result = chain.invoke({
    "role": "耐心的编程老师",
    "style": "简单直白",
    "question": " LCEL的流动顺序？"
})
print(result)

# ========= 附：手动维护 messages 历史（和 LCEL 对照看）=========
# LCEL 里一行 chain 就搞定；不用 LCEL 就得自己管 messages 列表
messages = [HumanMessage(content="LCEL 的流动顺序是什么？")]
reply = llm.invoke(messages)                        # 返回 AIMessage 对象，不是字符串
messages.append(AIMessage(content=reply.content))   # ★ 必须把 AI 回复也塞回历史
messages.append(HumanMessage(content="那我为什么还要用它？"))
print(llm.invoke(messages).content)