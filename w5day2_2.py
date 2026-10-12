import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI  # ChatModels：管"调模型API"
from langchain_core.prompts import ChatPromptTemplate  # Prompts：管"提示词模板"
from langchain_core.output_parsers import JsonOutputParser  # ← 新引入

# ① 创建模型对象 —— 替代你手写的一大堆 requests 代码
load_dotenv()  # 把 .env 里的内容读进环境变量
llm = ChatOpenAI(
    model="deepseek-v4-flash",  # 模型名（你 Day4 已经在用的那个）
    api_key=os.getenv("DEEPSEEK_API_KEY"),  # ★ 整个代码只有这里要改
    base_url="https://api.deepseek.com",
)
prompt = ChatPromptTemplate.from_template(
    "你是数据整理助手。\n"
    "根据问题回答，严格返回JSON，包含两个字段：answer、suggestions。\n"
    "suggestions 是一个字符串列表。\n"
    "问题：{question}"
)

chain = prompt | llm | JsonOutputParser()

data = chain.invoke({"question": "怎么学好 LangChain？"})

print(data["answer"])          # 模型给你的正文
print(data["suggestions"])     # 一个列表
print(type(data))              # <class 'dict'> ← 重点