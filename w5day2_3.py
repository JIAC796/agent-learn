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
    "你是学习规划助手。\n"
    "为今天安排一份AI学习计划，返回JSON，格式为：\n"
    '{{"plan": ["上午内容", "下午内容", "晚上内容"]}}\n'
    "计划要具体可执行，3条。"
)

chain = prompt | llm | JsonOutputParser()

data = chain.invoke({})        # 模板里没有占位符，就传空字典

# 用 for 循环把计划一条条打印出来
for item in data["plan"]:
    print("-", item)