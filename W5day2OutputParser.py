import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI              # ChatModels：管"调模型API"
from langchain_core.prompts import ChatPromptTemplate  # Prompts：管"提示词模板"
from langchain_core.output_parsers import StrOutputParser   # ← 新引入
# ① 创建模型对象 —— 替代你手写的一大堆 requests 代码
load_dotenv()   # 把 .env 里的内容读进环境变量
llm = ChatOpenAI(
    model="deepseek-v4-flash",          # 模型名（你 Day4 已经在用的那个）
    api_key=os.getenv("DEEPSEEK_API_KEY"),           # ★ 整个代码只有这里要改
    base_url="https://api.deepseek.com",
)
# ② 用模板管理 prompt —— 替代你手写的 f"你是{role}..." 字符串拼接
prompt = ChatPromptTemplate.from_template(
    "你是一个{role}。\n"
    "请用{style}的方式回答：\n"
    "{question}"
)
chain = prompt | llm | StrOutputParser()
result = chain.invoke({
    "role": "温柔且靠谱的猫娘编程导师",
    "style": "通俗易懂",
    "question": "OutputParser 是干嘛的？",
})


chain1 = prompt | llm
result1 = chain1.invoke({
    "role": "毒舌但靠谱的猫娘编程导师",
    "style": "通俗易懂",
    "question": "RAG 是干嘛的？"})

chain2 = prompt | llm | StrOutputParser()
result2 = chain2.invoke({
    "role": "毒舌但靠谱的猫娘编程导师",
    "style": "通俗易懂",
    "question": "RAG 是干嘛的？"})

print(type(result1))   # <class 'langchain_core.messages.ai.AIMessage'>   ← 是个对象
print(type(result2))   # <class 'str'>                                    ← 是字符串