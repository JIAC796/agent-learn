from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
import os
import time
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

llm = ChatOpenAI(
    model="deepseek-v4-flash",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)
class 报错诊断(BaseModel):
    错误类型: str = Field(description="报错类型，如ModuleNotFoundError")
    可能原因: str = Field(description="一句话解释为什么会报错")
    解决步骤: list[str] = Field(description="解决步骤列表")

prompt = ChatPromptTemplate.from_template(
    "我运行python文件报错：{error_message}，请帮我诊断，并以json格式输出。"
    "必须包含以下三个字段：错误类型、可能原因、解决步骤（解决步骤是字符串数组）"
)


chain = prompt | llm.with_structured_output(报错诊断, method="json_mode")

result = chain.invoke({ {'error': {'message': "Prompt must contain the word 'json' in some form to use 'response_format' of type 'json_object'.")

print("错误类型:", result.错误类型)
print("可能原因:", result.可能原因)
print("解决步骤:", result.解决步骤)
