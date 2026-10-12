"""
W6 Day2: 流式输出 + RunnableLambda
- stream() 打字机效果
- RunnableLambda 自定义节点进LCEL链
"""

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
import os

load_dotenv()

llm = ChatOpenAI(
    model="deepseek-v4-flash",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

# ============================================
# Part 1: invoke vs stream 对比
# ============================================

prompt = ChatPromptTemplate.from_template("用一句话解释：{concept}")
chain = prompt | llm | StrOutputParser()

print("=" * 50)
print("【invoke】等全想完再返回")
print("=" * 50)
result = chain.invoke({"concept": "什么是流式输出"})
print(result)

print("\n" + "=" * 50)
print("【stream】边想边吐，打字机效果")
print("=" * 50)
for chunk in chain.stream({"concept": "什么是流式输出"}):
    print(chunk, end="", flush=True)
print()  # 换行

# ============================================
# Part 2: RunnableLambda 自定义节点
# ============================================

print("\n" + "=" * 50)
print("RunnableLambda: 自定义函数进管道")
print("=" * 50)

# 场景：模型输出 → 加emoji → 加签名 → 最终输出

def add_emoji(text: str) -> str:
    """给输出加个emoji前缀"""
    return f"🤖 {text}"

def add_signature(text: str) -> str:
    """给输出加个签名后缀"""
    return f"{text}\n\n——来自你的AI导师 ✨"

# 用 RunnableLambda 包一下，就能用 | 串联
emoji_node = RunnableLambda(add_emoji)
signature_node = RunnableLambda(add_signature)

# 组装链：prompt → llm → parser → 加emoji → 加签名
fancy_chain = prompt | llm | StrOutputParser() | emoji_node | signature_node

result = fancy_chain.invoke({"concept": "什么是RunnableLambda"})
print(result)

# ============================================
# Part 3: 流式 + Lambda 组合
# ============================================

print("\n" + "=" * 50)
print("流式 + Lambda：边吐边处理")
print("=" * 50)

# 注意：stream遇到Lambda，Lambda会在每个chunk上执行
# 所以加emoji这种"只加一次"的操作，流式下会每个chunk都加
# 解决办法：最后统一处理，用 .stream() 只在 llm | parser 那段

# 方案：先流式拿到完整输出，再过Lambda
# （实际生产中，前端渲染流式chunk，后处理在前端做）
for chunk in (prompt | llm | StrOutputParser()).stream({"concept": "LangChain的LCEL"}):
    print(chunk, end="", flush=True)
print()
# 流式完了再过Lambda后处理也可以，但那就是两步了
# 关键认知：stream和Lambda组合时，注意Lambda是逐chunk执行的

# ============================================
# Part 4: RunnableLambda 的高级用法 - 输入变换
# ============================================

print("\n" + "=" * 50)
print("RunnableLambda: 输入变换（链中间改数据格式）")
print("=" * 50)

# 场景：用户输入一个主题词 → Lambda把它扩展成完整问题 → 再过prompt
def expand_topic(topic: str) -> dict:
    """把一个词扩展成结构化输入"""
    return {"concept": f"请详细解释{topic}，并举一个生活中的例子"}

expand_node = RunnableLambda(expand_topic)

# 链：用户输入主题词 → 扩展 → prompt → llm → parser
smart_chain = expand_node | prompt | llm | StrOutputParser()

result = smart_chain.invoke("LCEL链式编程")
print(result)

print("\n✅ Day2 完成！你学会了：stream()流式输出 + RunnableLambda自定义节点")
