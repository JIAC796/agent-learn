import os, shutil
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

embeddings = OpenAIEmbeddings(
    model="BAAI/bge-m3",
    api_key=os.getenv("SILICONFLOW_API_KEY"),
    base_url="https://api.siliconflow.cn/v1",
)
llm = ChatOpenAI(
    model="deepseek-v4-flash",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)

SRC_FILE = "复习笔记_AI_Agent_Day1-4.md"
DB_PATH = "./kb_chroma_d5"

# ===== 1. 按 Markdown 标题切，保住章节结构 =====
headers = [("#", "篇"), ("##", "节"), ("###", "小节")]
md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers)
big_chunks = md_splitter.split_text(open(SRC_FILE, encoding="utf-8").read())

# ===== 2. 再按长度二次切，避免单块超长 =====
recursive = RecursiveCharacterTextSplitter(
    chunk_size=300, chunk_overlap=40,
    separators=["\n", "。", "；", "，", ""],
)
chunks = recursive.split_documents(big_chunks)

# ===== 3. 滤掉太短 + 含水印的噪声块 =====
chunks = [c for c in chunks if len(c.page_content.strip()) >= 20]
junk = ["Coze AI 生成", "AIGC", "人工智能生成", "ContentProducer"]
chunks = [c for c in chunks if not any(k in c.page_content for k in junk)]
print(f"切成 {len(chunks)} 个有效 chunk")

# ===== 4. 建库 + 检索器（cosine，K=8）=====
if os.path.exists(DB_PATH):
    shutil.rmtree(DB_PATH)
vectorstore = Chroma.from_documents(
    documents=chunks, embedding=embeddings,
    persist_directory=DB_PATH,
    collection_metadata={"hnsw:space": "cosine"},
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 8})
print("入库条数:", vectorstore._collection.count())

# ===== 5. format_docs：把检索结果拼成带小节标题的文本 =====
def format_docs(ds):
    parts = []
    for d in ds:
        sec = d.metadata.get("节", d.metadata.get("小节", ""))
        parts.append(f"[{sec}] {d.page_content}")
    return "\n\n".join(parts)

# ===== 6. 问答链 =====
prompt = ChatPromptTemplate.from_template("""你是我（AI Agent学习者）的学习笔记问答助手。只能依据下面<笔记>里的内容回答。
要求：
1. 笔记里有的，简洁准确回答，尽量用笔记中原话；
2. 笔记里没有的，直接说"我的笔记里没有相关内容"，严禁编造；
3. 不要提"笔记""检索"这类词，自然作答。

<笔记>
{context}
</笔记>

问题：{question}""")

chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt | llm | StrOutputParser()
)

def ask(q):
    print("\n" + "=" * 50)
    print("Q:", q)
    print("A:", chain.invoke(q))

ask("我学过哪几种输出解析器？")
ask("Function Calling 和 ReAct 有什么区别？")
ask("WC所有语法？")