"""
W8 补充实验 · 知识库扩容（增量入库）
=============================================
诊断发现：原库 27 个 chunk **100% 来自 复习笔记_AI_Agent_Day1-4.md**（只覆盖 W1-W4）。
所以问「我学过哪几种输出解析器？（W5 学的）」「BM25 是什么？（W8 学的）」
时，reranker 判低分 + LLM 拒答 —— **它是对的，库里真的没有。**

本脚本做两件事：
  1) 从**原始库**读出已有的 27 个 chunk（原始库只读，不重新 embedding，省调用）
  2) 把新手册 AI_Agent速查手册_W1-W8.md 切分后**追加**进去 → 输出库 kb_chroma_v2

设计约定（重要，避免重复入库）：
  · BASE_DB = 原始只读基底（只有 W1-W4 笔记，绝不修改它）
  · NEW_DB  = 每次从基底**重建**的输出库
  所以本脚本可以反复跑：不会重复追加，改手册后重跑即可刷新。

跑法：
    python W8_build_kb_v2.py          # 建库
    python W8_build_kb_v2.py --query  # 建库并跑对比
"""
import os
import sys
import shutil
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
)
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

OLD_DB = os.getenv("OLD_DB", "./kb_chroma_d5")     # 原始只读基底（别指向输出库！）
NEW_DB = os.getenv("NEW_DB", "./kb_chroma_v2")     # 输出库，每次从基底重建
HANDBOOK = os.getenv("HANDBOOK", "./AI_Agent速查手册_W1-W8.md")

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

# 和 W7D5 完全一致的切分参数（保证新旧语料粒度统一）
HEADERS = [("#", "篇"), ("##", "节"), ("###", "小节")]
recursive = RecursiveCharacterTextSplitter(
    chunk_size=300, chunk_overlap=40,
    separators=["\n", "。", "；", "，", ""],
)
JUNK = ["Coze AI 生成", "AIGC", "人工智能生成", "ContentProducer"]


def chunk_markdown(path):
    """按 W7D5 同样的策略切分 + 清洗"""
    md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=HEADERS)
    big = md_splitter.split_text(open(path, encoding="utf-8").read())
    chunks = recursive.split_documents(big)
    chunks = [c for c in chunks if len(c.page_content.strip()) >= 20]
    chunks = [c for c in chunks if not any(k in c.page_content for k in JUNK)]
    return chunks


def build():
    # ===== 1. 读出旧库已有的 chunk（增量，不是重建）=====
    old = Chroma(collection_name="langchain", embedding_function=embeddings,
                 persist_directory=OLD_DB)
    data = old.get(include=["documents", "metadatas"])
    carried = [Document(page_content=t, metadata=m or {})
               for t, m in zip(data["documents"], data["metadatas"])]
    src = {m.get("篇") for m in data["metadatas"] if m}
    print(f"[读出旧库] {len(carried)} 个 chunk，来源：{src}")

    # ===== 2. 切分新手册 =====
    new_chunks = chunk_markdown(HANDBOOK)
    print(f"[切分新手册] {HANDBOOK} → {len(new_chunks)} 个 chunk")
    for c in new_chunks[:3]:
        print(f"    例: [{c.metadata.get('节','-')}] {' '.join(c.page_content.split())[:50]}")

    # ===== 3. 建新库（全新目录，不动旧库）=====
    if os.path.exists(NEW_DB):
        shutil.rmtree(NEW_DB)
    total = carried + new_chunks
    vs = Chroma.from_documents(
        documents=total, embedding=embeddings,
        collection_name="langchain",
        persist_directory=NEW_DB,
        collection_metadata={"hnsw:space": "cosine"},
    )
    print(f"[建库完成] {NEW_DB} 共 {vs._collection.count()} 个 chunk "
          f"(旧 {len(carried)} + 新 {len(new_chunks)})")
    return vs


PROMPT = ChatPromptTemplate.from_template("""你是我（AI Agent学习者）的学习笔记问答助手。只能依据下面<笔记>里的内容回答。
要求：
1. 笔记里有的，简洁准确回答，尽量用笔记中原话；
2. 笔记里没有的，直接说"我的笔记里没有相关内容"，严禁编造；
3. 不要提"笔记""检索"这类词，自然作答。

<笔记>
{context}
</笔记>

问题：{question}""")

QUERIES = [
    ("我学过哪几种输出解析器？", "W5 学的 → 旧库缺失"),
    ("BM25 是什么？", "W8 学的 → 旧库缺失"),
    ("Function Calling 和 ReAct 有什么区别？", "W1-4 → 旧库本来就答得出"),
    ("WC所有语法？", "笔记里真的没有 → 应该继续拒答"),
]


def ask(vs, title):
    r = vs.as_retriever(search_kwargs={"k": 4})
    chain = PROMPT | llm | StrOutputParser()
    print(f"\n{'#'*70}\n# {title}\n{'#'*70}")
    for q, note in QUERIES:
        docs = r.invoke(q)
        ctx = "\n\n".join(f"[{d.metadata.get('节','-')}] {d.page_content}" for d in docs)
        print(f"\nQ: {q}   （{note}）")
        print("A:", chain.invoke({"context": ctx, "question": q}).replace("\n", "\n   "))


if __name__ == "__main__":
    vs_new = build()
    if "--query" in sys.argv:
        old_vs = Chroma(collection_name="langchain", embedding_function=embeddings,
                        persist_directory=OLD_DB)
        ask(old_vs, "旧库（27 chunk，只有 W1-W4）")
        ask(vs_new, "新库（补上 W1-W8 手册）")
