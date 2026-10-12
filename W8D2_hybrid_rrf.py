"""
W8D2 · 混合检索（Hybrid Search）+ RRF 融合
=============================================
思路：向量检索（懂语义）+ BM25（认字面） 两路并跑
      再用 RRF 只融合"名次"，绕开分数不可比的问题

复用 W7D5 已经建好的库：./kb_chroma_d5（27 个 chunk）

跑法：
    直接跑        → 用你自己写的 rrf_fuse（没填 = 融合结果是空的）
    加 --ref      → 看参考实现的效果，对照检查
    KB_PATH=...   → 指定库路径（默认 ./kb_chroma_d5）
"""
import os
import sys
import jieba
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
jieba.setLogLevel(20)          # 关掉 jieba 的加载日志

USE_REF = "--ref" in sys.argv  # 加 --ref 看参考实现，直接跑就是你的实现

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

DB_PATH = os.getenv("KB_PATH", "./kb_chroma_d5")
TOP_K = 4          # D2 用 4：先看清两路各自"漏了什么"，比 8 更有对比感


def short(s, n=58):
    s = " ".join(s.split())
    return s[:n] + ("…" if len(s) > n else "")


def sec_of(d):
    md = d.metadata or {}
    return md.get("节") or md.get("小节") or md.get("篇") or "-"


# ============================================================
# 1. 加载 W7 的向量库，并取出全部 chunk
#    （BM25 必须对同一批 chunk 建索引，两路才能比较/融合）
# ============================================================
vectorstore = Chroma(
    collection_name="langchain",
    embedding_function=embeddings,
    persist_directory=DB_PATH,
)
data = vectorstore.get(include=["documents", "metadatas"])
all_ids = data["ids"]
all_texts = data["documents"]

doc_by_id = {
    i: Document(page_content=t, metadata=m or {})
    for i, t, m in zip(all_ids, all_texts, data["metadatas"])
}
print(f"从库里读到 {len(all_ids)} 个 chunk")

k = TOP_K
retriever = vectorstore.as_retriever(search_kwargs={"k": k})

# ============================================================
# 2. 构建 BM25 索引
# ============================================================
tokenized = [list(jieba.cut(t)) for t in all_texts]
bm25 = BM25Okapi(tokenized)
print("BM25 索引构建完成")


# ============================================================
# 3. 两路独立检索：各自返回按名次排好的 [(id, doc), ...]
# ============================================================
def vector_search(query, k=TOP_K):
    """向量路：Chroma 按余弦距离返回，已是从近到远"""
    docs = retriever.invoke(query)
    return [(d.id, d) for d in docs]     # d.id 就是库里那条 chunk 的 id


def bm25_search(query, k=TOP_K):
    """BM25 路：打分后自己排序"""
    scores = bm25.get_scores(list(jieba.cut(query)))
    ranked = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
    return [(all_ids[i], doc_by_id[all_ids[i]]) for i in ranked]


# ============================================================
# 4. ★ 今日主角：RRF 融合
# ============================================================
def rrf_fuse(result_lists, k=60, top_n=TOP_K):
    """
    result_lists: [list1, list2]，每个元素是 [(id, doc), ...]，按名次从高到低
    k:            平滑常数，论文默认 60
    返回:         [(id, doc, rrf_score), ...] 按融合分从高到低
    """
    scores, id2doc = {}, {}

    if USE_REF:
        # ===== 参考实现（--ref 时生效）=====
        for rl in result_lists:
            for rank, (doc_id, doc) in enumerate(rl, 1):
                id2doc[doc_id] = doc
                scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank)
        ordered = sorted(scores.items(), key=lambda kv: -kv[1])[:top_n]
        return [(i, id2doc[i], s) for i, s in ordered]

    # ===== TODO(你) =====
    # 遍历 result_lists 里的每一路；对每一路里第 rank 名（从 1 开始）的 (doc_id, doc)：
    #   1) id2doc[doc_id] = doc                      （记住 id 对应的原始文档）
    #   2) scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank)
    #
    # 最后按 scores 从高到低排序，返回 [(id, id2doc[id], score), ...][:top_n]
    for rl in result_lists:  # 遍历每一路
        for rank, (doc_id, doc) in enumerate(rl, 1):  # rank 从 1 开始！
            id2doc[doc_id] = doc
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank)
    ordered = sorted(scores.items(), key=lambda kv: -kv[1])[:top_n]
    return [(i, id2doc[i], s) for i, s in ordered]



# ============================================================
# 5. 三路对比：同一批问题，看各自捞到了什么
# ============================================================
QUERIES = [
    "WC所有语法？",                    # 生造缩写：向量大概率漂走，BM25 抓字面
    "我学过哪几种输出解析器？",         # 语义题：向量该赢
    "Function Calling 和 ReAct 有什么区别？",
]


def show(title, items):
    print(f"  【{title}】")
    for i, it in enumerate(items, 1):
        if len(it) == 3:
            doc_id, d, s = it
            print(f"    {i}. rrf={s:.5f}  [{sec_of(d)}] {short(d.page_content)}")
        else:
            doc_id, d = it
            print(f"    {i}. [{sec_of(d)}] {short(d.page_content)}")


def compare(query):
    print("\n" + "=" * 70)
    print("Q:", query)

    vec = vector_search(query)
    bm = bm25_search(query)
    fused = rrf_fuse([vec, bm], k=60, top_n=TOP_K)

    show("向量检索(语义)", vec)
    show("BM25(字面)", bm)
    show("RRF 融合", fused)
    return fused


if __name__ == "__main__":
    results = {q: compare(q) for q in QUERIES}

    # ========================================================
    # 6. 把 RRF 的 top4 拼进 prompt，看答案质量
    # ========================================================
    prompt = ChatPromptTemplate.from_template("""你是我（AI Agent学习者）的学习笔记问答助手。只能依据下面<笔记>里的内容回答。
要求：
1. 笔记里有的，简洁准确回答，尽量用笔记中原话；
2. 笔记里没有的，直接说"我的笔记里没有相关内容"，严禁编造；
3. 不要提"笔记""检索"这类词，自然作答。

<笔记>
{context}
</笔记>

问题：{question}""")

    def format_fused(items):
        return "\n\n".join(f"[{sec_of(d)}] {d.page_content}" for _, d, _ in items)

    chain = prompt | llm | StrOutputParser()

    print("\n\n" + "#" * 70)
    print("# 融合结果直接问答")
    print("#" * 70)
    for q, fused in results.items():
        print("\nQ:", q)
        if not fused:
            print("A: （rrf_fuse 还没填，融合结果为空）")
            continue
        print("A:", chain.invoke({"context": format_fused(fused), "question": q}))
