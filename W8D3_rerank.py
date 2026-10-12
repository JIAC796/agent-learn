"""
W8D3 · Rerank 重排 + 阈值拒答
=============================================
承接 W8D2：向量 + BM25 双路召回 → RRF 融合 → ★rerank 精排 → 阈值筛 → 生成

为什么需要 rerank：
  Bi-Encoder（bge-m3）把 query 和 doc 分开编码，看不到两者交互，快但粗；
  Cross-Encoder（bge-reranker）把 query 和 doc 拼在一起送进模型，准但慢。
  所以工业界两段式：先粗筛（召回 15 条），再精判（重排到 3 条）。

阈值怎么来的（★重点：不是拍脑袋，是标定出来的）：
  W8D3_calibrate.py 用 16 条标注测试集（10 该答 / 6 该拒）扫描阈值，结果：
     阈值 0.05  → 准确率 81.2%（13/16）  拍脑袋的基线
    阈值 0.138 → 准确率 87.5%（14/16）
    阈值 0.326 → 准确率 93.8%（15/16）  ← 采用这个
    阈值 ≥0.74  → 准确率跌回 81.2%     ← 开始假拒答（把该答的也拒了）
  结论：阈值不是越高越保守，0.326 附近才是最优；93.8% 已是这套语料的天花板。
  标定前还有两处语料问题（缺 LCEL 定义、自指污染），修完才收敛到这个值。

  ⚠️ 重要认知：reranker 输出的是【排序分数】不是【相关性概率】。
  它只保证 A 比 B 更相关，不保证"0.326 以上就算相关"。
  所以阈值只能当"省钱过滤器"，最终判断还是要靠 prompt 里的拒答约束。

跑法：
    直接跑        → 用你自己写的 rerank
    加 --ref      → 看参考实现
    KB_PATH=...   → 指定库路径（默认 ./kb_chroma_d5）
"""
import os
import sys
import jieba
import requests
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
jieba.setLogLevel(20)

USE_REF = "--ref" in sys.argv

DB_PATH = os.getenv("KB_PATH", "./kb_chroma_d5")
RECALL_K = 15      # 每路召回的候选数（放大！给 rerank 更多原料）
RERANK_TOP_N = 3   # rerank 之后真正喂给 LLM 的条数

# ★ 拒答阈值：由 W8D3_calibrate.py 在 16 条标注测试集上标定得出
#   0.05 → 81.2%（13/16）｜0.138 → 87.5%（14/16）｜0.326 → 93.8%（15/16，采用）
#   再往上（≥0.74）会跌回 81.2% —— 开始把该答的也拒了（假拒答），
#   所以不是越高越保守，0.326 附近才是最优；93.8% 是这套语料的天花板。
SCORE_THRESHOLD = 0.326

SILICONFLOW_KEY = os.getenv("SILICONFLOW_API_KEY")
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"

embeddings = OpenAIEmbeddings(
    model="BAAI/bge-m3",
    api_key=SILICONFLOW_KEY,
    base_url="https://api.siliconflow.cn/v1",
)
llm = ChatOpenAI(
    model="deepseek-v4-flash",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)


def short(s, n=56):
    s = " ".join(s.split())
    return s[:n] + ("…" if len(s) > n else "")


def sec_of(d):
    md = d.metadata or {}
    return md.get("节") or md.get("小节") or md.get("篇") or "-"


# ============================================================
# 1. 加载库 + 建 BM25（同 W8D2）
# ============================================================
vectorstore = Chroma(
    collection_name="langchain",
    embedding_function=embeddings,
    persist_directory=DB_PATH,
)
data = vectorstore.get(include=["documents", "metadatas"])
all_ids, all_texts = data["ids"], data["documents"]

# ===== 空库保护：路径不对时给明确提示，而不是后面 BM25 除零崩溃 =====
if len(all_ids) == 0:
    raise SystemExit(
        f"\n❌ 库 {os.path.abspath(DB_PATH)} 是空的（0 个 chunk）。\n"
        f"   当前运行目录：{os.getcwd()}\n"
        f"   DB_PATH 是相对路径，从【运行目录】开始找；找不到时 Chroma 会默默新建空库，\n"
        f"   然后 BM25 初始化时除以 0 报 ZeroDivisionError。\n"
        f"   解法：设 KB_PATH 环境变量指向真实库，或先跑 W8_build_kb_v2.py 在本目录建库。\n"
    )

doc_by_id = {
    i: Document(page_content=t, metadata=m or {})
    for i, t, m in zip(all_ids, all_texts, data["metadatas"])
}
print(f"库里有 {len(all_ids)} 个 chunk")

retriever = vectorstore.as_retriever(search_kwargs={"k": RECALL_K})
bm25 = BM25Okapi([list(jieba.cut(t)) for t in all_texts])


# ============================================================
# 2. 双路召回 + RRF（W8D2 的代码，直接用）
# ============================================================
def vector_search(query, k=RECALL_K):
    return [(d.id, d) for d in retriever.invoke(query)]


def bm25_search(query, k=RECALL_K):
    scores = bm25.get_scores(list(jieba.cut(query)))
    ranked = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
    return [(all_ids[i], doc_by_id[all_ids[i]]) for i in ranked]


def rrf_fuse(result_lists, k=60, top_n=RECALL_K):
    scores, id2doc = {}, {}
    for rl in result_lists:
        for rank, (doc_id, doc) in enumerate(rl, 1):
            id2doc[doc_id] = doc
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank)
    ordered = sorted(scores.items(), key=lambda kv: -kv[1])[:top_n]
    return [(i, id2doc[i], s) for i, s in ordered]


# ============================================================
# 3. ★ 今日主角：rerank 精排
#    待填
# ============================================================
def rerank(query, candidates, top_n=RERANK_TOP_N):
    """
    candidates: [(id, Document, rrf_score), ...]  来自 rrf_fuse
    返回:       [(Document, relevance_score), ...] 按分数从高到低

    步骤：
      1) 把 candidates 里的文档正文抽成 texts 列表（注意顺序！）
      2) POST https://api.siliconflow.cn/v1/rerank
         headers: Authorization: Bearer <key>
         body:    {"model": RERANK_MODEL, "query": query,
                   "documents": texts, "top_n": top_n, "return_documents": False}
      3) 响应 data["results"] 是 [{"index": i, "relevance_score": s}, ...]，已按分降序
         ★ index 是 texts 里的下标 → 用它去索引 candidates 拿回原 Document
      4) 返回 [(doc, score), ...]
    """
    if USE_REF:
        # ===== 参考实现 =====
        texts = [d.page_content for _, d, _ in candidates]
        resp = requests.post(
            "https://api.siliconflow.cn/v1/rerank",
            headers={"Authorization": f"Bearer {SILICONFLOW_KEY}",
                     "Content-Type": "application/json"},
            json={"model": RERANK_MODEL, "query": query, "documents": texts,
                  "top_n": top_n, "return_documents": False},
            timeout=60,
        )
        resp.raise_for_status()
        out = []
        for it in resp.json()["results"]:          # 已按分数降序
            doc = candidates[it["index"]][1]        # ★ index 对回原候选
            out.append((doc, it["relevance_score"]))
        return out

    # ===== 正式实现 =====
    # 1) 抽出文档正文（顺序和 candidates 一致，index 才对得上）
    texts = [d.page_content for _, d, _ in candidates]

    # 2) 调 rerank 接口
    resp = requests.post(
        "https://api.siliconflow.cn/v1/rerank",
        headers={"Authorization": f"Bearer {SILICONFLOW_KEY}",
                 "Content-Type": "application/json"},
        json={"model": RERANK_MODEL, "query": query, "documents": texts,
              "top_n": top_n, "return_documents": False},
        timeout=60,
    )
    resp.raise_for_status()

    # 3) results 已按 relevance_score 降序，index 是 texts 里的下标
    out = []
    for it in resp.json()["results"]:
        doc = candidates[it["index"]][1]      # ★ index 对回原候选
        out.append((doc, it["relevance_score"]))
    return out


# ============================================================
# 4. 完整链路：召回 → 融合 → 重排 → 拒答 → 生成
# ============================================================
def retrieve_and_rerank(query):
    """返回 (通过的文档列表, 全部重排结果)"""
    vec = vector_search(query)
    bm = bm25_search(query)
    fused = rrf_fuse([vec, bm], top_n=RECALL_K)
    reranked = rerank(query, fused, top_n=RERANK_TOP_N)
    kept = [(d, s) for d, s in reranked if s >= SCORE_THRESHOLD]
    return kept, reranked


def show(query, kept, reranked):
    print("\n" + "=" * 72)
    print("Q:", query)
    print(f"  【rerank 重排结果】阈值={SCORE_THRESHOLD}")
    for i, (d, s) in enumerate(reranked, 1):
        flag = "✅" if s >= SCORE_THRESHOLD else "❌拒"
        print(f"    {i}. {flag} score={s:.6f}  [{sec_of(d)}] {short(d.page_content)}")
    print(f"  → 通过阈值 {len(kept)} 条 / 候选 {len(reranked)} 条")


PROMPT = ChatPromptTemplate.from_template("""你是我（AI Agent学习者）的学习笔记问答助手。只能依据下面<笔记>里的内容回答。
要求：
1. 笔记里有的，简洁准确回答，尽量用笔记中原话；
2. 笔记里没有的，直接说"我的笔记里没有相关内容"，严禁编造；
3. 不要提"笔记""检索"这类词，自然作答。

<笔记>
{context}
</笔记>

问题：{question}""")
chain = PROMPT | llm | StrOutputParser()

QUERIES = [
    "Function Calling 和 ReAct 有什么区别？",   # 该答得出来
    "BM25 是什么？",                            # 该答（新语料补的）
    "WC所有语法？",                             # ★ 该被拦下、直接拒答
    "我学过哪几种输出解析器？",                  # 该答（新语料补的）
]

if __name__ == "__main__":
    print(f"模式：{'参考实现 --ref' if USE_REF else '你的实现'}")
    for q in QUERIES:
        kept, reranked = retrieve_and_rerank(q)
        show(q, kept, reranked)

        if not kept:
            print("  → 【阈值拒答】没有文档达标，直接返回，不调用 LLM 💰省钱")
            continue

        context = "\n\n".join(f"[{sec_of(d)}] {d.page_content}" for d, _ in kept)
        print("  → 【LLM 回答】")
        print("   ", chain.invoke({"context": context, "question": q}).replace("\n", "\n    "))
