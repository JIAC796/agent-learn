"""
W8D4 · RAG 评估框架
=============================================
为什么需要评估：
  前面 D1-D3 全是"造"（召回、融合、重排、拒答）。没有评估，你无法回答：
    · 我的检索到底好到什么程度？
    · 改了 chunk 大小/阈值/k，是变好还是变坏？
    · 上线后质量有没有退化？

两个层次的指标：
  【检索层】只看"有没有把正确内容捞出来"，不调 LLM —— 快、便宜、可回归
      Hit Rate@k : top-k 里是否至少有一条正确 → 衡量"能不能捞到"
      Recall@k   : top-k 覆盖了多少比例的正确内容 → 衡量"捞得全不全"
      MRR        : 第一条正确内容的排名倒数均值 → 衡量"排得靠不靠前"
      nDCG@k     : 考虑排序位置和多个正确项的加权指标 → 最接近用户体感
  【生成层】看最终答案，要调 LLM —— 慢、贵、有噪声
      关键词覆盖率 : 答案是否包含应有的关键术语（确定性的粗糙代理）
      拒答正确率  : 该拒答的有没有拒、不该拒的有没有乱拒

评估集怎么来：
  本文件用"关键词标注"自动生成 ground truth（query → 含某关键词的 chunk id）。
  生产环境应该是人工标注 query→相关 chunk，这里用自动标注降低门槛。

跑法：
    python W8D4_eval.py           # 跑完整评估
    python W8D4_eval.py --dump    # 只看题库和标注结果
"""
import os
import sys
import math
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

KB_PATH = os.getenv("KB_PATH", "./kb_chroma_v2")
RECALL_K = 15
RERANK_TOP_N = 3
THRESHOLD = 0.326

SILICONFLOW_KEY = os.getenv("SILICONFLOW_API_KEY")

embeddings = OpenAIEmbeddings(model="BAAI/bge-m3", api_key=SILICONFLOW_KEY,
                              base_url="https://api.siliconflow.cn/v1")
llm = ChatOpenAI(model="deepseek-v4-flash", api_key=os.getenv("DEEPSEEK_API_KEY"),
                 base_url="https://api.deepseek.com")

# ============================================================
# 1. 载入库
# ============================================================
vectorstore = Chroma(collection_name="langchain", embedding_function=embeddings,
                     persist_directory=KB_PATH)
raw = vectorstore.get(include=["documents", "metadatas"])
ids, texts, metas = raw["ids"], raw["documents"], raw["metadatas"]
if not ids:
    raise SystemExit(f"❌ 库 {os.path.abspath(KB_PATH)} 是空的，先跑 W8_build_kb_v2.py")
doc_by_id = {i: Document(page_content=t, metadata=m or {})
             for i, t, m in zip(ids, texts, metas)}
bm25 = BM25Okapi([list(jieba.cut(t)) for t in texts])
retriever = vectorstore.as_retriever(search_kwargs={"k": RECALL_K})


# ============================================================
# 2. 评估集：query → 关键词（用来自动标注 ground truth）
#    expect="答" 表示语料里确实有答案；expect="拒" 表示应该拒答
# ============================================================
CASES = [
    ("BM25 是什么？",                       "答", ["BM25", "词频统计"]),
    ("RRF 为什么用名次不用分数？",            "答", ["尺度", "名次", "加权"]),
    ("rerank 和向量检索有什么区别？",         "答", ["Bi-Encoder", "Cross-Encoder", "重排"]),
    ("Bi-Encoder 和 Cross-Encoder 差在哪？", "答", ["Bi-Encoder", "Cross-Encoder"]),
    ("tool_call_id 是干什么的？",            "答", ["tool_call_id"]),
    ("json_mode 要注意什么？",               "答", ["json_mode"]),
    ("chunk_size 和 chunk_overlap 设多少？", "答", ["chunk_size=300", "chunk_overlap=40"]),
    ("LCEL 是什么？",                       "答", ["LangChain Expression Language", "LCEL 是什么"]),
    ("ReAct 循环怎么防止死循环？",            "答", ["range(10)", "熔断"]),
    ("Function Calling 和 ReAct 的区别？",   "答", ["ReAct = Reasoning", "关键对比"]),
    ("RAG 怎么防止模型编造？",                "答", ["严禁编造", "防幻觉"]),
    ("怎么把普通函数放进 LCEL 管道？",         "答", ["RunnableLambda"]),
    ("输出解析器有哪几种？",                  "答", ["StrOutputParser", "JsonOutputParser", "PydanticOutputParser"]),
    ("做 RAG 时为什么要清洗语料？",            "答", ["水印", "自指污染", "语料有洞"]),
    ("拒答阈值怎么确定？",                    "答", ["标定", "阈值必须"]),
    ("今天北京天气怎么样？",                  "拒", []),
    ("怎么做红烧肉？",                       "拒", []),
    ("Python 的 GIL 是什么？",               "拒", []),
    ("帮我订一张去上海的机票",                 "拒", []),
    ("Transformer 的注意力机制怎么实现？",      "拒", []),
]


def build_ground_truth():
    """把"关键词"翻译成"哪些 chunk id 是正确答案" """
    gt = []
    for q, expect, kws in CASES:
        rel = set()
        if expect == "答":
            for i, t in zip(ids, texts):
                if any(k.lower() in t.lower() for k in kws):
                    rel.add(i)
        gt.append({"query": q, "expect": expect, "relevant": rel, "kws": kws})
    return gt


# ============================================================
# 3. 四种检索方案（对比用）—— 展示"评估框架能帮你选出最优组合"
# ============================================================
def rank_vector(q, k=RECALL_K):
    return [d.id for d in retriever.invoke(q)][:k]


def rank_bm25(q, k=RECALL_K):
    sc = bm25.get_scores(list(jieba.cut(q)))
    return [ids[i] for i in sorted(range(len(sc)), key=lambda i: -sc[i])[:k]]


def rrf(lists, k=60, top_n=RECALL_K):
    s = {}
    for rl in lists:
        for r, doc_id in enumerate(rl, 1):
            s[doc_id] = s.get(doc_id, 0) + 1 / (k + r)
    return [i for i, _ in sorted(s.items(), key=lambda kv: -kv[1])[:top_n]]


def rerank_api(q, doc_ids, top_n=RERANK_TOP_N):
    r = requests.post("https://api.siliconflow.cn/v1/rerank",
        headers={"Authorization": f"Bearer {SILICONFLOW_KEY}"},
        json={"model": "BAAI/bge-reranker-v2-m3", "query": q,
              "documents": [doc_by_id[i].page_content for i in doc_ids],
              "top_n": top_n, "return_documents": False}, timeout=60)
    r.raise_for_status()
    return [doc_ids[x["index"]] for x in r.json()["results"]]


def pipeline_vector(q):                       # 方案A：单路向量
    return rank_vector(q)[:RERANK_TOP_N]


def pipeline_hybrid(q):                       # 方案B：混合+RRF
    return rrf([rank_vector(q), rank_bm25(q)])[:RERANK_TOP_N]


def pipeline_full(q):                         # 方案C：混合+RRF+rerank
    return rerank_api(q, rrf([rank_vector(q), rank_bm25(q)]))


PIPELINES = {
    "A 单路向量": pipeline_vector,
    "B 混合+RRF": pipeline_hybrid,
    "C 混合+RRF+rerank": pipeline_full,
}


# ============================================================
# 4. 指标实现（核心，手写一遍就懂了）
# ============================================================
def hit_rate_at_k(ranked, relevant, k):
    """top-k 里只要有一条正确就算命中（0 或 1）"""
    if not relevant:
        return None                      # 该拒答的样本不参与检索指标
    return 1.0 if set(ranked[:k]) & relevant else 0.0


def recall_at_k(ranked, relevant, k):
    """top-k 覆盖了正确内容的多少比例"""
    if not relevant:
        return None
    return len(set(ranked[:k]) & relevant) / len(relevant)


def mrr(ranked, relevant):
    """第一条正确内容的排名倒数（第1名=1.0，第2名=0.5，第5名=0.2）"""
    if not relevant:
        return None
    for r, doc_id in enumerate(ranked, 1):
        if doc_id in relevant:
            return 1.0 / r
    return 0.0


def ndcg_at_k(ranked, relevant, k):
    """考虑位置权重的排序质量：越靠前命中，贡献越大"""
    if not relevant:
        return None
    dcg = sum(1.0 / math.log2(r + 1)
              for r, doc_id in enumerate(ranked[:k], 1) if doc_id in relevant)
    ideal = sum(1.0 / math.log2(r + 1) for r in range(1, min(len(relevant), k) + 1))
    return dcg / ideal if ideal else 0.0


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else float("nan")


# ============================================================
# 5. 跑评估
# ============================================================
def evaluate(pipelines, gt):
    print("\n" + "=" * 88)
    print("【检索层指标】只看有没有捞到正确内容（不调 LLM）")
    print("=" * 88)
    print(f"{'方案':<20}{'HitRate@3':>11}{'Recall@3':>11}{'MRR':>9}{'nDCG@3':>9}{'HitRate@15':>12}")
    print("-" * 88)

    results = {}
    for name, fn in pipelines.items():
        hits3, rec3, mrrs, ndcgs, hits15 = [], [], [], [], []
        for c in gt:
            if c["expect"] != "答":
                continue
            ranked = fn(c["query"])
            hits3.append(hit_rate_at_k(ranked, c["relevant"], 3))
            rec3.append(recall_at_k(ranked, c["relevant"], 3))
            mrrs.append(mrr(ranked, c["relevant"]))
            ndcgs.append(ndcg_at_k(ranked, c["relevant"], 3))
            # 召回宽度指标：看放大到 15 条时的上限
            wide = fn(c["query"]) if name != "C 混合+RRF+rerank" else None
            if name == "A 单路向量":
                wide = rank_vector(c["query"])
            elif name == "B 混合+RRF":
                wide = rrf([rank_vector(c["query"]), rank_bm25(c["query"])])
            if wide:
                hits15.append(hit_rate_at_k(wide, c["relevant"], 15))
        r = {"hit3": mean(hits3), "rec3": mean(rec3), "mrr": mean(mrrs),
             "ndcg3": mean(ndcgs), "hit15": mean(hits15)}
        results[name] = r
        print(f"{name:<20}{r['hit3']:>11.2%}{r['rec3']:>11.2%}{r['mrr']:>9.3f}"
              f"{r['ndcg3']:>9.3f}{r['hit15']:>12.2%}")
    return results


def show_ground_truth(gt):
    print("=" * 88)
    print("【题库与自动标注结果】")
    print("=" * 88)
    for c in gt:
        n = len(c["relevant"])
        tag = f"{n} 条正确答案" if c["expect"] == "答" else "应拒答"
        flag = "  ⚠️ 标注为空！" if (c["expect"] == "答" and n == 0) else ""
        print(f"  [{c['expect']}] {c['query']:<34} → {tag}{flag}")
    empty = [c["query"] for c in gt if c["expect"] == "答" and not c["relevant"]]
    if empty:
        print(f"\n⚠️ 有 {len(empty)} 条「该答」的题在语料里找不到关键词命中的 chunk，"
              f"说明关键词写错了或语料缺内容：{empty}")


# ============================================================
# 6. 生成层评估：拒答正确率 + 答案正确率 + 幻觉率
# ============================================================
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

REFUSAL_MARK = "没有相关内容"      # 约定好的拒答话术标志


def full_chain(q):
    """完整链路：召回 → RRF → rerank → 阈值 → (拒答 or 生成)"""
    fused = rrf([rank_vector(q), rank_bm25(q)])
    reranked = requests.post(
        "https://api.siliconflow.cn/v1/rerank",
        headers={"Authorization": f"Bearer {SILICONFLOW_KEY}"},
        json={"model": "BAAI/bge-reranker-v2-m3", "query": q,
              "documents": [doc_by_id[i].page_content for i in fused],
              "top_n": RERANK_TOP_N, "return_documents": False}, timeout=60).json()
    docs = [(fused[x["index"]], x["relevance_score"]) for x in reranked["results"]]
    kept = [(i, s) for i, s in docs if s >= THRESHOLD]
    if not kept:
        return None, "拒答(阈值)"          # 省钱路径：不调 LLM
    ctx = "\n\n".join(doc_by_id[i].page_content for i, _ in kept)
    return chain.invoke({"context": ctx, "question": q}), "生成"


def evaluate_generation(gt):
    print("\n" + "=" * 88)
    print("【生成层指标】看最终答案（要调 LLM，慢且贵）")
    print("=" * 88)

    n_answer = n_refuse = 0
    correct = refused_by_llm = hallucinated = 0
    refused_ok = 0
    details = []

    for c in gt:
        ans, mode = full_chain(c["query"])
        said_no = ans is not None and REFUSAL_MARK in ans

        if c["expect"] == "答":
            n_answer += 1
            # 答对了没？用关键词覆盖做确定性代理判断
            if ans is not None and any(k.lower() in ans.lower() for k in c["kws"]):
                correct += 1
                mark = "✅"
            elif said_no:
                refused_by_llm += 1
                mark = "❌漏答"          # 有答案却拒答 = 假拒答
            else:
                hallucinated += 1
                mark = "⚠️答案不符"
        else:
            n_refuse += 1
            # 该拒答的：阈值拒 or LLM 拒 都算对
            if ans is None or said_no:
                refused_ok += 1
                mark = "✅"
            else:
                hallucinated += 1
                mark = "❌幻觉"
        details.append((c["query"], mark, mode, (ans or "")[:60]))

    print(f"{'问题':<36}{'判定':<10}{'路径':<12}答案片段")
    print("-" * 88)
    for q, mark, mode, a in details:
        print(f"{q:<36}{mark:<10}{mode:<12}{a}")

    print("\n---- 汇总 ----")
    print(f"  答案正确率   : {correct}/{n_answer} = {correct/n_answer:.1%}")
    print(f"  假拒答(漏答) : {refused_by_llm}/{n_answer} = {refused_by_llm/n_answer:.1%}")
    print(f"  拒答正确率   : {refused_ok}/{n_refuse} = {refused_ok/n_refuse:.1%}")
    print(f"  幻觉数(兜底失败): {hallucinated}  ← 越低越好，理想为 0")
    return {"correct": correct / n_answer, "false_refuse": refused_by_llm / n_answer,
            "refuse_ok": refused_ok / n_refuse, "hallucinated": hallucinated}


if __name__ == "__main__":
    gt = build_ground_truth()
    show_ground_truth(gt)
    if "--dump" in sys.argv:
        sys.exit(0)
    evaluate(PIPELINES, gt)
    if "--gen" in sys.argv:
        evaluate_generation(gt)
