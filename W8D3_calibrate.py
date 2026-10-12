"""
W8D3 阈值标定实验
=============================================
目的：用数据找出「相关」和「不相关」的分界线，而不是拍脑袋定 0.05。

方法：
  1) 造测试集：标注哪些问题语料里"有答案"、哪些"没答案"
  2) 每个问题跑完整链路（向量+BM25 → RRF → rerank）
  3) 统计有答案组的 top1 分布 vs 没答案组的 top1 分布
  4) 扫描所有候选阈值，找准确率最高的那个

这是 RAG 工程的标准做法：**阈值必须在自己数据上标定**。
"""
import os, sys, json, time, jieba, requests
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

load_dotenv("./.env")
jieba.setLogLevel(20)
KEY = os.getenv("SILICONFLOW_API_KEY")
DB = "./kb_chroma_v2"
RECALL_K = 15
LABEL_K = 10        # 标定时看 top10，判断"有没有相关内容进入候选"

emb = OpenAIEmbeddings(model="BAAI/bge-m3", api_key=KEY,
                       base_url="https://api.siliconflow.cn/v1")

# ===== 空库保护：路径不对时给出明确提示，而不是 ZeroDivisionError =====
if not os.path.exists(DB):
    raise SystemExit(
        f"\n❌ 找不到向量库：{os.path.abspath(DB)}\n"
        f"   当前运行目录：{os.getcwd()}\n"
        f"   DB 是相对路径，会从【运行目录】开始找。两种解法：\n"
        f"     1) 改 DB 为绝对路径，例如 r\"C:\\Users\\xy\\Desktop\\学AI Agent开发\\W8D2\\kb_chroma_v2\"\n"
        f"     2) 先在本目录用 W8_build_kb_v2.py 建库（推荐，一次搞定）\n"
    )

vs = Chroma(collection_name="langchain", embedding_function=emb, persist_directory=DB)
raw = vs.get(include=["documents", "metadatas"])
ids, texts, metas = raw["ids"], raw["documents"], raw["metadatas"]

if len(ids) == 0:
    raise SystemExit(
        f"\n❌ 库 {os.path.abspath(DB)} 是空的（0 个 chunk）。\n"
        f"   常见原因：DB 路径指向了一个不存在的位置，Chroma 会默默新建一个空库。\n"
        f"   请确认路径，或先跑 W8_build_kb_v2.py 建库。\n"
    )

doc_by_id = {i: Document(page_content=t, metadata=m or {})
             for i, t, m in zip(ids, texts, metas)}
bm25 = BM25Okapi([list(jieba.cut(t)) for t in texts])
retr = vs.as_retriever(search_kwargs={"k": RECALL_K})

# ===== 测试集：(问题, 期望, 用于判定"命中"的关键词列表) =====
# 期望 = 答  → 候选里应该出现含任一关键词的块
# 期望 = 拒  → 语料里真的没有（关键词列表为空）
CASES = [
    # ---- 应该有答案（语料里明确有）----
    ("BM25 是什么？",                      "答", ["BM25", "词频统计"]),
    ("JsonOutputParser 和 StrOutputParser 有什么区别？", "答", ["JsonOutputParser", "StrOutputParser", "解析器"]),
    ("rerank 和向量检索有什么区别？",        "答", ["Bi-Encoder", "Cross-Encoder", "召回", "重排"]),
    ("RRF 为什么用名次不用分数？",           "答", ["尺度", "名次", "加权"]),
    ("tool_call_id 是干什么的？",           "答", ["tool_call_id", "tool"]),
    ("json_mode 要注意什么？",              "答", ["json_mode", "json"]),
    ("chunk_size 和 chunk_overlap 设多少？", "答", ["chunk_size", "chunk_overlap", "300"]),
    ("LCEL 里的管道符是干什么的？",          "答", ["LCEL", "管道", "|"]),
    ("ReAct 循环怎么防止死循环？",           "答", ["range(10)", "熔断", "最大轮数"]),
    ("RAG 怎么防止模型编造？",              "答", ["严禁编造", "没有相关内容", "防幻觉"]),
    # ---- 应该拒答（语料里真的没有）----
    ("今天北京天气怎么样？",                "拒", []),
    ("怎么做红烧肉？",                      "拒", []),
    ("Python 的 GIL 是什么？",             "拒", []),
    ("帮我订一张去上海的机票",              "拒", []),
    ("Transformer 的注意力机制怎么实现？",   "拒", []),
    ("WC所有语法？",                        "拒", []),
]


def rerank(query, cands, top_n=LABEL_K):
    r = requests.post("https://api.siliconflow.cn/v1/rerank",
        headers={"Authorization": f"Bearer {KEY}"},
        json={"model": "BAAI/bge-reranker-v2-m3", "query": query,
              "documents": [d.page_content for _, d, _ in cands],
              "top_n": top_n, "return_documents": False}, timeout=60)
    r.raise_for_status()
    return [(cands[x["index"]][1], x["relevance_score"]) for x in r.json()["results"]]


def rrf(lists, k=60, top_n=RECALL_K):
    s, m = {}, {}
    for rl in lists:
        for r, (i, d) in enumerate(rl, 1):
            m[i] = d
            s[i] = s.get(i, 0) + 1 / (k + r)
    return [(i, m[i], v) for i, v in sorted(s.items(), key=lambda kv: -kv[1])[:top_n]]


def pipeline(q):
    v = [(d.id, d) for d in retr.invoke(q)]
    sc = bm25.get_scores(list(jieba.cut(q)))
    b = [(ids[i], doc_by_id[ids[i]])
         for i in sorted(range(len(sc)), key=lambda i: -sc[i])[:RECALL_K]]
    return rerank(q, rrf([v, b]), top_n=LABEL_K)


print(f"库 {len(ids)} chunk | 测试集 {len(CASES)} 条\n")
print(f"{'问题':<40}{'期望':<5}{'top1':>10}{'top2':>10}{'top3':>10}  命中?")
print("-" * 100)

rows = []
for q, exp, kws in CASES:
    rr = pipeline(q)
    scores = [s for _, s in rr]
    hit = None
    if kws:
        # 看 top3 里有没有关键词命中的块（判断检索是否真的捞到了）
        hit = any(any(k.lower() in d.page_content.lower() for k in kws)
                  for d, _ in rr[:3])
    rows.append({"q": q, "exp": exp, "top1": scores[0], "top3": scores[:3], "hit": hit})
    mark = "✅" if hit else ("—" if hit is None else "❌漏")
    print(f"{q:<40}{exp:<5}{scores[0]:>10.5f}{scores[1]:>10.5f}{scores[2]:>10.5f}  {mark}")
    time.sleep(0.3)

# ===== 分布统计 =====
answer = [r["top1"] for r in rows if r["exp"] == "答"]
reject = [r["top1"] for r in rows if r["exp"] == "拒"]
print(f"\n【有答案组 top1】 n={len(answer)}  min={min(answer):.5f}  max={max(answer):.5f}  "
      f"中位={sorted(answer)[len(answer)//2]:.5f}")
print(f"【该拒答组 top1】 n={len(reject)}  min={min(reject):.5f}  max={max(reject):.5f}  "
      f"中位={sorted(reject)[len(reject)//2]:.5f}")

print(f"\n有答案组明细: {[round(x,4) for x in sorted(answer)]}")
print(f"该拒答组明细: {[round(x,5) for x in sorted(reject)]}")

# ===== 阈值扫描：找准确率最高的阈值 =====
print(f"\n{'阈值':>9}{'答对(放行)':>13}{'误放(噪音)':>13}{'准确率':>10}   错误明细")
print("-" * 96)
best = None
cands = sorted(set(round(x, 5) for x in answer + reject))
for th in cands:
    tp = sum(1 for r in rows if r["exp"] == "答" and r["top1"] >= th)
    fn = sum(1 for r in rows if r["exp"] == "答" and r["top1"] < th)
    fp = sum(1 for r in rows if r["exp"] == "拒" and r["top1"] >= th)
    tn = sum(1 for r in rows if r["exp"] == "拒" and r["top1"] < th)
    acc = (tp + tn) / len(rows)
    wrong = [r["q"][:16] for r in rows
             if (r["exp"] == "答") != (r["top1"] >= th)]
    if best is None or acc > best[1]:
        best = (th, acc, wrong)
    if acc >= 0.8:
        print(f"{th:>9}{tp:>13}{fp:>13}{acc:>10.1%}   {wrong}")

print(f"\n★ 最佳阈值 = {best[0]}  准确率 = {best[1]:.1%}")
print(f"  错判: {best[2]}")
print(f"\n（对比：拍脑袋的 0.05 准确率 = "
      f"{sum(1 for r in rows if (r['exp']=='答')==(r['top1']>=0.05))/len(rows):.1%}）")
