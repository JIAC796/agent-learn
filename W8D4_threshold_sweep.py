"""
阈值 × 端到端效果 扫描
=============================================
目的：用【独立于标定集】的题库，看阈值对最终效果的真实影响。
关键问题：阈值高会"假拒答"，阈值低会"多调 LLM"。
         到底该取多少？—— 用数据说话，看三件事：
           ① 答案正确率（越低说明假拒答越多）
           ② 幻觉数（阈值低会放进噪声，但 LLM 能兜住多少？）
           ③ LLM 调用占比（成本！阈值高省钱，阈值低费钱）
"""
import os, sys, json, math, jieba, requests
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
jieba.setLogLevel(20)
KEY = os.getenv("SILICONFLOW_API_KEY")
DB = os.getenv("KB_PATH", "./kb_chroma_v2")

emb = OpenAIEmbeddings(model="BAAI/bge-m3", api_key=KEY,
                       base_url="https://api.siliconflow.cn/v1")
llm = ChatOpenAI(model="deepseek-v4-flash", api_key=os.getenv("DEEPSEEK_API_KEY"),
                 base_url="https://api.deepseek.com")
vs = Chroma(collection_name="langchain", embedding_function=emb, persist_directory=DB)
raw = vs.get(include=["documents", "metadatas"])
ids, texts = raw["ids"], raw["documents"]
doc_by_id = {i: Document(page_content=t, metadata=m or {})
             for i, t, m in zip(ids, texts, raw["metadatas"])}
if not ids:
    raise SystemExit("❌ 空库")
bm25 = BM25Okapi([list(jieba.cut(t)) for t in texts])
retr = vs.as_retriever(search_kwargs={"k": 15})

CASES = [
    ("BM25 是什么？", "答", ["BM25", "词频统计"]),
    ("RRF 为什么用名次不用分数？", "答", ["尺度", "名次", "加权"]),
    ("rerank 和向量检索有什么区别？", "答", ["Bi-Encoder", "Cross-Encoder", "重排"]),
    ("Bi-Encoder 和 Cross-Encoder 差在哪？", "答", ["Bi-Encoder", "Cross-Encoder"]),
    ("tool_call_id 是干什么的？", "答", ["tool_call_id"]),
    ("json_mode 要注意什么？", "答", ["json_mode"]),
    ("chunk_size 和 chunk_overlap 设多少？", "答", ["chunk_size=300", "chunk_overlap=40"]),
    ("LCEL 是什么？", "答", ["LangChain Expression Language", "LCEL 是什么"]),
    ("ReAct 循环怎么防止死循环？", "答", ["range(10)", "熔断"]),
    ("Function Calling 和 ReAct 的区别？", "答", ["ReAct = Reasoning", "关键对比"]),
    ("RAG 怎么防止模型编造？", "答", ["严禁编造", "防幻觉"]),
    ("怎么把普通函数放进 LCEL 管道？", "答", ["RunnableLambda"]),
    ("输出解析器有哪几种？", "答", ["StrOutputParser", "JsonOutputParser", "PydanticOutputParser"]),
    ("做 RAG 时为什么要清洗语料？", "答", ["水印", "自指污染", "语料有洞"]),
    ("拒答阈值怎么确定？", "答", ["标定", "阈值必须"]),
    ("今天北京天气怎么样？", "拒", []),
    ("怎么做红烧肉？", "拒", []),
    ("Python 的 GIL 是什么？", "拒", []),
    ("帮我订一张去上海的机票", "拒", []),
    ("Transformer 的注意力机制怎么实现？", "拒", []),
]

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
MARK = "没有相关内容"


def rrf(lists, k=60, top_n=15):
    s = {}
    for rl in lists:
        for r, i in enumerate(rl, 1):
            s[i] = s.get(i, 0) + 1 / (k + r)
    return [i for i, _ in sorted(s.items(), key=lambda kv: -kv[1])[:top_n]]


def top_scores(q):
    """返回 [(doc_id, rerank_score), ...] top3"""
    v = [d.id for d in retr.invoke(q)]
    sc = bm25.get_scores(list(jieba.cut(q)))
    b = [ids[i] for i in sorted(range(len(sc)), key=lambda i: -sc[i])[:15]]
    fused = rrf([v, b])
    r = requests.post("https://api.siliconflow.cn/v1/rerank",
        headers={"Authorization": f"Bearer {KEY}"},
        json={"model": "BAAI/bge-reranker-v2-m3", "query": q,
              "documents": [doc_by_id[i].page_content for i in fused],
              "top_n": 3, "return_documents": False}, timeout=60).json()
    return [(fused[x["index"]], x["relevance_score"]) for x in r["results"]]


# 预先算好所有 query 的 rerank 结果（避免每个阈值重复调 API）
print(f"库 {len(ids)} chunk，预计算 {len(CASES)} 个 query 的 rerank 结果...")
cache = {}
for q, exp, kws in CASES:
    cache[q] = top_scores(q)
    print(f"  {q:<36} top1={cache[q][0][1]:.4f}")

print(f"\n{'阈值':<8}{'正确答案':<12}{'假拒答':<10}{'拒答正确':<12}{'幻觉':<8}{'LLM调用率':<10}")
print("-" * 62)

report = []
for th in [0.0, 0.05, 0.1, 0.2, 0.326, 0.5]:
    ok = fn = rok = hal = calls = 0
    n_ans = sum(1 for _, e, _ in CASES if e == "答")
    n_rej = sum(1 for _, e, _ in CASES if e == "拒")

    for q, exp, kws in CASES:
        kept = [i for i, s in cache[q] if s >= th]
        if not kept:
            ans, used = None, 0
        else:
            ctx = "\n\n".join(doc_by_id[i].page_content for i in kept)
            ans = chain.invoke({"context": ctx, "question": q})
            used = 1
        calls += used
        said_no = ans is not None and MARK in ans

        if exp == "答":
            if ans is not None and any(k.lower() in ans.lower() for k in kws):
                ok += 1
            elif said_no:
                fn += 1
            else:
                hal += 1
        else:
            if ans is None or said_no:
                rok += 1
            else:
                hal += 1

    rate = calls / len(CASES)
    print(f"{th:<8}{f'{ok}/{n_ans}':<12}{fn:<10}{f'{rok}/{n_rej}':<12}{hal:<8}{rate:<10.0%}")
    report.append({"th": th, "correct": ok, "false_refuse": fn, "refuse_ok": rok,
                   "hallucinated": hal, "llm_call_rate": rate})

best = max(report, key=lambda r: r["correct"] - r["hallucinated"] * 2)
print(f"\n★ 综合最优（正确答案数 - 2×幻觉）: 阈值 {best['th']}")
print(f"  正确答案 {best['correct']}，假拒答 {best['false_refuse']}，"
      f"幻觉 {best['hallucinated']}，LLM调用率 {best['llm_call_rate']:.0%}")
