# AI Agent 开发实践 · LLM 应用层（W1-W8）

从**裸调 API** 到 **RAG 检索优化**的完整代码记录。

这个仓库的特点：**每条结论都对应可运行的代码和实测数字**，不是从教程抄来的架构图。
适合想看"一个人怎么从零把 RAG 做到能上线"的读者。

---

## 一、这个仓库回答什么问题

| 问题 | 在哪个文件 |
|---|---|
| 不用框架怎么调 LLM？和用 LangChain 差在哪？ | `TEST.py` · `LLM.py` · `w5langchain.py` |
| 怎么让模型"动手"而不是"动嘴"？ | `day3_function_calling.py` · `day4.React Agent.py` |
| 手写 ReAct 循环怎么防止转死？ | `day4.React Agent.py`（`for i in range(10)` 硬熔断） |
| RAG 的每一环到底在干什么？ | `W7D1_embdding_api.py` → `W7D2_chunk.py` → `W7D5_kb.py` |
| 向量检索为什么不够用？ | `W8D1_BM25.py` · `W8D2_hybrid_rrf.py` |
| RRF 为什么不能换成加权求和？ | `W8D2_hybrid_rrf.py` |
| 拒答阈值凭什么定 0.326？ | `W8D3_calibrate.py` · `W8D3_rerank.py` |
| 怎么证明"改完真的变好了"？ | `W8D4_eval.py` |

---

## 二、检索链路（本仓库的核心）

### 建库（离线跑一次）
```
Markdown 笔记
   └─ 双层切分：先按 Markdown 标题切，再按 300 字 / 40 字重叠切
        └─ 过滤噪声（<20 字的碎块、水印行）
             └─ bge-m3 向量化（1024 维）
                  └─ Chroma 持久化  →  94 个 chunk
```

### 检索（每次提问）
```
query
  ├─ 向量召回 top15   （懂语义，但会把专有名词泛化掉）
  └─ BM25  召回 top15 （认字面，但不懂同义词）
        └─ RRF 融合：score += 1/(60+rank)，只融合名次不碰分数
             └─ bge-reranker-v2-m3 精排（Cross-Encoder，逐对打分）
                  └─ 阈值 0.326 过滤 —— 不过则直接拒答，不调 LLM
                       └─ top3 拼成带小节标题的 context
                            └─ DeepSeek 生成（prompt 里约束"只能依据笔记回答"）
```

**三个"为什么分层"**（面试常问）：

1. **召回要两路** —— 单路会系统性漏掉一整类查询。
   实测：问"Function Calling 和 ReAct 有什么区别"，向量检索 top4 全是噪声（装包配置、天气 404），
   BM25 精确命中了 ReAct 定义和核心对比表。
2. **融合用 RRF 而不是加权求和** —— 两路分数**量纲不同**：余弦相似度 0~1，
   BM25 是无上界词频累加（能到几十）。写 `0.3*向量 + 0.7*BM25` 时，
   实际贡献是 `0.216 : 5.95`，向量那一路几乎被完全忽略，权重根本调不动。
   RRF 只融合名次，天然免疫分数尺度差异。
3. **重排要独立一层** —— 召回决定"能不能捞到"，重排决定"排第几"，是两个问题。
   Bi-Encoder 把 query 和 doc 分开编码（编码 doc 时不知道 query 是什么），快但粗；
   Cross-Encoder 把两者拼在一起送进模型做 token 级交互，准但慢，且**无法预存文档向量**，
   所以只能对少量候选做精排。

---

## 三、实测数据

### 拒答阈值标定（`W8D3_calibrate.py`）
| 阈值来源 | 阈值 | 准确率 |
|---|---|---|
| 拍脑袋 | 0.05 | 81.2%（13/16） |
| 扫描中的候选点 | 0.138 | 87.5%（14/16） |
| **代码采用的阈值** | **0.326** | **93.8%（15/16）** |
| 阈值过高 | ≥0.74 | 81.2%（13/16）← 假拒答 |

> 两点值得注意：
> 1. **不是"阈值越高越安全"** —— 0.05 → 81.2%、0.138 → 87.5%、0.326 → 93.8% 一路往上，
>    但**再往上（≥0.74）就跌回 81.2%**：开始把该回答的问题也拒了（假拒答）。
>    所以最优值在 0.326 附近，而不是越高越好。
> 2. **93.8% 已经是这套语料的天花板** —— 剩下那 1 条错案（"WC所有语法？"被 reranker
>    打了 0.92 分）属于语料层问题，**调阈值治不好**，只能回语料和 prompt 层解决。

> ⚠️ 一条重要认知：**reranker 输出的是"排序分数"，不是"相关性概率"**。
> 它只保证 A 比 B 更相关，不保证"0.326 以上就算相关"。所以绝对阈值**不可移植**
> （换模型、换语料就失效），只能当"省钱过滤器"（拦掉明显垃圾），
> 最终判断交给 prompt 里的拒答约束。

### 评估框架（`W8D4_eval.py`）
两个层次、四个指标：

| 层次 | 指标 | 衡量什么 |
|---|---|---|
| 检索层（不调 LLM，快、可回归） | Hit Rate@k | 能不能捞到 |
| | Recall@k | 捞得全不全 |
| | MRR | 排得靠不靠前 |
| | nDCG@k | 最接近用户体感 |
| 生成层（调 LLM，慢、有噪声） | 关键词覆盖率 | 答案是否含应有术语 |
| | 拒答正确率 | 该拒的拒了、不该拒的没乱拒 |

评估集 20 条（15 条语料里确实有答案 / 5 条确实没有），并对
**纯向量 / 纯 BM25 / 混合 RRF / 混合 RRF + rerank** 四种方案做横向对比。

---

## 四、踩过的坑（这部分比代码更值钱）

1. **"语料有洞"** —— 一个问题反复答不出，查了半天发现是我的手册里**根本没写那个术语的定义**，
   直接从公式开始讲。RAG 里最贵的 bug 不是模型差，是知识库缺。它表现为"模型说不知道"，
   看起来像模型的问题，实际是数据的问题。
2. **"自指污染"** —— 语料里有一段在**讨论**某个测试查询，reranker 给它打了 0.92 的高分。
   它分不清"这段在**解释** X"和"这段在**讨论**有人问过 X"。这个连 reranker 也救不了，
   只能清理语料。
3. **扩库后效果反而变差** —— 一开始以为是"语料多了干扰变多"，后来发现是
   **我的对比脚本只用了一路向量检索**：新补的那块内容在 90 个 chunk 里向量排第 27 名，
   但 BM25 排第 1 名。换回完整链路效果立刻回来。
   → 同一个语料，检索方案不同，结论可能完全相反。

---

## 五、快速开始

```bash
git clone <本仓库地址>
cd agent-learn

python -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

copy .env.example .env            # macOS/Linux: cp .env.example .env
# 然后编辑 .env，填入自己的 DEEPSEEK_API_KEY / SILICONFLOW_API_KEY
```

按这个顺序跑（先建库，再检索）：

```bash
python W8_build_kb_v2.py     # 1. 建库 → 生成 ./kb_chroma_v2（94 chunk）
python W8D3_calibrate.py     # 2. 阈值标定（可选，验证 0.326 这个数）
python W8D3_rerank.py        # 3. 完整链路：双路召回 → RRF → rerank → 拒答 → 生成
python W8D4_eval.py          # 4. 评估：四种检索方案横向对比
python W8D4_eval.py --dump   #    只看题库和标注结果
```

> 加 `--ref` 参数可看参考实现（`W8D2_hybrid_rrf.py` / `W8D3_rerank.py` 里把
> 关键函数留成了待填的练习形式）。

---

## 六、目录结构

```
agent-learn/
├── TEST.py · LLM.py · weather.py          # W1-2  裸调 API / messages / system prompt
├── day3_function_calling.py               # W3    Function Calling（骨架三件套）
├── day4.React Agent.py                    # W4    手写 ReAct 循环（含硬熔断）
├── w5*.py · W5day2*.py                    # W5    LangChain 入门 / 输出解析器 / Tool
├── W6*.py                                 # W6    LCEL / 流式 / 结构化输出
├── W7D1_embdding_api.py                   # W7-1  embedding 接口
├── W7D2_chunk.py                          # W7-2  切分策略
├── W7D4_rag_chain.py                      # W7-4  RAG 链路拼装（※ 文件名建议去掉全角冒号）
├── W7D5_kb.py                             # W7-5  建库（27 chunk 初版）
├── W8D1_BM25.py                           # W8-1  BM25 关键词检索
├── W8D2_hybrid_rrf.py                     # W8-2  双路召回 + RRF 融合
├── W8D3_calibrate.py                      # W8-3  阈值标定实验
├── W8D3_rerank.py                         # W8-3  rerank 精排 + 阈值拒答（完整链路）
├── W8D4_eval.py                           # W8-4  RAG 评估框架
├── W8D4_threshold_sweep.py                # W8-4  阈值扫描
├── W8_build_kb_v2.py                      # 建库 v2（笔记 + 速查手册 → 94 chunk）
├── AI_Agent速查手册_W1-W8.md               # W1-W8 全栈速查 + 自测题
├── 复习笔记_AI_Agent_Day1-4.md              # 早期笔记
├── 面试话术_RAG项目.md                      # 这个项目的面试问答弹药库
└── requirements.txt
```

---

## 七、已知不足与后续计划

- **语料是玩具规模**（94 chunk）。链路和方法是生产级的，但规模不是。
  百万级要换 Milvus / Qdrant，并做 HNSW 参数调优和分片。
- **没有服务化**：目前是脚本，计划用 FastAPI + SSE 流式包成服务并 Docker 化部署。
- **没有 Agent 编排**：检索目前是固定流程，计划用 LangGraph 改成
  "自己决定要不要检索、检索几轮、答不出要不要追问"的有状态 Agent。
- **评估集是自动标注的**（用关键词反查 chunk），生产环境应人工标注 query → 相关 chunk。

---

## 技术栈

| 环节 | 选型 |
|---|---|
| 对话模型 | DeepSeek |
| 嵌入模型 | BAAI/bge-m3（1024 维，硅基流动） |
| 重排模型 | BAAI/bge-reranker-v2-m3 |
| 向量库 | Chroma |
| 编排 | LangChain（LCEL）/ hand-written ReAct |
| 中文分词 | jieba + rank_bm25 |
