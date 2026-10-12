# AI Agent 开发 · 全栈速查手册（W1-W8）

> 用途：忘掉某个点时，顺着**依赖链**找它在哪一层、上下游是什么。
> 结构不按周堆，按「从裸调 API → 到检索质量优化」的依赖顺序排。
> 每条技术点都对应本地真实代码，可跳回去看原文。

---

## ⚠️ 先修正两处旧笔记的错误

旧的 `复习笔记_AI_Agent_Day1-4.md` 有两处与代码不符，以本手册为准：

1. **第三方接口选型（旧笔记里的"天气 API"）**：旧笔记写"最终选 Open-Meteo"是错的。
   **实际代码用的是一个免费的无 Key 天气接口**（`day3_function_calling.py:72`、
   `day4.React Agent.py:20`，用 `?format=j1` 取 `current_condition[0]` 的
   `weatherDesc/temp_C/humidity`）。这里的重点是**接口选型过程**，不是天气数据本身。
2. **Day4 主循环**：旧笔记写 `while True` + `available_functions[...](**parsed["args"])`。
   **实际是 `for i in range(10)` 硬熔断**（`day4:136`），且传参是**单个位置参数**
   （`day4:153`），`parse_react` 返回的是 **tuple 不是 dict**（`day4:100-126`）。

### ✅ 已修（W01，2026-10-10）：Key 改为环境变量读取

原来 `day3_function_calling.py` 和 `day4.React Agent.py` 明文硬编码了 `sk-...`。
现已改成顶部 `load_dotenv()` + `os.getenv("DEEPSEEK_API_KEY")`
（现在读 key 的行号：`day3:18` / `day4:11`）。

⚠️ **代码改了 ≠ 旧 Key 失效**：仍必须去 DeepSeek 后台 revoke 旧 Key 并重发，
否则那个 Key 一直有效。

---

## 第 0 层 · 调用模型

| 技术点 | 一句话 | 代码 |
|---|---|---|
| LLM API | HTTP 请求发 messages，模型返回文本 | `TEST.py` |
| messages | `system → user → assistant → user...` 顺序排列 | `LLM.py` |
| system prompt | 定角色、语气、边界 | `TEST.py:14` |
| `.env` + `load_dotenv` | key 不进源码 | `TEST.py:5-7` |

**两种调用方式**（对照理解 LangChain 到底省了什么）：
- 裸 SDK：`client.chat.completions.create(model=..., messages=[...])` → `response.choices[0].message.content`
- 手写 HTTP：`data=json.dumps(body, ensure_ascii=False).encode("utf-8")` + `Content-Type: application/json; charset=utf-8`（`day3:120-128`）

**易错**
- base_url 是 `https://api.deepseek.com`
- 不取 `.content` 拿到的是对象不是字符串（W5 后也一样）
- 两份 W3-4 代码都**没设 temperature/max_tokens，无重试无流式**

---

## 第 1 层 · Prompt 工程

| 技巧 | 干什么 | 代码 |
|---|---|---|
| 结构化输出 | 让 AI 返回 JSON，方便解析 | 旧笔记 Day2 |
| 分隔符 `"""` | 区分"指令"和"数据"，**防注入** | 旧笔记 Day2 |
| few-shot | 给 2-3 个例子教格式，比写规则管用 | 旧笔记 Day2 |

**核心认知：Prompt 不是聊天，是编程。**

**易错**
- 模板里输出真 JSON 花括号要**双写转义**：`'{{"plan": [...]}}'`（`w5day2_3.py:17`）

---

## 第 2 层 · 让模型动手（Function Calling → ReAct）

### 2.1 Function Calling（`day3_function_calling.py`）

> **LLM 不动手，只动嘴。** 它说"我想调哪个工具、传什么参数"，你执行，把结果喂回去。

```
用户输入 → LLM 判断
  ├─ 不需要 → 直接回答
  └─ 需要 → 返回 tool_calls（工具名+参数）→ 你执行 → 结果喂回 → LLM 出最终答案
```

**骨架三件套**：JSON Schema 描述工具 → `available_functions` 映射表 → 两轮调用。

**🔑 命门（旧笔记漏了，最容易卡死）**
```python
msg = data["choices"][0]["message"]
messages.append(msg)                                   # ① 带 tool_calls 的 assistant 消息原样带回
for tc in msg["tool_calls"]:                           # ② 一轮可能返回多个（并行调用）
    fn_args = json.loads(tc["function"]["arguments"])  # ③ 是 JSON 字符串！要先 loads
    result = available_functions[tc["function"]["name"]](**fn_args)
    messages.append({"role": "tool",
                     "tool_call_id": tc["id"],         # ④ 少了这个第二轮必 400
                     "content": result})
```

### 2.2 手写 ReAct（`day4.React Agent.py`）

> **ReAct = Reasoning + Acting**。`Thought → Action → Observation` 循环，**LLM 自己决定何时停**。

**与 Function Calling 的区别**

| | Day3 Function Calling | Day4 ReAct |
|---|---|---|
| 工具描述 | JSON Schema | **纯文本 Prompt**（`TOOLS_DESC`） |
| 调用决策 | `tool_calls` 字段（结构化） | 文本里输出 `Action:`（靠正则解析） |
| 循环控制 | 你写死两轮 | LLM 自己决定 |
| 可靠性 | 高（协议保证） | 低（靠模型听话） |

**真实实现的关键细节（旧笔记全写错了）**
- 循环是 **`for i in range(10)` 硬熔断**，超限返回"超过最大轮数"（`day4:136,164`）
- `parse_react` 返回 **tuple**：`("final",a)` / `("action",tool,inp)` / `("thought",text)`
- **`Final Answer` 判定优先于 `Action`** —— 模型顺手带一句 Final Answer，Action 会被吞
- Thought 其实**没被解析**，是三者都不匹配时的兜底整段原文
- Observation 是**人为注入的独立 user 消息**，不在 Prompt 模板里
- 传参是**单个位置参数** `available_functions[name](tool_input)`，**只支持单参数工具**（多参数必炸），且工具名写错直接 KeyError
- `calculate` 有**字符白名单安检**才 `eval`
- 工具**返回错误字符串而不是抛异常** —— 有意设计，异常会打断循环

---

## 第 3 层 · LangChain 抽象（W5-W6）

### 3.1 固定三件套（每个文件都长这样）

```python
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
load_dotenv()

llm = ChatOpenAI(model="deepseek-v4-flash",
                 api_key=os.getenv("DEEPSEEK_API_KEY"),
                 base_url="https://api.deepseek.com")
prompt = ChatPromptTemplate.from_template("你是{role}，请用{style}回答：\n{question}")
chain = prompt | llm | StrOutputParser()          # ← LCEL
result = chain.invoke({"role": ..., "style": ..., "question": ...})
```

**LCEL 是什么**：LangChain Expression Language（LangChain 表达式语言），
W5-W6 学的核心抽象。它用管道符 `|` 把「提示词模板 → 模型 → 输出解析器」
串成一条可组合的链，每个节点叫 Runnable。**LCEL 的 `|` 就是"左边产物喂右边，数据单向流动"**。
它的价值是：组件像积木一样可插拔（换模型只改中间一节），还天然支持 `invoke`/`stream`/`batch` 三种调用方式。

```python
chain = prompt | llm | StrOutputParser()   # ← 这就是 LCEL 写法
```

### 3.2 OutputParser 的意义 = **改变返回类型**

| 链 | 返回 |
|---|---|
| `prompt \| llm` | `AIMessage` 对象（要 `.content`） |
| `prompt \| llm \| StrOutputParser()` | `str` |
| `prompt \| llm \| JsonOutputParser()` | `dict`（可 `data["answer"]`） |

> 这是他自己做过的对照实验（`W5day2OutputParser.py:27-40`）。**JSON 格式完全靠 prompt 约束，parser 不纠错。**

### 3.3 Runnable 家族

| 组件 | 作用 | 代码 |
|---|---|---|
| `RunnableLambda(fn)` | 把**普通函数**变成管道节点 | `emoji_node = RunnableLambda(add_emoji)` |
| 同上，改输入格式 | `expand_topic(str) -> dict` 放链首，`invoke("话题")` 直接传字符串 | `expand_node \| prompt \| llm \| StrOutputParser()` |
| `RunnablePassthrough()` | 原样传递（RAG 里拼 context 用） | `W7D4`、`W7D5` |
| `RunnableWithMessageHistory` | 多轮记忆 | `AI_agent_teacher.py` |
| `MessagesPlaceholder("history")` | prompt 里给历史留位置 | `AI_agent_teacher.py` |

**记忆调用方式**
```python
chat_agent = RunnableWithMessageHistory(chain, get_session_history,
                 input_messages_key="input", history_messages_key="history")
chat_agent.invoke({"input": q}, config={"configurable": {"session_id": SID}})
```

### 3.4 流式 `stream()`

```python
for chunk in chain.stream({...}):
    print(chunk, end="", flush=True)      # 打字机效果
```

**⚠️ 坑**：stream 下 Lambda **每个 chunk 都跑一次**，加 emoji/签名这种一次性操作会重复加。
他自己在文件里注释过，所以最终只流式跑 `(prompt|llm|StrOutputParser()).stream(...)`。

### 3.5 结构化输出 `with_structured_output`

```python
class 报错诊断(BaseModel):
    错误类型: str = Field(description="报错类型，如ModuleNotFoundError")
    解决步骤: list[str] = Field(description="解决步骤列表")

chain = prompt | llm.with_structured_output(报错诊断, method="json_mode")
# 结果按属性取：result.错误类型 / result.解决步骤
```

**两种 method**：`json_mode` / `function_calling`。

**⚠️ 坑**：`json_mode` 要求 **prompt 里必须出现 "json" 字样**，否则报
`Prompt must contain the word 'json' ...`（他文件里第 31 行就是这个报错）。

### 3.6 `@tool` + `bind_tools`

```python
@tool
def calculate(表达式: str) -> str:
    """计算数学表达式，例如 '3*7+2' 返回 23"""     # ← docstring 就是给模型的说明书
    return str(eval(表达式))

llm_with_tools = llm.bind_tools([calculate])       # ← 这才真正把工具交给模型
result = llm_with_tools.invoke(messages)
if result.tool_calls:
    for tc in result.tool_calls:
        out = tool_map[tc["name"]].invoke(tc["args"])
        messages.append(ToolMessage(content=str(out), tool_call_id=tc["id"]))
```

**⚠️ 坑**：`@tool` 产出的是**工具对象**，直接丢进 `|` 管道是错的（`tool.invoke` 期望字典 args）。
要进管道必须用 `RunnableLambda(普通函数版)`。

---

## 第 4 层 · RAG 全链路（W7）

> 切分 → embedding → 入库 → 检索 → 拼 prompt → 生成

| 步骤 | 实现 | 关键参数 |
|---|---|---|
| **切分** | `MarkdownHeaderTextSplitter` + `RecursiveCharacterTextSplitter` | `chunk_size=300, chunk_overlap=40`，separators `["\n","。","；","，",""]` |
| 切分后清洗 | 滤太短（<20字）+ 滤水印噪声 | 得到 **27 个有效 chunk** |
| **embedding** | `OpenAIEmbeddings(model="BAAI/bge-m3")` ← 硅基流动 | **1024 维** |
| **入库** | `Chroma.from_documents(..., collection_metadata={"hnsw:space": "cosine"})` | `collection="langchain"`，`persist_directory="./kb_chroma_d5"` |
| **检索** | `vectorstore.as_retriever(search_kwargs={"k": 8})` | cosine 距离 |
| **拼 prompt** | `{"context": retriever \| format_docs, "question": RunnablePassthrough()}` | `format_docs` 拼上小节标题 |
| **生成** | `\| prompt \| llm \| StrOutputParser()` | DeepSeek |

**W7 有一条"手写版"验证链**（`W7D1_embdding_api.py`）—— 不看 LangChain，直接理解 embedding 本质：
4 条文档 → 返回 `np.array` **shape (4, 1024)** → 自己写余弦
`cosine(a,b) = a @ b / (np.linalg.norm(a) * np.linalg.norm(b))` → 排序。
**bge-m3 就是 1024 维**，这个数字要记住。

**D4 / D5 两个脚本的定位不同，别搞混**

| | D4（`W7D4_rag_chain.py`） | D5（`W7D5_kb.py`） |
|---|---|---|
| 用途 | 3 条玩具语料，**跑通链路 + 测防幻觉** | ⭐ **真实知识库**，切自己的笔记 |
| collection | `handbook_d4` | （D5 用 `from_documents` 建） |
| 库路径 | `./chroma_rag_d4` | `./kb_chroma_d5` |
| k | 2 | 8 |
| 相似度空间 | 未显式声明（默认 L2） | **显式 `{"hnsw:space":"cosine"}`** |
| 特色 | `ask()` 先打印"检索到的原文"再打印答案，**检索通路可视化** | 双层切分 + 水印过滤，27 个有效 chunk |

> ⚠️文件名坑已修（W01 · 2026-10-12）：D4 原文件名是 `W7D4_rag_chain.py：.py`（**全角冒号在 `.py` 之前**，Linux/macOS 上非法），已改名为 `W7D4_rag_chain.py`。

> ⚠️ 两个脚本都是 `if os.path.exists → shutil.rmtree` **全量重建**，不是增量入库。
> 反复跑会反复花 embedding 调用；要加文档得改成 append。

### 防幻觉机制（prompt 原话）

```
只能依据下面<笔记>里的内容回答。要求：
1. 笔记里有的，简洁准确回答，尽量用笔记中原话；
2. 笔记里没有的，直接说"我的笔记里没有相关内容"，严禁编造；
3. 不要提"笔记""检索"这类词，自然作答。
```

**实测有效**：问"电脑坏了怎么办"（原文没写维修），模型老实承认没相关内容。

### 为什么要二次切分
`MarkdownHeaderTextSplitter` 保住**章节结构**（metadata 带 `节`/`小节`），
`RecursiveCharacterTextSplitter` 防止单块**超长**。中文按**字符数**算，不按空格分词。

---

## 第 5 层 · 检索质量优化（W8）

### 5.1 BM25 关键词检索（`W8D1_BM25.py`）

**BM25 是什么**：一种**基于词频统计的经典检索算法**（Best Matching 25），
用"查询里的词在文档中出现得多少、这些词有多稀有"给文档打分排序。
中文场景必须先用 jieba 分词，再用 `rank_bm25` 的 `BM25Okapi` 建索引查询。

**它的本质是关键词字面匹配，不是语义理解** —— 所以问"电脑坏了"匹配不到"主机故障"，
但问"RTX 5060"能精确命中，这正是它和向量检索互补的地方。

$$\text{score}(q,d)=\sum_{t\in q}\text{IDF}(t)\cdot\frac{f(t,d)\cdot(k_1+1)}{f(t,d)+k_1(1-b+b\cdot|d|/\text{avgdl})}$$

| 组件 | 作用 |
|---|---|
| IDF | 词越稀有越值钱 → "的/了"权重≈0 |
| 词频 f | 出现越多分越高，但 **k1 饱和**，防刷词 |
| 长度归一化 b | 短文档命中更值钱，防长文档堆词取胜 |

默认 `k1=1.5, b=0.75`。中文**必须先 jieba 分词**，且**入库和查询用同一套分词规则**。

**核心局限：只认字面，不懂语义。"电脑坏了" 匹配不到 "主机故障"。**

### 5.2 混合检索 + RRF（`W8D2_hybrid_rrf.py`）

**问题**：两路**分数不可比**（余弦 0~1 vs BM25 0~几十），加权求和会被数值大的那路通吃。

**解法**：只融合**名次**。

$$\text{RRF}(d)=\sum_{r\in R}\frac{1}{k+\text{rank}_r(d)},\quad k=60$$

```python
for rl in result_lists:                            # 每一路
    for rank, (doc_id, doc) in enumerate(rl, 1):   # rank 从 1 开始！
        id2doc[doc_id] = doc
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank)   # ← 累加，不是赋值
```

**为什么比加权求和好**
- 名次天然归一化，**免疫分数尺度差异**
- 各路长度**可以不同**，不需要对齐/填充
- 想表达偏好就加**权重**：`scores[doc_id] += w / (k + rank)`

**k 的作用**：k 越小，名次差异被放得越大（k=1 → 第1名是第4名的 2.5 倍；k=1000 → 几乎无差别）。60 是论文推荐平衡点。

### 📊 实测结论（可写进简历）

同一份笔记语料、同一个问题「Function Calling 和 ReAct 有什么区别？」：

| | 向量检索（bge-m3） | BM25 |
|---|---|---|
| 第 1 名 | ❌ Day1 装包/环境配置 | ✅ ReAct 定义原文 |
| 第 2 名 | Day4 messages 循环 | ✅ **【关键对比】表格** |
| 第 3 名 | Day3 天气 404 | Day4 模板 |

> **向量检索一个相关块都没捞到，BM25 捞到了核心对比表。**
> 原因：专有名词被语义模型"泛化"成了"工具调用"这类概念。

**同时实测出 RRF 的局限**：用一个**语料里不存在的问题**去问时，两路**都没捞到相关内容**，
RRF 照样把 4 条噪声喂给 LLM —— **RRF 只做排序融合，不做相关性判断。**

> 这说明 RRF 只能合并排序，**无法判断"这些内容其实都不相关"**，
> 所以后面才必须补 rerank + 拒答机制（见 5.3）。

---

## 📖 精确参数与名词对照表

> 检索"具体名词"时命中率最高的地方。想不起来某个名字、某个参数的准确值，先查这里。

### LangChain 组件名词（别记混）

| 名字 | 一句话作用 | 不接它会怎样 |
|---|---|---|
| `ChatOpenAI` | 调对话模型（ChatModel） | — |
| `ChatPromptTemplate` | 提示词模板，`{变量}` 占位 | — |
| `StrOutputParser` | ✅ **把 `AIMessage` 转成 `str`** | 拿到 `AIMessage` 对象 |
| `JsonOutputParser` | ✅ **把模型输出解析成 `dict`** | 拿到 `AIMessage` 对象 |
| `RunnablePassthrough` | 把输入**原样透传**（RAG 里传 question） | prompt 缺 `{question}` |
| `RunnableLambda` | 把**普通函数**变成管道节点 | `@tool` 的工具对象不能进管道 |
| `RunnableWithMessageHistory` | 多轮对话记忆 | 每轮失忆 |
| `MessagesPlaceholder` | prompt 里给历史消息留位置 | 历史塞不进 prompt |
| `ToolMessage` | 工具执行结果回填，要带 `tool_call_id` | 第二轮 400 |

### 输出解析器（W5 学的三兄弟 + W6 的升级版）

| 解析器 | 输入 | 输出 | 场景 |
|---|---|---|---|
| `StrOutputParser` | `AIMessage` | `str` | 只要正文 |
| `JsonOutputParser` | `AIMessage` | `dict` | prompt 里写死 JSON 字段，然后 `data["answer"]` |
| `PydanticOutputParser` | `AIMessage` | Pydantic 对象 | 要类型校验 |
| `with_structured_output(模型类)` | `AIMessage` | Pydantic 对象 | ⭐ W6 的写法，最省事 |

**`with_structured_output` 的两种 method**：`method="json_mode"`（prompt 里必须出现 "json" 字样）、
`method="function_calling"`（走工具调用协议）。

### W7 RAG 精确参数速查

| 参数 | 值 |
|---|---|
| 切分器（一次） | `MarkdownHeaderTextSplitter(headers_to_split_on=[("#","篇"),("##","节"),("###","小节")])` |
| 切分器（二次） | `RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=40)` |
| separators | `["\n", "。", "；", "，", ""]`（D2 里用得更细，含 `\n\n`、`！`、`？`） |
| 中文长度 | `length_function=len` —— **按字符数，不按空格分词** |
| 清洗规则 | 滤 `len < 20` 的碎块 + 滤 `["Coze AI 生成","AIGC","人工智能生成","ContentProducer"]` |
| 有效 chunk | **27 个**（W7D5 实测） |
| embedding 模型 | `BAAI/bge-m3`，**1024 维**，走硅基流动 |
| 入库 | `Chroma.from_documents(..., collection_metadata={"hnsw:space": "cosine"})` |
| collection 名 | `langchain`（W8D2/D3 读的库）；W7D4 是 `handbook_d4` |
| 库路径 | `./kb_chroma_d5`（W7D5/W8D2/W8D3）；W7D4 是 `./chroma_rag_d4` |
| 检索 k | W7D5 是 **8**；W8D2 是 **4**；W8D3 召回 **15** 再 rerank 到 3 |
| 手写余弦 | `cosine(a,b) = a @ b / (np.linalg.norm(a) * np.linalg.norm(b))` |

### W8 混合检索精确参数速查

| 参数 | 值 |
|---|---|
| BM25 库 | `from rank_bm25 import BM25Okapi`，`BM25Okapi(tokenized_docs)` |
| 中文分词 | `jieba.cut(text)`，**库和查询必须同一套分词规则** |
| BM25 默认参数 | `k1=1.5`（词频饱和）、`b=0.75`（长度归一化） |
| RRF 公式 | `score = Σ 1 / (k + rank)`，**rank 从 1 开始**，必须**累加** |
| RRF 的 k | **60**（论文默认）。k 越小名次差异越被放大 |
| 加权 RRF | `score += w / (k + rank)`，`w` 表达"更信哪一路" |
| **rerank 模型** | `BAAI/bge-reranker-v2-m3`（硅基流动，Cross-Encoder） |
| **rerank 接口** | `POST https://api.siliconflow.cn/v1/rerank` |
| **rerank 请求体** | `{"model","query","documents":[文本...],"top_n","return_documents"}` |
| **rerank 响应** | `data["results"]` = `[{"index": i, "relevance_score": s}, ...]`，**已按分降序** |
| **index 的含义** | 是 `documents` 列表里的下标 → 用它索引回原候选列表 |
| **实测分数尺度** | 相关 ≈ **0.998**；弱相关 ≈ 0.015；无关 ≈ **0.00002 ~ 0.026** |
| **阈值警告** | ⚠️ 绝对阈值**不可移植**！不同模型尺度不同，必须自己标定 |

### rerank vs 向量检索（Bi-Encoder vs Cross-Encoder）

| | Bi-Encoder（bge-m3） | Cross-Encoder（bge-reranker） |
|---|---|---|
| 编码方式 | query / doc **分别**编码 | query + doc **拼在一起** |
| 看得到交互吗 | ❌ 编码 doc 时不知道 query | ✅ 每层都在做交叉注意力 |
| 向量能否预存 | ✅ 可以，所以能查百万级 | ❌ 不能，每次重算 |
| 速度 / 精度 | 快 / 粗 | 慢 / 准 |
| 在架构里的位置 | **召回**（粗筛） | **重排**（精判） |

> 一句话：**召回靠混合（向量+BM25），精度靠 rerank。**
> 两段式：先用 Bi-Encoder 从全库挑 15-20 条，再用 Cross-Encoder 精排出 3 条。

### 阈值拒答的三种做法（按可靠性排）

| 方法 | 做法 | 代价 |
|---|---|---|
| ① LLM 自己判断 ⭐ | 不设阈值，靠 prompt 让模型说"没有相关内容" | 多一次 LLM 调用 |
| ② 标定阈值 | 自造 50-100 条测试查询，看分数分布取交点 | 需人工标注 |
| ③ 相对阈值 | 看 top1 和 top2 的**差距**，不看绝对值 | 需调参 |

---

## 🎯 一句话自测（遮住右列）

| 问题 | 答案 |
|---|---|
| `prompt \| llm` 和加 `StrOutputParser()` 差在哪？ | 前者 `AIMessage`（要 `.content`），后者 `str` |
| 为什么一次 `invoke` 拿不到工具调用最终答案？ | 第一次只是"决定"调工具，要回填 `ToolMessage(tool_call_id=...)` 再调一次 |
| `tool_call_id` 漏了会怎样？ | 第二轮直接 400 |
| `tc["function"]["arguments"]` 是什么类型？ | **JSON 字符串**，要 `json.loads` |
| ReAct 和 Function Calling 最本质区别？ | FC 靠结构化协议（可靠），ReAct 靠文本 Prompt（灵活但靠模型听话） |
| ReAct 循环怎么防死循环？ | `for i in range(10)` 硬熔断 |
| 普通函数怎么进 LCEL 管道？ | `RunnableLambda(fn)` 包一层 |
| stream + Lambda 有什么坑？ | Lambda 每个 chunk 都跑，一次性后处理会重复加 |
| `json_mode` 的 prompt 必须含什么？ | 单词 "json" |
| RAG 为什么要切两次？ | 第一次保章节结构，第二次防单块超长 |
| chunk_size/overlap 是多少？ | 300 / 40，共 27 个有效 chunk |
| 防幻觉靠什么？ | prompt 里写死"没有就说没有，严禁编造" |
| BM25 的 k1 和 b 管什么？ | k1 词频饱和防刷词；b 长度归一化 |
| RRF 为什么用名次不用分数？ | 两路分数尺度不可比，加权求和会被大数值那路通吃 |
| RRF 的 k 大了会怎样？ | 名次差异被抹平，趋近"谁上榜多谁赢" |
| RRF 的局限？ | 只排序不判相关性，两路都没捞到的找不回来 → 需要 rerank |

---

## 🔜 下一站

**W8D3：rerank 重排** —— 解决 RRF「噪声照喂」的问题。
链路目标：`双路召回 → RRF → rerank → prompt → LLM`

之后：W9-10 LangGraph → W11-12 FastAPI 部署。

---

## 附录 · 关于早期脚本里的测试查询

W7/W8 的练习脚本里有几个**随手写的测试查询**，它们不是真实技术概念，仅作实验用途：

| 测试查询 | 性质 | 正确行为 |
|---|---|---|
| 一个无意义的占位问法（早期脚本里叫「WC所有语法？」） | 语料里完全不存在 | **拒答**（用于验证防幻觉） |
| "今天天气怎么样？" | 语料里没有实时数据 | **拒答** |
| "怎么做红烧肉？" | 完全无关领域 | **拒答** |

> **为什么要专门记这一条**：语料里如果反复"提及"某个词，
> 检索器会把"讨论过这个词"误判成"解释过这个词"，
> 于是本该拒答的问题反而拿到高分 —— 这叫**自指污染**。
> 处理方式：把这类"元讨论"从主文移到附录，降低它的检索权重。
