import os, shutil
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

# ===== 1. 两个模型，各走各家（和前几天一致）=====
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

# ===== 2. 建库（自包含，每次重建，逻辑同 D3）=====
chunks = [
    "员工请假需提前在OA系统提交申请，经直属主管审批后方可休假。事假为无薪假期，每次事假原则上不超过三天。员工连续工作满一年享有5天带薪年假，满十年享有10天年假。",
    "差旅费需在出差结束后7个工作日内提交报销单，逾期不予受理。市内出租车票需在发票背面注明起止地点和事由。所有报销须附正规发票，收据和截图无效。",
    "公司为每位新员工配备笔记本电脑一台，入职当天到IT部领取。员工离职时须将电脑、电源和门禁卡一并归还，如有遗失或人为损坏需照价赔偿。",
]
metadatas = [{"section": "请假制度"}, {"section": "费用报销"}, {"section": "办公设备"}]

path = "./chroma_rag_d4"
if os.path.exists(path):
    shutil.rmtree(path)

vectorstore = Chroma.from_texts(
    texts=chunks, embedding=embeddings, metadatas=metadatas,
    collection_name="handbook_d4", persist_directory=path,
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# ===== 3. 把检索到的 Document 拼成纯文本 =====
def format_docs(docs):
    return "\n\n".join(
        f"【{d.metadata.get('section', '')}】{d.page_content}" for d in docs
    )

# ===== 4. Prompt：压幻觉的关键约束（呼应 D3 第6题）=====
prompt = ChatPromptTemplate.from_template("""你是员工手册问答助手。只能依据下面<资料>里的内容回答。
要求：
1. 资料里有的，简洁准确回答；
2. 资料里没有的信息，直接说"员工手册中没有相关规定，建议咨询HR或对应部门"，严禁编造；
3. 自然作答，不要提"资料""检索"这类词。

<资料>
{context}
</资料>

问题：{question}""")

# ===== 5. 完整 RAG 链 =====
chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# ===== 6. 先看检索通路，再看最终答案 =====
def ask(q):
    print("=" * 50)
    print("问题：", q)
    docs = retriever.invoke(q)          # 单独看检索捞到了啥
    print("— 检索到的原文 —")
    print(format_docs(docs))
    print("— DeepSeek 的回答 —")
    print(chain.invoke(q))
    print()

ask("年假有几天？")
ask("入职能领电脑吗？")
ask("电脑坏了怎么办？")      # 压轴：原文没写维修，看它会不会老实承认