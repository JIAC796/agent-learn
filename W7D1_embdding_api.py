import os
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# embedding 走硅基流动，对话以后才走 DeepSeek——两家是不同模型
client = OpenAI(
    api_key=os.getenv("SILICONFLOW_API_KEY"),
    base_url="https://api.siliconflow.cn/v1",
)
EMBED_MODEL = "BAAI/bge-m3"


def embed(texts):
    """把一批句子变成向量，返回 shape=(句子数, 1024) 的数组"""
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return np.array([d.embedding for d in resp.data])


def cosine(a, b):
    return a @ b / (np.linalg.norm(a) * np.linalg.norm(b))


if __name__ == "__main__":
    # 知识库里的 4 段话
    docs = [
        "公司每月十号发放工资，遇节假日提前",
        "员工的薪资由基本工资和绩效构成",
        "今天外面阳光很好，适合出去散步",
        "LangChain 中文档切分推荐用 RecursiveCharacterTextSplitter",
    ]
    doc_vecs = embed(docs)
    print("向量形状:", doc_vecs.shape)   # 期望 (4, 1024)

    # 用户提问
    query = "什么时候发薪水？"
    q_vec = embed([query])[0]

    # 算问题和每段话的相似度，按从高到低排
    ranked = sorted(
        ((cosine(q_vec, v), d) for v, d in zip(doc_vecs, docs)),
        key=lambda x: -x[0],
    )
    print(f"\n问题：{query}")
    for score, d in ranked:
        print(f"{score:.3f}  {d}")