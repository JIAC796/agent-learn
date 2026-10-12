import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()
emb = OpenAIEmbeddings(model="BAAI/bge-m3", api_key=os.getenv("SILICONFLOW_API_KEY"),
                       base_url="https://api.siliconflow.cn/v1")
for p in ["./kb_chroma_d5", "./kb_chroma_v2"]:
    if os.path.exists(p):
        vs = Chroma(collection_name="langchain", embedding_function=emb, persist_directory=p)
        n = len(vs.get(include=["documents"])["ids"])
        print(f"{p}: {n} chunk")
    else:
        print(f"{p}: 不存在")
