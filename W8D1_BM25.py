import jieba
from rank_bm25 import BM25Okapi

# 1. 造 4 条小语料（先别接你 W7 的知识库，先用这个看效果）
docs = [
    "RTX 5060 显卡 8G 显存 适合 1080P 游戏",
    "AMD 9600X 处理器 六核 十二线程 低功耗",
    "DDR5 内存 32G 套装 价格 回落到 650 元",
    "电脑 坏了 需要 联系 售后 维修",
]

# 2. 每句用 jieba 分词，变成 token 列表
tokenized_docs = [list(jieba.cut(doc)) for doc in docs]
print("分词结果示例：", tokenized_docs[0])

# 3. 建 BM25
bm25 = BM25Okapi(tokenized_docs)

# 4. 查询（同样要分词）
query = list(jieba.cut("RTX 5060 多少钱"))

# 5. 打分，按分数从高到低排出前 3
scores = bm25.get_scores(query)
ranked = sorted(enumerate(scores), key=lambda x: -x[1])[:3]

print("\n查询：RTX 5060 多少钱")
for idx, s in ranked:
    print(f"  分数 {s:.4f}  →  {docs[idx]}")
    # ===== 实验二：换个口语化的说法 =====
    print("\n" + "=" * 40)
    print("换口语查询：主机出故障了该找谁处理")
    query2 = list(jieba.cut("主机出故障了该找谁处理"))
    scores2 = bm25.get_scores(query2)
    ranked2 = sorted(enumerate(scores2), key=lambda x: -x[1])[:3]
    for idx, s in ranked2:
        print(f"  分数 {s:.4f}  →  {docs[idx]}")
        print("查询分词：", query2)
        for i, d in enumerate(tokenized_docs):
            print(f"文档{i} 分词：", d)