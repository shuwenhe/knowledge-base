# data.py - 用于测试 FAISS 索引是否正常加载

import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings

FAISS_INDEX_PATH = "./faiss_index"  # 或 "faiss_index"，根据你的路径

embeddings = OllamaEmbeddings(
    model="bge-m3",  # 确保和你构建时用的模型一致
    base_url="http://localhost:11434"
)

# 加载本地 FAISS 索引
if not os.path.exists(FAISS_INDEX_PATH):
    print(f"❌ 未找到 FAISS 索引目录: {FAISS_INDEX_PATH}")
    print("请先运行 build_knowledge_base.py 构建知识库")
else:
    try:
        db = FAISS.load_local(
            folder_path=FAISS_INDEX_PATH,
            embeddings=embeddings,
            allow_dangerous_deserialization=True  # 必须加这个参数
        )
        print(f"✅ FAISS 索引加载成功！")
        print(f"   共包含 {db.index.ntotal} 条向量文档")
        
        # 可选：做一次相似度搜索测试
        query = "测试问题"  # 你可以改成知识库里有的内容
        results = db.similarity_search(query, k=3)
        print(f"\n🔍 对查询 '{query}' 的前3个最相似结果：")
        for i, doc in enumerate(results):
            print(f"{i+1}. {doc.page_content[:200]}...")  # 只显示前200字符
        
    except Exception as e:
        print(f"❌ 加载失败: {e}")
