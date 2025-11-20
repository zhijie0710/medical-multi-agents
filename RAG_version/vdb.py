from huggingface_hub import InferenceClient
import faiss
import numpy as np
from medical_docs import medical_docs
from dotenv import load_dotenv
from dotenv import load_dotenv
import os
import faiss
import pickle


load_dotenv("/Users/zhijietang/Desktop/medical/hf_1.env", override=True)  # 确保文件名正确
hf_token = os.environ.get("HF_TOKEN")
assert hf_token is not None, "HF_TOKEN not loaded!"


client = InferenceClient(token=hf_token)

emb = client.feature_extraction(medical_docs, model="BAAI/bge-small-en")
emb = np.array(emb) # mean pooling


# emb: numpy array, shape (num_docs, embedding_dim)
embedding_dim = emb.shape[1]

# 创建 FAISS 索引
index = faiss.IndexFlatL2(embedding_dim)
index.add(emb)  # 添加所有文档向量

# 保存索引到磁盘
faiss.write_index(index, "medical_docs.index")

# 保存原始文档内容（对应向量）
with open("medical_docs.pkl", "wb") as f:
    pickle.dump(medical_docs, f)