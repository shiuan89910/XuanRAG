import gc
import logging
import param
import time
import torch
import torch.nn.functional as F
from saveload import load_all_settings, path_combine
from sentence_transformers.util import semantic_search
from transformers import AutoModel, AutoTokenizer


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# 釋放 Embedding 模型記憶體資源
def unload_embedding_model():
    try:
        param.embedding_model = param.embedding_tokenizer = None
        gc.collect()
        torch.cuda.empty_cache()
    except Exception as e:
        logging.error(f"unload_embedding_model() ERR: {e}")


# 加載 Embedding 模型
def load_embedding_model(kwargs=None):
    try:
        kwargs = kwargs or param.embedding_config
        embedding_model_path = path_combine(kwargs["embedding_model_folder_path"], kwargs["embedding_model_name"])
        param.embedding_tokenizer, param.embedding_model = AutoTokenizer.from_pretrained(embedding_model_path), AutoModel.from_pretrained(embedding_model_path)
        logging.info("Load Embedding Model Success")
    except Exception as e:
        logging.error(f"load_embedding_model() ERR: {e}")


# 將文本創建 Embedding 向量
def create_embedding(context):
    try:
        encode_in = param.embedding_tokenizer(context, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            model_out = param.embedding_model(**encode_in)
        token_embedding = model_out[0]
        msk_expand = encode_in["attention_mask"].unsqueeze(-1).expand(token_embedding.size()).float()
        embedding = torch.sum(token_embedding * msk_expand, 1) / torch.clamp(msk_expand.sum(1), min=1e-9)
        return F.normalize(embedding, p=2, dim=1)
    except Exception as e:
        logging.error(f"create_embedding() ERR: {e}")
        return None


# 執行語義搜尋來找到查詢和響應之間的相似度
def similar_search(query, response):
    try:
        query_embedding = torch.FloatTensor(create_embedding(query)).to("cuda")
        response_embedding = torch.FloatTensor(create_embedding(response)).to("cuda")
        return semantic_search(query_embedding, response_embedding)
    except Exception as e:
        logging.error(f"similar_search() ERR: {e}")
        return None


# 獲取序列的 Embedding 向量長度
def embedding_token_num(sequence):
    try:
        return len(param.embedding_tokenizer.encode(str(sequence), return_tensors="pt").cuda()[0])
    except Exception as e:
        logging.error(f"embedding_token_num() ERR: {e}")
        return 0


#  測試 Embedding 與語義搜尋
def test_embedding():
    load_all_settings()
    load_embedding_model()
    start_time = time.time()
    similar_embedding = similar_search("The sentence 1", "The sentence 2")  
    end_time = time.time()
    execution_time = end_time - start_time
    logging.info(f"Time Similar Search (Sec):\n{execution_time}\n")
    logging.info(similar_embedding)