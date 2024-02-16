import chromadb
import logging
import nltk
import numpy as np
import param
import string
import time
from chromadb import Settings
from embedding import create_embedding, load_embedding_model
from langchain.text_splitter import RecursiveCharacterTextSplitter
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.tokenize import word_tokenize
from saveload import load_all_settings, load_txt, path_combine
from util import ls_mpt, unique_uuid


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# 獲取文本關鍵詞
def get_keywords(text, top_n=10):
    try:
        words = word_tokenize(text)
        stop_words = set(stopwords.words("english")) | set([
            "the", "a", "an", "and", "or", "but", "if", "it", "he", "she", "they",
            "of", "in", "on", "at", "for", "with", "as", "to", "this", "that", "these", "those",
            "be", "is", "are", "was", "were", "am", "been", "being",
            "have", "has", "had", "having",
            "do", "does", "did", "doing", "done",
            "can", "could", "will", "would", "shall", "should", "may", "might", "must",
            ])
        words_filtered = [word.lower() for word in words if word.lower() not in stop_words and word.lower() not in string.punctuation]
        keywords = FreqDist(words_filtered).most_common(top_n)
        return [keyword[0] for keyword in keywords]
    except Exception as e:
        logging.error(f"get_keywords() Err: {e}")
        return []


# 上下文分割器
def context_splitter(context, kwargs=None):
    try:
        kwargs = kwargs or param.saveload_config
        splitter = RecursiveCharacterTextSplitter(
            separators=kwargs["separators"],
            chunk_size=kwargs["chunk_size"],
            chunk_overlap=kwargs["chunk_overlap"],
            length_function=len,
            is_separator_regex=kwargs["is_separator_regex"],
            )
        return [split.page_content for split in splitter.create_documents([context])]
    except Exception as e:
        logging.error(f"context_splitter() ERR: {e}")
        return []


# 重置 (query) 數據庫 Collection
def reset_query_collection():
    try:
        param.query_client.reset()
        param.query_2nd_collection = None
        param.query_1st_collection = None
        param.query_client = None
    except Exception as e:
        logging.error(f"reset_query_collection Err: {e}")


# 創建與加載 (query) 數據庫 Collection
def create_load_query_collection(kwargs=None):
    try:
        kwargs = kwargs or param.database_config
        #nltk.download("stopwords", quiet=True)
        #nltk.download("punkt", quiet=True)
        query_database_path = path_combine(kwargs["database_folder_path"], kwargs["query_database_name"])
        param.query_client = chromadb.PersistentClient(path=query_database_path, settings=Settings(allow_reset=True))
        param.query_1st_collection = param.query_client.get_or_create_collection(name=kwargs["query_1st_collection_name"], metadata={"hnsw:space": kwargs["query_distance_fn"]})
        param.query_2nd_collection = param.query_client.get_or_create_collection(name=kwargs["query_2nd_collection_name"], metadata={"hnsw:space": kwargs["query_distance_fn"]})
    except Exception as e:
        logging.error(f"create_load_query_collection() ERR: {e}")


# 將數據添加到 (query) 數據庫 Collection 中
def query_collection_add(sl_kwargs=None, db_kwargs=None):
    # 將數據進行 embedding
    def doc_embed(uuid_list, ids_list, doc_list, embed_list, meta_list, ids_2nd, embedding_2nd, doc):
        try:
            ids_list.append(unique_uuid(uuid_list))
            doc_list.append(doc)
            embed_list.extend(create_embedding(doc).tolist())
            meta_list.append({"ids_2nd": ids_2nd, "embedding_2nd": embedding_2nd})
        except Exception as e:
            logging.error(f"doc_embed() ERR: {e}")

    try:
        sl_kwargs = sl_kwargs or param.saveload_config
        db_kwargs = db_kwargs or param.database_config
        data_path = path_combine(sl_kwargs["data_folder_path"], sl_kwargs["data_name"])
        contexts = load_txt(data_path).split("\n" * 5)
        ids_1, doc_1, embed_1, meta_1 = [], [], [], []
        ids_2, doc_2, embed_2, meta_2 = [], [], [], []
        uuid_list = []
        for context in contexts:
            context_sep = context.split("\n")
            #questions = context_sep[0:0]
            title = context_sep[0]
            content = context_sep[1:]
            title_content = title + "\n" + "\n".join(content)
            ids_2nd = unique_uuid(uuid_list)       
            content_embedding = create_embedding(title_content).tolist()
            embedding_2nd = ",".join(map(str, content_embedding[0]))
            ##### Questions #####
            #for question in questions:
            #    doc_embed(uuid_list, ids_1, doc_1, embed_1, meta_1, ids_2nd, embedding_2nd, question)
            ##### Title #####
            doc_embed(uuid_list, ids_1, doc_1, embed_1, meta_1, ids_2nd, embedding_2nd, title)     
            ##### Chunk by line #####
            for chunk in content:
                doc_embed(uuid_list, ids_1, doc_1, embed_1, meta_1, ids_2nd, embedding_2nd, chunk)
            ##### Chunk by line contains title #####
                title_chunk = title + "\n" + chunk
                doc_embed(uuid_list, ids_1, doc_1, embed_1, meta_1, ids_2nd, embedding_2nd, title_chunk)
            ##### Content contains title or content contains title keywords #####
            ids = unique_uuid(uuid_list)
            ids_1.append(ids)
            doc_1.append(title_content)
            #content_keywords = "\n".join(keyword for keyword in get_keywords(title_content, db_kwargs["nltk_keyword_top_n"]))
            #embed_1.extend(create_embedding(content_keywords).tolist())
            embed_1.extend(create_embedding(title_content).tolist())
            meta_1.append({"ids_2nd": ids_2nd, "embedding_2nd": embedding_2nd})
            ##### Parents #####
            ids_2.append(ids)
            doc_2.append(title_content)
            embed_2.extend(content_embedding)
            meta_2.append({"info": "this is parent"})
        param.query_1st_collection.add(ids=ls_mpt(ids_1), documents=ls_mpt(doc_1), embeddings=ls_mpt(embed_1), metadatas=ls_mpt(meta_1))
        param.query_2nd_collection.add(ids=ls_mpt(ids_2), documents=ls_mpt(doc_2), embeddings=ls_mpt(embed_2), metadatas=ls_mpt(meta_2))
        logging.info("Query Collection Data Create and Add Success")
    except Exception as e:
        logging.error(f"query_collection_add() ERR: {e}")


# 根據 (query) 數據庫 Collection 進行索引返回最相關的文檔
def index_query_collection(query, kwargs=None):
    try:
        kwargs = kwargs or param.database_config
        query_embedding = create_embedding(query).tolist()
        result_1st = param.query_1st_collection.query(query_embeddings=query_embedding, n_results=kwargs["n_result_query_1st"])
        for itm, distance in zip(result_1st["metadatas"][0], result_1st["distances"][0]):
            itm["distances"] = distance
        min_dist_dict = {}
        for itm in result_1st["metadatas"][0]:
            ids = itm["ids_2nd"]
            dist = itm["distances"]
            if ids not in min_dist_dict or dist < min_dist_dict[ids]["distances"]:
                min_dist_dict[ids] = itm
        ret_rm_dupl = list(min_dist_dict.values())
        dists = [itm["distances"] for itm in ret_rm_dupl]
        dist_thresh = np.mean(dists) - (kwargs["rm_dupl_dist_thresh_std_ratio"] * np.std(dists))
        ret_thresh = [itm for itm in ret_rm_dupl if itm["distances"] <= dist_thresh]
        ret_top_n = sorted(ret_thresh, key=lambda x: x["distances"])[:kwargs["rm_dupl_result_1st_top_n"]] 
        result_2nd = [param.query_2nd_collection.query(query_embeddings=[[float(ef.strip()) for ef in itm["embedding_2nd"].split(",")]], n_results=kwargs["n_result_query_2nd"]) for itm in ret_top_n]  
        return result_1st, result_2nd
    except Exception as e:
        logging.error(f"index_query_collection() ERR: {e}")
        return None, None


# 測試創建與加載 (query) 數據庫 Collection
def test_create_query_collection():
    load_all_settings()
    load_embedding_model()
    create_load_query_collection()
    query_collection_add()


# 測試對 (query) 數據庫 Collection 進行索引
def test_index_query_collection():
    load_all_settings()
    load_embedding_model()
    create_load_query_collection()
    start_time = time.time()
    _, result_2nd = index_query_collection("The query")
    end_time = time.time()
    execution_time = end_time - start_time
    logging.info(f"Time Database Index (Sec):\n{execution_time}\n")
    for itm in result_2nd:
        logging.info(itm["documents"][0][0] + "\n\n")