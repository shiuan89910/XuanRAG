import chromadb
import logging
import param
from chromadb import Settings
from embedding import load_embedding_model
from saveload import load_all_settings, path_combine
from util import unique_uuid


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# 創建與加載數據庫 Collection
def create_load_collection(database_type, db_kwargs=None, st_kwargs=None):
    try:
        db_kwargs = db_kwargs or param.database_config
        st_kwargs = st_kwargs or param.history_config
        database_name = st_kwargs[f"{database_type}_database_name"]
        collection_name = st_kwargs[f"{database_type}_collection_name"]
        database_path = path_combine(db_kwargs["database_folder_path"], database_name)
        client = chromadb.PersistentClient(path=database_path, settings=Settings(allow_reset=True))
        collection = client.get_or_create_collection(name=collection_name, metadata={"hnsw:space": db_kwargs["query_distance_fn"]})
        return client, collection
    except Exception as e:
        logging.error(f"create_load_collection() ERR: {e}")
        return None, None


# 重置 (question) 數據庫 Collection
def reset_question_collection():
    try:
        param.question_client.reset()
        param.question_collection = None
        param.question_client = None
    except Exception as e:
        logging.error(f"reset_question_collection() ERR: {e}")


# 創建與加載 (question) 數據庫 Collection
def create_load_question_collection(db_kwargs=None, st_kwargs=None):
    try:
        param.question_client, param.question_collection = create_load_collection("question", db_kwargs, st_kwargs)
    except Exception as e:
        logging.error(f"create_load_question_collection() ERR: {e}")


# 獲取 (question) 數據庫 Collection 中的數據
def question_collection_get():
    try:
        q_list, r_list, well_list, worse_list, fb_list = [], [], [], [], []
        ret = param.question_collection.get()
        for i in range(len(ret["ids"])):
            q_list.append(ret["documents"][i])
            r_list.append(ret["metadatas"][i]["response"])
            well_list.append(ret["metadatas"][i]["well_rate"])
            worse_list.append(ret["metadatas"][i]["worse_rate"])
            fb_list.append(ret["metadatas"][i]["feedback"])
        return q_list, r_list, well_list, worse_list, fb_list
    except Exception as e:
        logging.error(f"question_collection_get() ERR: {e}")
        return [], [], [], [], []


# 將數據添加到 (question) 數據庫 Collection 中
def question_collection_add(sav_dict):
    try:
        param.question_collection.add(
            ids=[unique_uuid(param.question_collection.get()["ids"]) for _ in sav_dict["ids"]],
            documents=[document for document in sav_dict["documents"]], 
            metadatas=[metadata for metadata in sav_dict["metadatas"]],
            embeddings=[[] for _ in sav_dict["ids"]],
            )
    except Exception as e:
        logging.error(f"question_collection_add() ERR: {e}")


# 重置 (history) 數據庫 Collection
def reset_history_collection(kwargs=None):
    try:
        kwargs = kwargs or param.database_config
        sav_dict = {"ids": [], "documents": [], "metadatas": [], "embeddings": None}
        ret = param.history_collection.get()
        for ids in ret["ids"]:
            param.history_subcollection = param.history_client.get_or_create_collection(name=ids, metadata={"hnsw:space": kwargs["query_distance_fn"]})
            dict_get = param.history_subcollection.get()
            sav_dict["ids"].extend(dict_get["ids"])
            sav_dict["documents"].extend(dict_get["documents"])
            sav_dict["metadatas"].extend(dict_get["metadatas"])
        param.history_client.reset()
        param.history_subcollection = None
        param.history_collection = None
        param.history_client = None
        return sav_dict
    except Exception as e:
        logging.error(f"reset_history_collection() ERR: {e}")
        return None


# 創建與加載 (history) 數據庫 Collection
def create_load_history_collection(db_kwargs=None, st_kwargs=None):
    try:
        param.history_client, param.history_collection = create_load_collection("history", db_kwargs, st_kwargs)
    except Exception as e:
        logging.error(f"create_load_history_collection() ERR: {e}")


# 獲取 (history) 數據庫 Collection 中的數據
def history_collection_get():
    try:
        ret = param.history_collection.get()
        ids_list, qt_list = [], []
        for i in range(len(ret["ids"])):
            ids_list.append(ret["ids"][i])
            qt_list.append(ret["documents"][i])
        return ids_list, qt_list
    except Exception as e:
        logging.error(f"history_collection_get() ERR: {e}")
        return [], []


# 將數據添加到 (history) 數據庫 Collection 中
def history_collection_add(question_title, kwargs=None):
    try:
        kwargs = kwargs or param.database_config
        h_ids = unique_uuid(param.history_collection.get()["ids"])
        param.history_collection.add(ids=[h_ids], documents=[question_title], embeddings=[[]])
        param.history_subcollection = param.history_client.get_or_create_collection(name=h_ids, metadata={"hnsw:space": kwargs["query_distance_fn"]})
        return h_ids
    except Exception as e:
        logging.error(f"history_collection_add() ERR: {e}")
        return None


# 更新 (history) 數據庫 Collection 中的數據
def history_collection_update(ids, question_title=""):
    try:
        param.history_collection.get(ids=[ids])
        param.history_collection.update(ids=[ids], documents=[question_title], embeddings=[[]])
    except Exception as e:
        logging.error(f"history_collection_update() ERR: {e}")


# 刪除 (history) 數據庫 Collection 中的數據
def history_collection_delete(h_ids, kwargs=None):
    try:
        kwargs = kwargs or param.database_config
        param.history_subcollection = param.history_client.get_or_create_collection(name=h_ids, metadata={"hnsw:space": kwargs["query_distance_fn"]})
        sav_dict = param.history_subcollection.get()
        param.history_collection.delete(ids=[h_ids])
        param.history_client.delete_collection(name=h_ids)
        param.history_subcollection = None
        return sav_dict
    except Exception as e:
        logging.error(f"history_collection_delete() ERR: {e}")
        return None


# 獲取 (history) 子數據庫 Collection 中的數據
def history_subcollection_get(h_ids, kwargs=None):
    try:
        kwargs = kwargs or param.database_config
        param.history_subcollection = param.history_client.get_or_create_collection(name=h_ids, metadata={"hnsw:space": kwargs["query_distance_fn"]})
        q_list, r_list, well_list, worse_list, fb_list = [], [], [], [], []
        ret = param.history_subcollection.get()
        for i in range(len(ret["ids"])):
            q_list.append(ret["documents"][i])
            r_list.append(ret["metadatas"][i]["response"])
            well_list.append(ret["metadatas"][i]["well_rate"])
            worse_list.append(ret["metadatas"][i]["worse_rate"])
            fb_list.append(ret["metadatas"][i]["feedback"])
        return q_list, r_list, well_list, worse_list, fb_list
    except Exception as e:
        logging.error(f"history_subcollection_get() ERR: {e}")
        return [], [], [], [], []


# 將數據添加到 (history) 子數據庫 Collection 中
def history_subcollection_add(question, response):
    try:
        ids = unique_uuid(param.history_subcollection.get()["ids"])
        param.history_subcollection.add(
            ids=[ids], 
            documents=[question], 
            metadatas=[{"response": response, "well_rate": False, "worse_rate": False, "feedback": ""}],
            embeddings=[[]],
            )
        return ids, question, response
    except Exception as e:
        logging.error(f"history_subcollection_add() ERR: {e}")
        return None, None, None
   

# 更新 (history) 子數據庫 Collection 中的數據
def history_subcollection_update(ids, question="", response="", well_rate=False, worse_rate=False, feedback=""):
    try:
        sav_dict = param.history_subcollection.get(ids=[ids])
        param.history_subcollection.update(
            ids=[ids],
            documents=[question], 
            metadatas=[{"response": response, "well_rate": well_rate, "worse_rate": worse_rate, "feedback": feedback}],
            embeddings=[[]],
            )
        return sav_dict
    except Exception as e:
        logging.error(f"history_subcollection_update() ERR: {e}")
        return None


# 測試創建與加載 (question) 與 (history) 數據庫 Collection
def test_create_question_history_collection():
    load_all_settings()
    load_embedding_model()
    create_load_question_collection()
    create_load_history_collection()
    sav_dict = {"ids":[unique_uuid([])], "documents":["q_question"], "embeddings":[], "metadatas":[{"response": "q_response", "well_rate": False, "worse_rate": False, "feedback":"q_feedback"}]}
    question_collection_add(sav_dict)
    uuid = history_collection_add("title")
    history_subcollection_get(uuid)
    sub_uuid, _, _ = history_subcollection_add("h_question", "h_response")
    return uuid, sub_uuid


# 測試對 (question) 與 (history) 數據庫 Collection 進行操作
def test_operate_question_history_collection(uuid, sub_uuid):
    load_all_settings()
    load_embedding_model()
    q_list, r_list, well_list, worse_list, fb_list = question_collection_get()
    logging.info("documents: " + q_list[0] + ", response: " + r_list[0] + ", well_rate: " + str(well_list[0]) + ", worse_rate: " + str(worse_list[0]) + ", feedback: " + fb_list[0] + "\n\n")
    ids_list, qt_list = history_collection_get()
    logging.info("ids: " + ids_list[0] + ", title: " + qt_list[0] + "\n\n")
    history_collection_update(uuid, "new_title")
    ids_list, qt_list = history_collection_get()
    logging.info("ids: " + ids_list[0] + ", title: " + qt_list[0] + "\n\n")
    q_list, r_list, well_list, worse_list, fb_list = history_subcollection_get(uuid)
    logging.info("documents: " + q_list[0] + ", response: " + r_list[0] + ", well_rate: " + str(well_list[0]) + ", worse_rate: " + str(worse_list[0]) + ", feedback: " + fb_list[0] + "\n\n")
    history_subcollection_update(sub_uuid, question="new_h_question", response="new_h_response", well_rate=True, worse_rate=True, feedback="new_h_feedback")
    q_list, r_list, well_list, worse_list, fb_list = history_subcollection_get(uuid)
    logging.info("documents: " + q_list[0] + ", response: " + r_list[0] + ", well_rate: " + str(well_list[0]) + ", worse_rate: " + str(worse_list[0]) + ", feedback: " + fb_list[0] + "\n\n")
    sav_dict = history_collection_delete(uuid)
    logging.info(sav_dict)