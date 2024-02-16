import json
import logging
import param
from collections import deque
from database import create_load_query_collection, index_query_collection, query_collection_add, reset_query_collection
from embedding import embedding_token_num, load_embedding_model, similar_search, unload_embedding_model
from flask import Flask, jsonify, redirect, render_template, request, Response, stream_with_context
from gpt import gptc_load_gpt_model, gptgtpq_load_gpt_model
from history import(
     create_load_question_collection, 
     create_load_history_collection,
     history_collection_add, 
     history_collection_delete, 
     history_collection_get, 
     history_collection_update,
     history_subcollection_add, 
     history_subcollection_get, 
     history_subcollection_update,
     question_collection_add
)
from llama import exllama_load_gpt_model, exllama2_load_gpt_model, llamacpp_load_gpt_model
from saveload import load_all_settings, path_combine, save_json
from util import clear_cache, en_to_lang, get_prompt_template, gpt_inference, lang_to_en, rm_cr, unique_uuid, unload_gpt_model


index_save = []
table_data = deque()
selected_data = {"uuid": None, "selected": False}
uuid_now, question_now, response_now = None, None, None


app = Flask(__name__)


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# 加載歷史資料
def load_history_data():
    global table_data
    try:
        uuids, names = history_collection_get()
        table_data.extendleft({"name": name, "selected": False, "uuid": uuid} for uuid, name in zip(uuids, names))
    except Exception as e:
        logging.error(f"load_history_data() ERR: {e}")


# 加載頁面
@app.route("/")
def webui():
    try:
        return render_template("webui.html", param=param, table_data=table_data)
    except Exception as e:
        logging.error(f"webui() ERR: {e}")
        return jsonify({"error": "webui() ERR"}), 500


# 回答問題的路由
@app.route("/answerthequestion", methods=["POST"])
def answerthequestion():
    try:
        clear_cache()
        param.instruction = param.prompt = param.completion = param.context = param.prompt_all = ""
        param.prompt = lang_to_en(rm_cr(request.form["query"]))
        #logging.info(param.prompt + "\n\n")
        if (param.chat_mode or param.webui_config["chat_mode"]) == "qa":
            _, result_2nd = index_query_collection(param.prompt)
            param.context = "\n\n".join([itm["documents"][0][0] for itm in result_2nd])
        q_list, r_list = None, None
        if selected_data["uuid"] and param.prompt_config["is_conversation"]:
            q_list, r_list, _, _, _ = history_subcollection_get(selected_data["uuid"])
        get_prompt_template(q_list, r_list)
        logging.info(param.prompt_all + "\n\n")
        return Response(stream_with_context(gpt_inference(param.prompt_all, param.gpt_inference_params)), content_type="text/plain")
    except Exception as e:
        logging.error(f"answerthequestion() ERR: {e}")
        return jsonify({"error": "answerthequestion() ERR"}), 500


# 儲存問答歷史的路由
@app.route("/historythequestionanswer", methods=["POST"])
def historythequestionanswer():
    global uuid_now, question_now, response_now
    try:
        rsponse = json.loads(request.data).get("text")
        if selected_data["selected"]:
            reanswerchecked = request.args.get("reanswerchecked") == "true"
            if reanswerchecked and uuid_now is not None:
                sav_dict = history_subcollection_update(uuid_now, question=param.prompt, response=rsponse)
                question_collection_add(sav_dict)
            else:
                history_subcollection_get(selected_data["uuid"])
                uuid_now, question_now, response_now = history_subcollection_add(param.prompt, rsponse)
        else:
            sav_dict = {"ids":[unique_uuid([])], "documents":[param.prompt], "embeddings":[], "metadatas":[{"response": rsponse, "well_rate": False, "worse_rate": False, "feedback":""}]}
            question_collection_add(sav_dict)
        return jsonify(success=True)
    except Exception as e:
        logging.error(f"historythequestionanswer() ERR: {e}")
        return jsonify({"error": "historythequestionanswer() ERR"}), 500


# 翻譯回答的路由
@app.route("/translatetheanswer", methods=["POST"])
def translatetheanswer():
    try:
        return jsonify({"translate_answer": en_to_lang(json.loads(request.data).get("text"))})
    except Exception as e:
        logging.error(f"translatetheanswer() ERR: {e}")
        return jsonify({"error": "translatetheanswer() ERR"}), 500


# 停止回答的路由
@app.route("/stoptheanswer", methods=["POST"])
def stoptheanswer():
    try:
        param.stop_all = True
        return Response(content_type="text/plain")
    except Exception as e:
        logging.error(f"stoptheanswer() ERR: {e}")
        return jsonify({"error": "stoptheanswer() ERR"}), 500


# 反饋回答的路由
@app.route("/feedbacktheanswer", methods=["POST"])
def feedbacktheanswer():
    global uuid_now, question_now, response_now
    try:
        if None not in (uuid_now, question_now, response_now):
            well_rate = request.form.get("wellrate") == "true"
            worse_rate = request.form.get("worserate") == "true"
            feedback = request.form.get("yourfeedback")
            history_subcollection_update(uuid_now, question=question_now, response=response_now, well_rate=well_rate, worse_rate=worse_rate, feedback=feedback)
        return redirect("/")
    except Exception as e:
        logging.error(f"feedbacktheanswer() ERR: {e}")
        return jsonify({"error": "feedbacktheanswer() ERR"}), 500


# 添加歷史紀錄的路由
@app.route("/addtableitem", methods=["POST"])
def addtableitem():
    global uuid_now, question_now, response_now
    try:
        uuid_now, question_now, response_now = None, None, None
        item_name = request.form.get("name")
        if item_name:
            item_uuid = history_collection_add(item_name)
            table_data.appendleft({"name": item_name, "selected": False, "uuid": item_uuid})
        return jsonify(success=True)
    except Exception as e:
        logging.error(f"addtableitem() ERR: {e}")
        return jsonify({"error": "addtableitem() ERR"}), 500


# 重命名歷史紀錄的路由
@app.route("/renametableitem/<int:index>/<uuid>", methods=["POST"])
def renametableitem(index, uuid):
    try:
        new_name = request.form.get("new_name")
        item = table_data[index]
        item["name"] = new_name
        history_collection_update(uuid, new_name)
        return jsonify(success=True)
    except Exception as e:
        logging.error(f"renametableitem() ERR: {e}")
        return jsonify({"error": "renametableitem() ERR"}), 500


# 刪除歷史紀錄的路由
@app.route("/deletetableitem/<int:index>/<uuid>", methods=["POST"])
def deletetableitem(index, uuid):
    global uuid_now, question_now, response_now
    try:
        uuid_now, question_now, response_now = None, None, None
        del table_data[index]
        sav_dict = history_collection_delete(uuid)
        if sav_dict["ids"] and sav_dict["documents"] and sav_dict["metadatas"]:
            question_collection_add(sav_dict)  
        return jsonify(success=True)
    except Exception as e:
        logging.error(f"deletetableitem() ERR: {e}")
        return jsonify({"error": "deletetableitem() ERR"}), 500


# 切換歷史紀錄選擇狀態的路由
@app.route("/toggletableitem/<uuid>", methods=["POST"])
def toggletableitem(uuid):
    global uuid_now, question_now, response_now
    try:
        uuid_now, question_now, response_now = None, None, None
        selected = request.args.get("selected") == "true"
        selected_data["selected"] = not selected
        if selected_data["selected"]:
            selected_data["uuid"] = uuid
            q_list, r_list, _, _, _ = history_subcollection_get(uuid)
        else:
            selected_data["uuid"] = None
            q_list, r_list = [], []
        return jsonify({ "q_list": q_list, "r_list": r_list })
    except Exception as e:
        logging.error(f"toggletableitem() ERR: {e}")
        return jsonify({"error": "toggletableitem() ERR"}), 500


# 載入語言頁面的路由
@app.route("/languagepage", methods=["GET"])
def languagepage():
    try:
        return render_template("languagepage.html")
    except Exception as e:
        logging.error(f"languagepage() ERR: {e}")
        return jsonify({"error": "languagepage() ERR"}), 500


# 切換至英文的路由
@app.route("/englishlanguage", methods=["POST"])
def englishlanguage():
    try:
        param.lang_code = "en"
        logging.info("\n" + "Change English Finish" + "\n")
        return render_template("languagepage.html")
    except Exception as e:
        logging.error(f"englishlanguage() ERR: {e}")
        return jsonify({"error": "englishlanguage() ERR"}), 500


# 切換至繁體中文的路由
@app.route("/chinesetraditionallanguage", methods=["POST"])
def chinesetraditionallanguage():
    try:
        param.lang_code = "zh-TW"
        logging.info("\n" + "Change Chinese Traditional Finish" + "\n")
        return render_template("languagepage.html")
    except Exception as e:
        logging.error(f"chinesetraditionallanguage() ERR: {e}")
        return jsonify({"error": "chinesetraditionallanguage() ERR"}), 500


# 切換至簡體中文的路由
@app.route("/chinesesimplifiedlanguage", methods=["POST"])
def chinesesimplifiedlanguage():
    try:
        param.lang_code = "zh-CN"
        logging.info("\n" + "Change Chinese Simplified Finish" + "\n")
        return render_template("languagepage.html")
    except Exception as e:
        logging.error(f"chinesesimplifiedlanguage() ERR: {e}")
        return jsonify({"error": "chinesesimplifiedlanguage() ERR"}), 500


# 切換至日文的路由
@app.route("/japaneselanguage", methods=["POST"])
def japaneselanguage():
    try:
        param.lang_code = "ja"
        logging.info("\n" + "Change Japanese Finish" + "\n")
        return render_template("languagepage.html")
    except Exception as e:
        logging.error(f"japaneselanguage() ERR: {e}")
        return jsonify({"error": "japaneselanguage() ERR"}), 500


# 切換至韓文的路由
@app.route("/koreanlanguage", methods=["POST"])
def koreanlanguage():
    try:
        param.lang_code = "ko"
        logging.info("\n" + "Change Korean Finish" + "\n")
        return render_template("languagepage.html")
    except Exception as e:
        logging.error(f"koreanlanguage() ERR: {e}")
        return jsonify({"error": "koreanlanguage() ERR"}), 500


# 載入設定頁面的路由
@app.route("/loadpage", methods=["GET"])
def loadpage():
    try:
        return render_template("loadpage.html")
    except Exception as e:
        logging.error(f"loadpage() ERR: {e}")
        return jsonify({"error": "loadpage() ERR"}), 500


# 加載所有設置的路由
@app.route("/loadallsettings", methods=["POST"])
def loadallsettings():
    try:
        load_all_settings()
        logging.info("\n" + "Load All Settings Finish" + "\n")
        return render_template("loadpage.html")
    except Exception as e:
        logging.error(f"loadallsettings() ERR: {e}")
        return jsonify({"error": "loadallsettings() ERR"}), 500


# 加載 Embedding 模型的路由
@app.route("/loadembeddingmodel", methods=["POST"])
def loadembeddingmodel():
    try:
        unload_embedding_model()
        load_embedding_model()
        logging.info("\n" + "Load Embedding Model Finish" + "\n")
        return render_template("loadpage.html")
    except Exception as e:
        logging.error(f"loadembeddingmodel() ERR: {e}")
        return jsonify({"error": "loadembeddingmodel() ERR"}), 500


# 創建與加載數據庫 Collection 的路由
@app.route("/createloadcollection", methods=["POST"])
def createloadcollection():
    try:
        reset_query_collection()
        create_load_query_collection()
        query_collection_add()
        logging.info("\n" + "Create Load Collection Finish" + "\n")
        return render_template("loadpage.html")
    except Exception as e:
        logging.error(f"createloadcollection() ERR: {e}")
        return jsonify({"error": "createloadcollection() ERR"}), 500


# 加載 GPT 模型的路由
@app.route("/loadgptmodel", methods=["POST"])
def loadgptmodel():
    try:
        unload_gpt_model()
        load_gpt_model_fn = {
            "exllama": exllama_load_gpt_model,
            "exllama2": exllama2_load_gpt_model,
            "llamacpp": llamacpp_load_gpt_model,
            "gptc": gptc_load_gpt_model,
        }
        load_gpt_model_fn.get((param.load_type or param.inference_config["load_type"]), gptgtpq_load_gpt_model)()
        logging.info("\n" + "Load GPT Model Finish" + "\n")
        return render_template("loadpage.html")
    except Exception as e:
        logging.error(f"loadgptmodel() ERR: {e}")
        return jsonify({"error": "loadgptmodel() ERR"}), 500


# 載入模式頁面的路由
@app.route("/modepage", methods=["GET"])
def modepage():
    try:
        return render_template("modepage.html")
    except Exception as e:
        logging.error(f"modepage() ERR: {e}")
        return jsonify({"error": "modepage() ERR"}), 500


# 切換至 Chat 模式的路由
@app.route("/chatmode", methods=["POST"])
def chatmode():
    try:
        load_all_settings("Chat.json")
        param.chat_mode = param.webui_config["chat_mode"]
        logging.info("\n" + "Change Chat Mode Finish" + "\n")
        return render_template("modepage.html")
    except Exception as e:
        logging.error(f"chatmode() ERR: {e}")
        return jsonify({"error": "chatmode() ERR"}), 500


# 切換至 QA 模式的路由
@app.route("/qamode", methods=["POST"])
def qamode():
    try:
        load_all_settings("QA.json")
        param.chat_mode = param.webui_config["chat_mode"]
        logging.info("\n" + "Change QA Mode Finish" + "\n")
        return render_template("modepage.html")
    except Exception as e:
        logging.error(f"qamode() ERR: {e}")
        return jsonify({"error": "qamode() ERR"}), 500


# 數據庫索引頁面的路由
@app.route("/databaseindexpage", methods=["GET"])
def databaseindexpage():
    try:
        return render_template("databaseindexpage.html")
    except Exception as e:
        logging.error(f"databaseindexpage() ERR: {e}")
        return jsonify({"error": "databaseindexpage() ERR"}), 500


# 數據庫索引的路由
@app.route("/embeddingdatabaseindex", methods=["POST"])
def embeddingdatabaseindex():
    try:
        clear_cache()
        param.instruction = param.prompt = param.completion = param.context = param.prompt_all = ""
        saveindex = request.form.get("saveindex") 
        param.prompt = rm_cr(request.form["query"])     
        _, result_2nd = index_query_collection(param.prompt)
        sep = "\n\n" + "-" * 100
        tokens = sum(embedding_token_num(itm["documents"][0][0]) for itm in result_2nd)
        param.context = "\n\n".join([itm["documents"][0][0] for itm in result_2nd])
        context = "\n\n".join([itm["documents"][0][0] + sep for itm in result_2nd])   
        if saveindex:
            global index_save
            index_data = {"Question": param.prompt, "Reference": context, "Answer": ""}
            index_save.append(index_data)
            index_save_path = path_combine(param.saveload_config["data_folder_path"], "index.json")
            save_json(index_save_path, index_save)
        get_prompt_template()
        tokens_with_template = param.gpt_model.gpt_token_num(param.prompt_all)
        response = f"Tokens: {tokens}\nTokens (with Prompt Template): {tokens_with_template}{sep}\n\n{context}"
        return render_template("databaseindexpage.html", query=param.prompt, response=response, checked=saveindex)
    except Exception as e:
        logging.error(f"embeddingdatabaseindex() ERR: {e}")
        return jsonify({"error": "embeddingdatabaseindex() ERR"}), 500


# 語義搜索頁面的路由
@app.route("/semanticsearchpage", methods=["GET"])
def semanticsearchpage():
    try:
        return render_template("semanticsearchpage.html")
    except Exception as e:
        logging.error(f"semanticsearchpage() ERR: {e}")
        return jsonify({"error": "semanticsearchpage() ERR"}), 500


# 語義搜索的路由
@app.route("/embeddingsemanticsearch", methods=["POST"])
def embeddingsemanticsearch():
    try:
        clear_cache()
        context_1 = rm_cr(request.form["context1"])
        context_2 = rm_cr(request.form["context2"])
        context_2_list = context_2.split("\n")
        similarity_list = []
        for i, context in enumerate(context_2_list):
            similar = similar_search(context_1, context)
            score = similar[0][0]["score"]
            tokens = embedding_token_num(context)
            similarity_list.append((i + 1, score, tokens))
        similarity_list_sorted = sorted(similarity_list, key=lambda x: x[1], reverse=True) 
        sortedbyscore = request.form.get("sorted")
        result = f"Tokens: {embedding_token_num(context_1)}\n\n"
        if sortedbyscore:
            result += "\n\n".join([f"{i}. score: {score}, tokens: {tokens}" for i, score, tokens in similarity_list_sorted])
        else:
            result += "\n\n".join([f"{i}. score: {score}, tokens: {tokens}" for i, score, tokens in similarity_list])
        return render_template("semanticsearchpage.html", context1=context_1, context2=context_2, result=result, checked=sortedbyscore)
    except Exception as e:
        logging.error(f"embeddingsemanticsearch() ERR: {e}")
        return jsonify({"error": "embeddingsemanticsearch() ERR"}), 500


# 啟動 WebUI
if __name__ == "__main__":
    try:
        load_all_settings()
        unload_embedding_model()
        load_embedding_model()
        create_load_query_collection()
        create_load_question_collection()
        create_load_history_collection()
        load_history_data()
        unload_gpt_model()
        load_gpt_model_fn = {
            "exllama": exllama_load_gpt_model,
            "exllama2": exllama2_load_gpt_model,
            "llamacpp": llamacpp_load_gpt_model,
            "gptc": gptc_load_gpt_model,
            } 
        load_gpt_model_fn.get((param.load_type or param.inference_config["load_type"]), gptgtpq_load_gpt_model)()
        app.run(host=(param.host_ip or param.webui_config["host_ip"]))
        logging.info("\n" + "WebUI Start" + "\n")
    except Exception as e:
        logging.error(f"WebUI Fail: {e}")