import json
import logging
import os
import param
import pdfplumber


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# 從指定路徑加載 PDF 文件，並提取每一頁的文字
def load_pdf(pdf_path):
    try:
        with pdfplumber.open(pdf_path) as pdf:
            return [page.extract_text() for page in pdf.pages]
    except Exception as e:
        logging.error(f"load_pdf() ERR: {e}")
        return None


# 將文字資料儲存到 TXT 檔案
def save_txt(txt_path, txt_data, sep="\n" * 5):
    try:
        with open(txt_path, "w", encoding="utf-8") as fw:
            fw.write(sep.join(txt_data))
    except Exception as e:
        logging.error(f"save_txt() ERR: {e}")


# 從 TXT 檔案加載文字資料
def load_txt(txt_path):
    try:
        with open(txt_path,"r", encoding="utf-8") as fr:
            return fr.read()
    except Exception as e:
        logging.error(f"load_txt() ERR: {e}")
        return None


# 將資料儲存為 JSON 格式
def save_json(json_path, data):
    try:
        with open(json_path, "w") as fw:
            json.dump(data, fw, indent=4)
    except Exception as e:
        logging.error(f"save_json() ERR: {e}")


# 從 JSON 檔案加載資料
def load_json(json_path):
    try:
        with open(json_path, "r") as fr:
            return json.load(fr)
    except Exception as e:
        logging.error(f"load_json() ERR: {e}")
        return None


# 組合文件夾路徑和檔案名稱
def path_combine(folder_path, name=None):
    try:
        folder_path = folder_path.replace("{user_name}", param.user_name).replace("{proj_name}", param.proj_name)
        return os.path.join(folder_path, name) if name else folder_path
    except Exception as e:
        logging.error(f"path_combine() ERR: {e}")
        return None


# 加載所有設定
def load_all_settings(setting_save_name=None):
    try:
        setting_save_name = setting_save_name or param.setting_save_name
        param.setting_save_path = os.path.join(param.setting_save_folder_path, param.setting_save_name)
        param.all_setting_config = load_json(param.setting_save_path)
        config_keys = [
            "saveload_config", "embedding_config", "database_config", "gpt_config","bitsandbites_config", 
            "autogptq_config", "ctransformers_config", "exllama_config", "exllama2_config", "llamacpp_config",
            "inference_config", "prompt_config", "webui_config", "history_config",
            ]
        for key in config_keys:
            setattr(param, key, param.all_setting_config[key])
        param.gpt_model_path = path_combine(param.gpt_config["gpt_model_folder_path"], param.inference_config["gpt_model_name"])
        for key in ["load_type", "gpt_type", "bitsandbites", "gptq", "offload", "gpt_inference_params"]:
            setattr(param, key, param.inference_config[key])
    except Exception as e:
        logging.error(f"load_all_settings() ERR: {e}")