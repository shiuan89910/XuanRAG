import gc
import logging
import param
import time
import torch
import transformers
import uuid
from deep_translator import GoogleTranslator
from queue import Queue
from threading import Thread


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# 檢查列表是否為空
def ls_mpt(lst):
    try:
        return None if not lst else lst
    except Exception as e:
        logging.error(f"ls_mpt() ERR: {e}")
        return None


# 移除字串中的 "\r"
def rm_cr(string):
    try:
        return string.replace("\r", "")
    except Exception as e:
        logging.error(f"rm_cr() ERR: {e}")
        return string


# 生成唯一的 UUID，並確保不與提供的 UUID 列表重複
def unique_uuid(uuid_list):
    try:
        while True:
            id = str(uuid.uuid4())
            if id not in uuid_list:
                uuid_list.append(id)
                return id
    except Exception as e:
        logging.error(f"unique_uuid() ERR: {e}")
        return None


# 根據提供的替換字典進行字串替換
def string_replace(sequences, replacements):
    try:
        for ori, new in replacements.items():
            if new is not None: 
                sequences = sequences.replace(ori, new)
        return sequences
    except Exception as e:
        logging.error(f"string_replace() ERR: {e}")
        return sequences


# 根據提供的問題列表和回答列表以及其他參數，生成完整的對話模板
def get_prompt_template(q_list=None, r_list=None, kwargs=None):
    try:
        kwargs = kwargs or param.prompt_config
        param.conversation_list = []
        if kwargs["is_conversation"]:
            if q_list and r_list:
                for q, r in zip(q_list, r_list):
                    param.conversation_list.append(string_replace(kwargs["prompt"], {"{prompt}": q}))
                    param.conversation_list.append(string_replace(kwargs["completion"], {"{completion}": r}))
            param.conversation_list = param.conversation_list[-(kwargs["max_conversation_len"] * 2):]
        param.conversation_list.insert(0, string_replace(kwargs["template"], {"{instruction}": param.instruction or kwargs["instruction"], "{context}": param.context}))
        param.conversation_list.append(string_replace(kwargs["prompt"], {"{prompt}": param.prompt}))
        param.prompt_all = "".join(param.conversation_list)
    except Exception as e:
        logging.error(f"get_prompt_template() ERR: {e}") 


# 將提供的文本從指定語言翻譯成英語
def lang_to_en(context, kwargs=None):
    try:
        kwargs = kwargs or param.webui_config
        if (param.lang_code or kwargs["lang_code"]) == "en":
            return context
        else:
            return GoogleTranslator(source=(param.lang_code or kwargs["lang_code"]), target="en").translate(context)
    except Exception as e:
        logging.error(f"lang_to_en() ERR: {e}")
        return context


# 將提供的英文文本翻譯成指定的語言
def en_to_lang(context, kwargs=None):
    try:
        kwargs = kwargs or param.webui_config
        if (param.lang_code or kwargs["lang_code"]) == "en":
            return context
        else:
            return GoogleTranslator(source="en", target=(param.lang_code or kwargs["lang_code"])).translate(context)
    except Exception as e:
        logging.error(f"en_to_lang() ERR: {e}")
        return context


# 清除內存和 GPU 緩存
def clear_cache():
    try:
        gc.collect()
        torch.cuda.empty_cache()
    except Exception as e:
        logging.error(f"clear_cache() ERR: {e}")


# 卸載 GPT 模型並清除緩存
def unload_gpt_model():
    try:
        param.gpt_model = None
        clear_cache()
    except Exception as e:
        logging.error(f"unload_gpt_model() ERR: {e}")


# 計算提示的最大長度
def prompt_max_length(kwargs):
    try:
        return kwargs["truncation_length"] - kwargs["max_new_tokens"]
    except Exception as e:
        logging.error(f"prompt_max_length() ERR: {e}")
        return None


# 進行 GPT 推理
def gpt_inference(query, kwargs, if_kwargs=None):
    try:
        if_kwargs = if_kwargs or param.inference_config
        param.stop_all = False
        clear_cache()
        time_update = -1
        response = ""
        if not if_kwargs["stream"]:
            generator = param.gpt_model.generate(query, kwargs)
        else:
            generator = param.gpt_model.generate_stream(query, kwargs)
        for response in generator:
            is_stop = False
            for string in kwargs["stop_strings"]:
                idx = response.find(string)
                if idx != -1:
                    response = response[:idx]
                    is_stop = True
                    break
            if not is_stop:
                for string in kwargs["stop_strings"]:
                    for j in range(len(string) - 1, 0, -1):
                        if response[-j:] == string[:j]:
                            response = response[:-j]
                            break
                    else:
                        continue
                    break
            time_now = time.time()
            if kwargs["max_tokens_per_sec"] > 0:
                time_diff = 1 / kwargs["max_tokens_per_sec"] - (time_now - time_update)
                if time_diff > 0:
                    time.sleep(time_diff)
                time_update = time.time()
                yield response
            else:    
                if time_now - time_update > 0.041666666666666664:
                    time_update = time_now
                    yield response    
            if is_stop:
                break
        yield response
    except Exception as e:
        logging.error(f"gpt_inference() ERR: {e}")
        yield ""


# 異步生成器類別
class Generatior:
    def __init__(self, fn, args=None, kwargs=None, callback=None):
        self.fn = fn
        self.args = args or []
        self.kwargs = kwargs or {}
        self.callback = callback
        self.q = Queue()
        self.sentinel = object()
        self.stop = False

        def callback_fun(ids):
            """將任務結果放入隊列"""
            if self.stop or param.stop_all:
                raise ValueError
            self.q.put(ids)

        def task():
            """執行傳入的函數並處理結果"""
            try:
                ret = self.fn(callback=callback_fun, *self.args, **self.kwargs)
            except ValueError:
                pass
            clear_cache()
            self.q.put(self.sentinel)
            if self.callback:
                self.callback(ret)
        self.thread = Thread(target=task)
        self.thread.start()

    def __iter__(self):
        return self

    def __next__(self):
        """獲取隊列中的下一個結果"""
        ids = self.q.get(True, None)
        if ids is self.sentinel:
            raise StopIteration
        else:
            return ids

    def __del__(self):
        clear_cache()

    def __exit__(self, type, val, tb):
        self.stop = True
        clear_cache()

    def __enter__(self):
        return self


# 定義流式生成停止條件
class Stream(transformers.StoppingCriteria):
    def __init__(self, callback):
        self.callback = callback

    def __call__(self, input_ids, score) -> bool:
        """呼叫回調函數並決定是否停止生成"""
        try:
            if self.callback:
                self.callback(input_ids[0])
            return False
        except Exception as e:
            logging.error(f"__call__() ERR: {e}")
            return True


# 定義自定義停止條件
class StopCriteria(transformers.StoppingCriteria):
    def __init__(self):
        try:
            super().__init__()
        except Exception as e:
            logging.error(f"__init__() ERR: {e}")

    def __call__(self, input_ids: torch.LongTensor, score: torch.FloatTensor) -> bool:
        """根據全域停止標誌決定是否停止生成"""
        return param.stop_all