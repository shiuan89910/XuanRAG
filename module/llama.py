#import flash_attn
import logging
import numpy as np
import os
import param
import random
import re
import time
import torch
import torch.nn.functional as F
from pathlib import Path
from exllama.generator import ExLlamaGenerator
from exllama.model import ExLlama, ExLlamaCache, ExLlamaConfig
from exllama.tokenizer import ExLlamaTokenizer
from exllamav2 import ExLlamaV2, ExLlamaV2Cache, ExLlamaV2Config, ExLlamaV2Tokenizer
from exllamav2.generator import ExLlamaV2BaseGenerator, ExLlamaV2Sampler
from llama_cpp_cuda import Llama, LlamaCache, LlamaGrammar, LogitsProcessorList
from saveload import load_all_settings
from torch import version as torch_version
from util import clear_cache, en_to_lang, Generatior, get_prompt_template, gpt_inference, lang_to_en, prompt_max_length


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# Exllama2 預訓練模型類別
class Exllama2:
    def __init__(self):
        pass

    @classmethod
    def from_pretrained(cls, gpt_kwargs=None, exl2_kwargs=None):
        """加載預訓練模型"""
        try:
            gpt_kwargs = gpt_kwargs or param.gpt_config
            exl2_kwargs = exl2_kwargs or param.exllama2_config
            ret = cls()
            ret.gpt_config = ExLlamaV2Config()
            ret.gpt_config.model_dir = str(param.gpt_model_path)
            ret.gpt_config.prepare()
            ret.gpt_config.max_seq_len = exl2_kwargs["max_seq_len"]
            ret.gpt_config.scale_pos_emb = gpt_kwargs["compress_pos_emb"]
            ret.gpt_config.scale_alpha_value = gpt_kwargs["alpha_value"] 
            gpu_split = [float(split) for split in exl2_kwargs["gpu_split"].split(",")] if exl2_kwargs.get("gpu_split") else None
            ret.gpt_model = ExLlamaV2(ret.gpt_config)
            ret.gpt_model.load(gpu_split)
            ret.gpt_tokenizer = ExLlamaV2Tokenizer(ret.gpt_config)
            ret.gpt_cache = ExLlamaV2Cache(ret.gpt_model)
            ret.gpt_generator = ExLlamaV2BaseGenerator(ret.gpt_model, ret.gpt_cache, ret.gpt_tokenizer)
            return ret
        except Exception as e:
            logging.error(f"from_pretrained() ERR: {e}")
            raise

    def logits_fn(self, token_ids, **kwargs):
        """計算模型的 logits"""
        try:
            self.gpt_cache.current_seq_len = 0
            self.gpt_model.forward(token_ids[:, :-1], self.gpt_cache, input_mask=None, preprocess_only=True)
            return self.gpt_model.forward(token_ids[:, -1:], self.gpt_cache, **kwargs).float().cpu()
        except Exception as e:
            logging.error(f"logits_fn() ERR: {e}")
            raise

    def gpt_token_num(self, sequence):
        """計算序列的 token 數量"""
        try:
            return len(self.gpt_tokenizer.encode(sequence).cuda()[0])
        except Exception as e:
            logging.error(f"gpt_token_num() ERR: {e}")
            raise

    def encode(self, prompt, **kwargs):
        """對提示進行編碼"""
        try:
            return self.gpt_tokenizer.encode(prompt, add_bos=True)
        except Exception as e:
            logging.error(f"encode() ERR: {e}")
            raise

    def decode(self, output_ids, **kwargs):
        """對輸出進行解碼"""
        try:
            output_ids = torch.tensor(output_ids) if isinstance(output_ids, list) else output_ids
            output_ids = output_ids.view(1, -1) if isinstance(output_ids, torch.Tensor) and output_ids.numel() == 1 else output_ids
            return self.gpt_tokenizer.decode(output_ids)[0]
        except Exception as e:
            logging.error(f"decode() ERR: {e}")
            raise

    def generate_stream(self, prompt, kwargs):
        """生成文本流"""
        try:
            sets = ExLlamaV2Sampler.Settings()
            sets.temperature = kwargs["temperature"]
            sets.top_k = kwargs["top_k"]
            sets.top_p = kwargs["top_p"]
            sets.token_repetition_penalty = kwargs["repetition_penalty"]
            sets.token_repetition_range = -1 if kwargs["repetition_penalty_range"] <= 0 else kwargs["repetition_penalty_range"]
            if kwargs["ban_eos_token"]:
                sets.disallow_tokens(self.gpt_tokenizer, [self.gpt_tokenizer.eos_token_id])
            if kwargs["ban_token_customized"]:
                ban_token = [int(t) for t in kwargs["ban_token_customized"].split(",") if t]
                sets.disallow_tokens(self.gpt_tokenizer, ban_token)
            ids = self.gpt_tokenizer.encode(prompt, add_bos=kwargs["add_bos_token"])   
            ids = ids[:, -prompt_max_length(kwargs):]
            init_len = ids.shape[-1]
            if kwargs["max_new_tokens_auto"]:
                max_new_tokens = kwargs["truncation_length"] - ids.shape[-1]
            else:
                max_new_tokens = kwargs["max_new_tokens"]
            self.gpt_cache.current_seq_len = 0
            self.gpt_model.forward(ids[:, :-1], self.gpt_cache, input_mask=None, preprocess_only=True)
            leading_space = False
            for i in range(max_new_tokens):
                logits = self.gpt_model.forward(ids[:, -1:], self.gpt_cache, input_mask=None).float().cpu()
                token, _ = ExLlamaV2Sampler.sample(logits, sets, ids, random.random())
                ids = torch.cat([ids, token], dim=1)
                if i == 0 and self.gpt_tokenizer.tokenizer.IdToPiece(int(token)).startswith("▁"):
                    leading_space = True
                response = self.gpt_tokenizer.decode(ids[:, init_len:])[0]
                if leading_space:
                    response = " " + response
                if token.item() == self.gpt_tokenizer.eos_token_id or param.stop_all:
                    break
                yield response
        except Exception as e:
            logging.error(f"generate_stream() ERR: {e}")
            raise

    def generate(self, prompt, kwargs):
        """生成文本"""
        try:
            for response in self.generate_stream(prompt, kwargs):
                pass
            return response
        except Exception as e:
            logging.error(f"generate() ERR: {e}")
            raise
    
# 加載 Exllama2 模型
def exllama2_load_gpt_model():
    try:
        param.gpt_model = Exllama2.from_pretrained()
        logging.info("Load GPT Model Success")
    except Exception as e:
        logging.error(f"exllama2_load_gpt_model() ERR: {e}")


# Exllama 預訓練模型類別
class Exllama:
    def __init__(self):
        pass

    @classmethod
    def from_pretrained(cls, gpt_kwargs=None, exl_kwargs=None):
        """加載預訓練模型"""
        try:
            gpt_kwargs = gpt_kwargs or param.gpt_config
            exl_kwargs = exl_kwargs or param.exllama_config
            gpt_config = os.path.join(param.gpt_model_path, "config.json")
            gpt_tokenizer = os.path.join(param.gpt_model_path, "tokenizer.model")
            basename_path = next((file for ext in [".safetensors", ".pt", ".bin"] for file in Path(param.gpt_model_path).glob(f"*{ext}")), None)  
            ret = cls()
            ret.gpt_config = ExLlamaConfig(str(gpt_config))
            ret.gpt_config.model_path = str(basename_path)
            ret.gpt_config.max_seq_len = exl_kwargs["max_seq_len"]
            ret.gpt_config.compress_pos_emb = gpt_kwargs["compress_pos_emb"]
            if exl_kwargs["gpu_split"]:
                ret.gpt_config.set_auto_map(exl_kwargs["gpu_split"])
                ret.gpt_config.gpu_peer_fix = True
            if gpt_kwargs["alpha_value"] > 1 and gpt_kwargs["rope_freq_base"] == 0:
                ret.gpt_config.alpha_value = gpt_kwargs["alpha_value"]
                ret.gpt_config.calculate_rotary_embedding_base()
            elif gpt_kwargs["rope_freq_base"] > 0:
                ret.gpt_config.rotary_embedding_base = gpt_kwargs["rope_freq_base"]
            if torch_version.hip:
                ret.gpt_config.rmsnorm_no_half2 = True
                ret.gpt_config.rope_no_half2 = True
                ret.gpt_config.matmul_no_half2 = True
                ret.gpt_config.silu_no_half2 = True
            ret.gpt_model = ExLlama(ret.gpt_config)
            ret.gpt_tokenizer = ExLlamaTokenizer(str(gpt_tokenizer))
            ret.gpt_cache = ExLlamaCache(ret.gpt_model)
            ret.gpt_generator = ExLlamaGenerator(ret.gpt_model, ret.gpt_tokenizer, ret.gpt_cache)
            return ret
        except Exception as e:
            logging.error(f"from_pretrained() ERR: {e}")
            raise

    def logits_fn(self, token_ids, **kwargs):
        """計算模型的 logits"""
        try:
            self.gpt_cache.current_seq_len = 0
            self.gpt_model.forward(token_ids[:, :-1], self.gpt_cache, input_mask=None, preprocess_only=True)
            return self.gpt_model.forward(token_ids[:, -1:], self.gpt_cache, **kwargs).float().cpu()
        except Exception as e:
            logging.error(f"logits_fn() ERR: {e}")
            raise

    def gpt_token_num(self, sequence):
        """計算序列的 token 數量"""
        try:
            return len(self.gpt_tokenizer.encode(sequence).cuda()[0])
        except Exception as e:
            logging.error(f"gpt_token_num() ERR: {e}")
            raise

    def encode(self, prompt, **kwargs):
        """對提示進行編碼"""
        try:
            return self.gpt_tokenizer.encode(prompt, max_seq_len=self.gpt_model.config.max_seq_len, add_bos=True)
        except Exception as e:
            logging.error(f"encode() ERR: {e}")
            raise

    def decode(self, output_ids, **kwargs):
        """對輸出進行解碼"""
        try:
            output_ids = torch.tensor(output_ids) if isinstance(output_ids, list) else output_ids
            output_ids = output_ids.view(1, -1) if isinstance(output_ids, torch.Tensor) and output_ids.numel() == 1 else output_ids
            return self.gpt_tokenizer.decode(output_ids)[0]
        except Exception as e:
            logging.error(f"decode() ERR: {e}")
            raise

    def generate_response(self, i, token, init_len, leading_space, gpt_generator):
        """根據生成的 token 生成響應文本"""
        try:
            if i == 0 and gpt_generator.tokenizer.tokenizer.IdToPiece(int(token)).startswith("▁"):
                leading_space = True
            response = gpt_generator.tokenizer.decode(gpt_generator.sequence[0][init_len:])
            if leading_space:
                response = " " + response
            return response
        except Exception as e:
            logging.error(f"generate_response() ERR: {e}")
            raise

    def generate_stream(self, prompt, kwargs):
        """生成文本流"""
        try:
            if kwargs["guidance_scale"] == 1 and self.gpt_cache.batch_size == 2:
                del self.gpt_cache
                clear_cache()
                self.gpt_cache = ExLlamaCache(self.gpt_model)
            elif kwargs["guidance_scale"] != 1 and self.gpt_cache.batch_size == 1:
                del self.gpt_cache
                clear_cache()
                self.gpt_cache = ExLlamaCache(self.gpt_model, batch_size=2)
            self.gpt_generator = ExLlamaGenerator(self.gpt_model, self.gpt_tokenizer, self.gpt_cache)
            self.gpt_generator.settings.temperature = kwargs["temperature"]
            self.gpt_generator.settings.top_p = kwargs["top_p"]
            self.gpt_generator.settings.top_k = kwargs["top_k"]
            self.gpt_generator.settings.typical = kwargs["typical_p"]
            self.gpt_generator.settings.token_repetition_penalty_max = kwargs["repetition_penalty"]
            self.gpt_generator.settings.token_repetition_penalty_sustain = -1 if kwargs["repetition_penalty_range"] <= 0 else kwargs["repetition_penalty_range"]
            if kwargs["ban_eos_token"]:
                self.gpt_generator.disallow_tokens([self.gpt_tokenizer.eos_token_id])
            else:
                self.gpt_generator.disallow_tokens(None)
            if kwargs["ban_token_customized"]:
                ban_token = [int(t) for t in kwargs["ban_token_customized"].split(",") if t]
                self.gpt_generator.disallow_tokens(ban_token)
            max_new_tokens = kwargs["truncation_length"] - ids.shape[-1] if kwargs["max_new_tokens_auto"] else kwargs["max_new_tokens"]
            leading_space = False
            if kwargs["guidance_scale"] == 1:
                self.gpt_generator.end_beam_search()
                ids = self.gpt_generator.tokenizer.encode(prompt, max_seq_len=self.gpt_model.config.max_seq_len)    
                if kwargs["add_bos_token"]:
                    ids = torch.cat([torch.tensor([[self.gpt_tokenizer.bos_token_id]]).to(ids.device), ids], dim=1).to(torch.int64)   
                ids = ids[:, -prompt_max_length(kwargs):]
                self.gpt_generator.gen_begin_reuse(ids)
                init_len = self.gpt_generator.sequence[0].shape[0]
                for i in range(max_new_tokens):
                    token = self.gpt_generator.gen_single_token()
                    if token.item() == self.gpt_generator.tokenizer.eos_token_id or param.stop_all:
                        break
                    yield self.generate_response(i, token, init_len, leading_space, self.gpt_generator)
            else:
                alpha = kwargs["guidance_scale"]
                prompts = [prompt, kwargs["negative_prompt_ids"] or ""]
                ids, mask = self.gpt_tokenizer.encode(prompts, return_mask=True, max_seq_len=self.gpt_model.config.max_seq_len, add_bos=kwargs["add_bos_token"])       
                self.gpt_generator.gen_begin(ids, mask=mask)
                init_len = self.gpt_generator.sequence[0].shape[0]
                for i in range(max_new_tokens):
                    logits = self.model.forward(self.gpt_generator.sequence[:, -1:], self.gpt_cache, input_mask=mask)
                    self.gpt_generator.apply_rep_penalty(logits)
                    logits = F.log_softmax(logits, dim=-1)
                    merge_logits = alpha * logits[0] + (1 - alpha) * logits[1]
                    token, _ = self.gpt_generator.sample_current(merge_logits)
                    if token.item() == self.gpt_tokenizer.eos_token_id or param.stop_all:
                        break
                    yield self.generate_response(i, token, init_len, leading_space, self.gpt_generator)
                    token_batch = token.repeat(2, 1)
                    self.gpt_generator.gen_accept_token(token_batch)
        except Exception as e:
            logging.error(f"generate_stream() ERR: {e}")
            raise

    def generate(self, prompt, kwargs):
        """生成文本"""
        try:
            for response in self.generate_stream(prompt, kwargs):
                pass
            return response
        except Exception as e:
            logging.error(f"generate() ERR: {e}")
            raise

# 加載 Exllama 模型
def exllama_load_gpt_model():
    try:
        param.gpt_model = Exllama.from_pretrained()
        logging.info("Load GPT Model Success")
    except Exception as e:
        logging.error(f"exllama_load_gpt_model() ERR: {e}")


# LlamaCpp 預訓練模型類別
class LlamaCpp:
    def __init__(self):
        self.init = False
        self.grammar_string = ""
        self.grammar = None

    def __del__(self):
        self.model.__del__()

    @classmethod
    def from_pretrained(cls, gpt_kwargs=None, lcpp_kwargs=None):
        """加載預訓練模型"""
        try:
            gpt_kwargs = gpt_kwargs or param.gpt_config
            lcpp_kwargs = lcpp_kwargs or  param.llamacpp_config
            ret = cls()
            capacity_bytes = 0
            if lcpp_kwargs["capacity_bytes"] is not None:
                if "GiB" in lcpp_kwargs["capacity_bytes"]:
                    capacity_bytes = int(re.sub("[a-zA-Z]", "", lcpp_kwargs["capacity_bytes"])) * 1000 * 1000 * 1000
                elif "MiB" in lcpp_kwargs["capacity_bytes"]:
                    capacity_bytes = int(re.sub("[a-zA-Z]", "", lcpp_kwargs["capacity_bytes"])) * 1000 * 1000
                else:
                    capacity_bytes = int(lcpp_kwargs["capacity_bytes"])
            tensor_split = [float(ts) for ts in lcpp_kwargs["tensor_split"].split(",")] if lcpp_kwargs["tensor_split"] and lcpp_kwargs["tensor_split"].strip() else None
            sets = {
                "model_path": str(Path(param.gpt_model_path)) if Path(param.gpt_model_path).is_file() else str(list(Path(param.gpt_model_path).glob("*.gguf"))[0]),
                "n_ctx": lcpp_kwargs["n_ctx"],
                "seed": int(lcpp_kwargs["seed"]),
                "n_threads": lcpp_kwargs["n_threads"] or None,
                "n_batch": lcpp_kwargs["n_batch"],
                "use_mmap": not lcpp_kwargs["use_mmap"],
                "use_mlock": lcpp_kwargs["use_mlock"],
                "low_vram": lcpp_kwargs["low_vram"],
                "n_gpu_layers": lcpp_kwargs["n_gpu_layers"],    
                "rope_freq_base": gpt_kwargs["rope_freq_base"] if gpt_kwargs["rope_freq_base"] > 0 else 10000 * gpt_kwargs["alpha_value"] ** (64 / 63.),
                "rope_freq_scale": 1.0 / gpt_kwargs["compress_pos_emb"],
                "mul_mat_q": lcpp_kwargs["mul_mat_q"],
                "tensor_split": tensor_split}
            ret.gpt_model = Llama(**sets)
            if capacity_bytes > 0:
                ret.gpt_model.set_cache(LlamaCache(capacity_bytes=capacity_bytes))
            return ret
        except Exception as e:
            logging.error(f"from_pretrained() ERR: {e}")
            raise

    def logits_fn(self, tokens):
        """計算模型的 logits"""
        try:
            self.gpt_model.eval(tokens)
            lgs = np.expand_dims(self.gpt_model._scores, 0)
            return torch.tensor(lgs, dtype=torch.float32)
        except Exception as e:
            logging.error(f"logits_fn() ERR: {e}")
            raise

    def grammar_fn(self, grammar_str):
        """更新語法分析器的語法字串"""
        try:
            if grammar_str != self.grammar_string:
                self.grammar_string = grammar_str
                self.grammar = LlamaGrammar.from_string(grammar_str) if grammar_str.strip() else None
        except Exception as e:
            logging.error(f"grammar_fn() ERR: {e}")
            raise

    def gpt_token_num(self, sequence):
        """計算序列的 token 數量"""
        try:
            return len(self.gpt_model.tokenize(sequence.encode() if isinstance(sequence, str) else sequence))
        except Exception as e:
            logging.error(f"gpt_token_num() ERR: {e}")
            raise

    def encode(self, prompt, **kwargs):
        """對輸入進行編碼"""
        try:
            return self.gpt_model.tokenize(prompt.encode() if isinstance(prompt, str) else prompt)
        except Exception as e:
            logging.error(f"encode() ERR: {e}")
            raise

    def decode(self, output_ids, **kwargs):
        """對輸出進行解碼"""
        try:
            return self.gpt_model.detokenize(output_ids).decode("utf-8")
        except Exception as e:
            logging.error(f"decode() ERR: {e}")
            raise

    def generate(self, prompt, kwargs, callback=None, if_kwargs=None):
        """生成文本"""
        try:
            if_kwargs = if_kwargs or param.inference_config
            prompt = prompt if isinstance(prompt, str) else prompt.decode()
            prompt = self.encode(prompt)[-prompt_max_length(kwargs):]
            prompt = self.decode(prompt)
            self.grammar_fn(kwargs["grammar_str"])
            logit_processors = LogitsProcessorList()
            if kwargs["ban_eos_token"]:
                logit_processors.append(lambda logits: {k: v if k != self.model.token_eos() else -float("inf") for k, v in logits.items()})
            if kwargs["ban_token_customized"]:
                ban_tokens = [int(t) for t in kwargs["ban_token_customized"].split(",") if t]
                for ban_token in ban_tokens:
                    logit_processors.append(lambda logits: {k: v if k != ban_token else -float("inf") for k, v in logits.items()})
            generator = self.gpt_model.create_completion(
                prompt=prompt,
                max_tokens=kwargs["max_new_tokens"],
                temperature=kwargs["temperature"],
                top_p=kwargs["top_p"],
                top_k=kwargs["top_k"],
                repeat_penalty=kwargs["repetition_penalty"],
                tfs_z=kwargs["tfs"],
                mirostat_mode=int(kwargs["mirostat_mode"]),
                mirostat_tau=kwargs["mirostat_tau"],
                mirostat_eta=kwargs["mirostat_eta"],
                stream=if_kwargs["stream"],
                logits_processor=logit_processors,
                grammar=self.grammar,
                )
            response = ""       
            for token in generator:
                if param.stop_all:
                    break
                response += token["choices"][0]["text"] 
                if callback:
                    callback(token["choices"][0]["text"])
            return response
        except Exception as e:
            logging.error(f"generate() ERR: {e}")
            raise

    def generate_stream(self, *args, **kwargs):
        """生成文本流"""
        try:
            with Generatior(self.generate, args, kwargs, callback=None) as generator:
                response = ""
                for token in generator:
                    response += token
                    yield response
        except Exception as e:
            logging.error(f"generate_stream() ERR: {e}")
            raise

# 加載 LlamaCpp 模型
def llamacpp_load_gpt_model():
    try:
        param.gpt_model = LlamaCpp.from_pretrained()
        logging.info("Load GPT Model Success")
    except Exception as e:
        logging.error(f"llamacpp_load_gpt_model() ERR: {e}")


def test_llama():
    # 加載所有設定
    load_all_settings()

    # 加載模型
    exllama_load_gpt_model()
    #exllama2_load_gpt_model()
    #llamacpp_load_gpt_model()

    # 將輸入從中文轉換為英文
    prompt_zh_tw = "問題"
    start_time = time.time()
    param.prompt = lang_to_en(prompt_zh_tw)
    end_time = time.time()
    execution_time = end_time - start_time
    logging.info(f"Time Translate (Q) (Sec):\n{execution_time}\n")
    #logging.info(f"{param.prompt}\n\n")

    # 設定上下文和獲取提示模板
    param.context = "The context"
    get_prompt_template()

    # 進行推理
    start_time = time.time()
    for token in gpt_inference(param.prompt_all, param.gpt_inference_params):
        response = token
    end_time = time.time()
    execution_time = end_time - start_time
    response_without_prompt = response.replace(param.prompt_all, "")
    num_token = param.gpt_model.gpt_token_num(response_without_prompt)
    logging.info(f"Time Total (Sec):\n{execution_time}\n")
    logging.info(f"Token Total:\n{num_token}\n")
    logging.info(f"Token per Time (Sec):\n{num_token / execution_time}\n")
    #logging.info(f"{response_without_prompt}\n\n")

    # 將輸出從英文轉換回中文
    start_time = time.time()
    response_without_prompt_zh_tw = en_to_lang(response_without_prompt)
    end_time = time.time()
    execution_time = end_time - start_time
    logging.info(f"Time Translate (A) (Sec):\n{execution_time}\n")
    #logging.info(f"{response_without_prompt_zh_tw}\n\n")