import logging
import numpy as np
import os
import param
import time
import torch
import ctransformers
import transformers
from accelerate import infer_auto_device_map, init_empty_weights
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from pathlib import Path
from saveload import load_all_settings, path_combine
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GPTQConfig 
from util import clear_cache, en_to_lang, Generatior, get_prompt_template, gpt_inference, lang_to_en, prompt_max_length, StopCriteria, Stream


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# ctransformers 預訓練模型類別
class gptC:
    def __init__(self):
        pass

    @classmethod
    def from_pretrained(cls, ctrans_kwargs=None, if_kwargs=None):
        """加載預訓練模型"""
        try:
            ctrans_kwargs = ctrans_kwargs or param.ctransformers_config
            if_kwargs = if_kwargs or param.inference_config
            gpt_model_path_name = Path(param.gpt_model_path) if ctrans_kwargs["model_type"] is None else Path(param.gpt_model_path) if Path(param.gpt_model_path).is_file() else list(Path(param.gpt_model_path).glob("*.gguf"))[0]
            ret = cls()
            gpt_config = ctransformers.AutoConfig.from_pretrained(
                str(gpt_model_path_name),
                threads=ctrans_kwargs["threads"] if ctrans_kwargs["threads"] != 0 else -1,
                gpu_layers=ctrans_kwargs["gpu_layers"],
                batch_size=ctrans_kwargs["batch_size"],
                context_length=ctrans_kwargs["context_length"],
                stream=if_kwargs["stream"],
                mmap=not ctrans_kwargs["mmap"],
                mlock=ctrans_kwargs["mlock"],
                )
            ret.gpt_model = ctransformers.AutoModelForCausalLM.from_pretrained(
                str(gpt_model_path_name.parent if gpt_model_path_name.is_file() and ctrans_kwargs["model_type"] is None else gpt_model_path_name),
                model_type=(None if ctrans_kwargs["model_type"] is None else ctrans_kwargs["model_type"]),
                config=gpt_config,
                )
            return ret
        except Exception as e:
            logging.error(f"from_pretrained() ERR: {e}")
            raise

    def gpt_token_num(self, sequence):
        """計算序列的 token 數量"""
        try:
            return len(self.gpt_model.tokenize(sequence))
        except Exception as e:
            logging.error(f"gpt_token_num() ERR: {e}")
            raise

    def encode(self, prompt, **kwargs):
        """對提示進行編碼"""
        try:
            return self.gpt_model.tokenize(prompt)
        except Exception as e:
            logging.error(f"encode() ERR: {e}")
            raise

    def decode(self, output_ids, **kwargs):
        """對輸出進行解碼"""
        try:
            return self.gpt_model.detokenize(output_ids)
        except Exception as e:
            logging.error(f"decode() ERR: {e}")
            raise

    def generate(self, prompt, kwargs, callback=None):
        """生成文本"""
        try:
            prompt = prompt if isinstance(prompt, str) else prompt.decode()
            generator = self.gpt_model(
                prompt=prompt,
                max_new_tokens=kwargs["max_new_tokens"],
                temperature=kwargs["temperature"],
                top_p=kwargs["top_p"],
                top_k=kwargs["top_k"],
                repetition_penalty=kwargs["repetition_penalty"],
                last_n_tokens=kwargs["repetition_penalty_range"],
                seed=int(kwargs["seed"]),
                )
            response = ""
            for token in generator:
                if callback:
                    callback(token)
                response += token
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

# 加載 ctransformers 模型
def gptc_load_gpt_model():
    try:
        param.gpt_model = gptC.from_pretrained()
        logging.info("Load GPT Model Success")
    except Exception as e:
        logging.error(f"gptc_load_gpt_model() ERR: {e}")


# transformers 或 autogptq 預訓練模型類別
class gptgptq:
    def __init__(self):
        pass

    @classmethod
    def from_pretrained(cls, gpt_kwargs=None, agptq_kwargs=None, bsb_kwargs=None):
        """加載預訓練模型"""
        try:
            gpt_kwargs = gpt_kwargs or param.gpt_config
            agptq_kwargs = agptq_kwargs or param.autogptq_config
            bsb_kwargs = bsb_kwargs or param.bitsandbites_config
            param.max_memory = {0: gpt_kwargs["gpu_max_memory"], "cpu": gpt_kwargs["cpu_max_memory"]}
            ret = cls()
            # autogptq 加載預訓練模型
            if (param.gpt_type or param.inference_config["gpt_type"]) == "autogptq":
                basename_path = None
                for ext in [".safetensors", ".pt", ".bin"]:
                    exist = list(Path(param.gpt_model_path).glob(f"*{ext}"))
                    if exist:
                        basename_path = exist[-1]
                        break
                use_safetensors = basename_path.suffix == ".safetensors"
                quantize_config = None
                if not os.path.exists(os.path.join(param.gpt_model_path, "quantize_config.json")):
                    quantize_config = BaseQuantizeConfig(bits=agptq_kwargs["bits"], group_size=agptq_kwargs["group_size"], desc_act=agptq_kwargs["desc_act"])
                ret.gpt_model = AutoGPTQForCausalLM.from_quantized(
                    param.gpt_model_path,
                    model_basename=Path(Path(basename_path).name).stem,
                    trust_remote_code=gpt_kwargs["trust_remote_code"],
                    low_cpu_mem_usage=gpt_kwargs["low_cpu_mem_usage"],
                    max_memory=param.max_memory,
                    device_map=gpt_kwargs["device_map"],
                    device=gpt_kwargs["device"],
                    use_safetensors=use_safetensors,
                    quantize_config=quantize_config,
                    use_triton=agptq_kwargs["use_triton"],
                    inject_fused_attention=agptq_kwargs["inject_fused_attention"],
                    inject_fused_mlp=agptq_kwargs["inject_fused_mlp"],
                    use_cuda_fp16=agptq_kwargs["use_cuda_fp16"],
                    disable_exllama=agptq_kwargs["disable_exllama"],
                    )
            # transformers 加載預訓練模型
            else:
                offload_folder, offload_state_dict, rope_scaling, quantization_config = None, None, None, None
                device_map = gpt_kwargs["device_map"]  
                if (param.offload or param.inference_config["offload"]):
                    offload_folder = path_combine(gpt_kwargs["offload_folder"])
                    offload_state_dict = gpt_kwargs["offload_state_dict"]
                if gpt_kwargs["compress_pos_emb"] > 1:        
                    rope_scaling = {"type": "linear", "factor": gpt_kwargs["compress_pos_emb"]}
                elif gpt_kwargs["alpha_value"] > 1:    
                    rope_scaling = {"type": "dynamic", "factor": (gpt_kwargs["rope_freq_base"] / 10000.) ** (63 / 64.) if gpt_kwargs["rope_freq_base"] > 0 else gpt_kwargs["alpha_value"]}  
                if (param.bitsandbites or param.inference_config["bitsandbites"]):
                    quantization_config = BitsAndBytesConfig(
                        load_in_8bit=bsb_kwargs["load_in_8bit"], 
                        llm_int8_enable_fp32_cpu_offload=bsb_kwargs["llm_int8_enable_fp32_cpu_offload"], 
                        load_in_4bit=bsb_kwargs["load_in_4bit"], 
                        bnb_4bit_quant_type=bsb_kwargs["bnb_4bit_quant_type"], 
                        bnb_4bit_use_double_quant=bsb_kwargs["bnb_4bit_use_double_quant"], 
                        bnb_4bit_compute_dtype=eval("torch.{}".format(bsb_kwargs["bnb_4bit_compute_dtype"])) if bsb_kwargs["bnb_4bit_compute_dtype"] in ["bfloat16", "float16", "float32"] else None,
                        )
                    if not bsb_kwargs["load_in_4bit"] and param.max_memory is not None:
                        with init_empty_weights():
                            config = transformers.AutoConfig.from_pretrained(param.gpt_model_path, trust_remote_code=gpt_kwargs["trust_remote_code"])
                            model = AutoModelForCausalLM.from_config(config, trust_remote_code=gpt_kwargs["trust_remote_code"])
                        model.tie_weights()
                        device_map = infer_auto_device_map(model, dtype=torch.int8, max_memory=param.max_memory, no_split_module_classes=model._no_split_modules)
                        param.max_memory = None   
                if (param.gptq or param.inference_config["gptq"]):
                    config = transformers.AutoConfig.from_pretrained(param.gpt_model_path, trust_remote_code=gpt_kwargs["trust_remote_code"])
                    quantization_config = GPTQConfig(bits=config.quantization_config.get("bits", 4), disable_exllama=True)  
                ret.gpt_model = transformers.AutoModelForCausalLM.from_pretrained(
                    param.gpt_model_path,
                    trust_remote_code=gpt_kwargs["trust_remote_code"],
                    low_cpu_mem_usage=gpt_kwargs["low_cpu_mem_usage"],
                    max_memory=param.max_memory,
                    device_map=device_map,
                    torch_dtype=eval("torch.{}".format(gpt_kwargs["torch_dtype"])) if gpt_kwargs["torch_dtype"] in ["bfloat16", "float16", "float32"] else "float16",
                    offload_folder=offload_folder,
                    offload_state_dict=offload_state_dict,
                    rope_scaling=rope_scaling,
                    quantization_config=quantization_config,
                    )
                if device_map != "auto" and not (param.offload or param.inference_config["offload"]) and not rope_scaling and not (param.bitsandbites or param.inference_config["bitsandbites"]) and not (param.gptq or param.inference_config["gptq"]):
                    ret.gpt_model.to(gpt_kwargs["device"])
            try:
                ret.gpt_tokenizer = AutoTokenizer.from_pretrained(param.gpt_model_path, use_fast=False, trust_remote_code=gpt_kwargs["trust_remote_code"])
            except:
                ret.gpt_tokenizer = AutoTokenizer.from_pretrained(param.gpt_model_path, use_fast=True, trust_remote_code=gpt_kwargs["trust_remote_code"])
            return ret
        except Exception as e:
            logging.error(f"from_pretrained() ERR: {e}")
            raise

    def gpt_token_num(self, sequence):
        """計算序列的 token 數量"""
        try:
            return len(self.gpt_tokenizer.encode(str(sequence), return_tensors="pt").cuda()[0])
        except Exception as e:
            logging.error(f"gpt_token_num() ERR: {e}")
            raise

    def encode(self, prompt, kwargs, truncation_length=None):
        """對輸入進行編碼"""
        try:
            input_ids = self.gpt_tokenizer.encode(str(prompt), return_tensors="pt", add_special_tokens=kwargs["add_special_tokens"])
            if not kwargs["add_bos_token"] and input_ids[0][0] == self.gpt_tokenizer.bos_token_id:
                input_ids = input_ids[:, 1:]
            if truncation_length is not None:
                input_ids = input_ids[:, -truncation_length:]
            return input_ids.cuda()
        except Exception as e:
            logging.error(f"encode() ERR: {e}")
            raise

    def decode(self, output_ids, kwargs):
        """對輸出進行解碼"""
        try:
            return self.gpt_tokenizer.decode(output_ids, kwargs["skip_special_tokens"])
        except Exception as e:
            logging.error(f"decode() ERR: {e}")
            raise

    def generate_fn(self, prompt, kwargs):
        """生成函數的配置"""
        try:
            sets = {}
            for s in [
                "num_return_sequences", "max_length",
                "max_new_tokens", "temperature", "top_p", "top_k", "typical_p", 
                "repetition_penalty", "encoder_repetition_penalty", "no_repeat_ngram_size", "min_length", 
                "do_sample", "penalty_alpha", "num_beams", "length_penalty", "early_stopping", "guidance_scale",
                ]:
                sets[s] = kwargs[s]
            if kwargs["negative_prompt_ids"] != "":
                sets["negative_prompt_ids"] = self.encode(kwargs["negative_prompt_ids"])        
            for s in ["epsilon_cutoff", "eta_cutoff"]:
                sets[s] = kwargs.get(s, 0) * 1e-4 if kwargs.get(s, 0) > 0 else 0
            if kwargs["ban_eos_token"]:
                sets["suppress_tokens"] = self.gpt_tokenizer.eos_token_id
            if kwargs["ban_token_customized"]:
                ban_token = [int(t) for t in kwargs["ban_token_customized"].split(",")]
                if ban_token > 0:
                    sets["suppress_tokens"] = sets["suppress_tokens"] + ban_token if sets.get("suppress_tokens") else ban_token        
            sets["use_cache"] = kwargs["use_cache"]
            input_ids = self.encode(prompt, kwargs, truncation_length=prompt_max_length(kwargs))
            if kwargs["max_new_tokens_auto"]:
                sets["max_new_tokens"] = kwargs["truncation_length"] - input_ids.shape[-1]
            sets.update({"inputs": input_ids})
            eos_token_ids = [self.gpt_tokenizer.eos_token_id] if self.gpt_tokenizer.eos_token_id is not None else []
            sets["eos_token_id"] = eos_token_ids
            sets["stopping_criteria"] = transformers.StoppingCriteriaList()
            sets["stopping_criteria"].append(StopCriteria())
            return sets, input_ids, eos_token_ids
        except Exception as e:
            logging.error(f"generate_fn() ERR: {e}")
            raise    

    def output_ids_fn(self, output_ids, input_ids, kwargs):
        """處理輸出 ID，將其轉換為文本"""
        try:
            new_tokens = len(output_ids) - len(input_ids[0])
            response = self.decode(output_ids[-new_tokens:], kwargs)
            if type(self.gpt_tokenizer) in [transformers.LlamaTokenizer, transformers.LlamaTokenizerFast] and len(output_ids) > 0:
                if self.gpt_tokenizer.convert_ids_to_tokens(int(output_ids[-new_tokens])).startswith("▁"):
                    response = " " + response
            return response
        except Exception as e:
            logging.error(f"output_ids_fn() ERR: {e}")
            raise

    def generate(self, prompt, kwargs):
        """生成文本"""
        try:
            sets, input_ids, _ = self.generate_fn(prompt, kwargs)
            with torch.no_grad():
                output_ids = self.gpt_model.generate(**sets)[0].cuda()
            yield self.output_ids_fn(output_ids, input_ids, kwargs)
        except Exception as e:
            logging.error(f"generate() ERR: {e}")
            raise

    def generate_stream(self, prompt, kwargs):
        """生成文本流"""
        def generator_fn(callback=None, *prompt, **kwargs):
            """實際生成文本的內部生成器函數"""
            try:
                kwargs["stopping_criteria"].append(Stream(callback=callback))
                clear_cache()
                with torch.no_grad():
                    self.gpt_model.generate(**kwargs)
            except Exception as e:
                logging.error(f"generator_fn() ERR: {e}")
                raise
        
        try:
            sets, input_ids, eos_token_ids = self.generate_fn(prompt, kwargs)
            with Generatior(generator_fn, [], sets, callback=None) as generator:
                for output_ids in generator:
                    yield self.output_ids_fn(output_ids, input_ids, kwargs)
                    if output_ids[-1] in eos_token_ids:
                        break
        except Exception as e:
            logging.error(f"generate_stream() ERR: {e}")
            raise

# 加載 transformers 或 autogptq 模型
def gptgtpq_load_gpt_model():
    try:
        param.gpt_model = gptgptq.from_pretrained()
        logging.info("Load GPT Model Success")
    except Exception as e:
        logging.error(f"gptgtpq_load_gpt_model() ERR: {e}")


def test_gpt():
    # 加載所有設定
    load_all_settings()

    # 加載模型
    #gptc_load_gpt_model()
    gptgtpq_load_gpt_model()
    
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