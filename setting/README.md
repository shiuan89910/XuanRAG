# 0. 目錄
- [1. 增強檢索大型語言模型 (RAG LLM) .json 配置檔](#1-增強檢索大型語言模型-rag-llm-json-配置檔)
  - [1.1. 配置區塊 (Configuration Blocks)](#11-配置區塊-configuration-blocks)
    - [1.1.1. saveload_config](#111-saveload_config)
    - [1.1.2. embedding_config](#112-embedding_config)
    - [1.1.3. database_config](#113-database_config)
    - [1.1.4. gpt_config](#114-gpt_config)
    - [1.1.5. 大型語言模型-量化模型載入 (Loading)](#115-大型語言模型-量化模型載入-loading)
      - `bitsandbites_config`
      - `gptq_config`
      - `autogptq_config`
      - `ctransformers_config`
      - `exllama_config`
      - `exllama2_config`
      - `llamacpp_config`
    - [1.1.6. inference_config](#116-inference_config)
    - [1.1.7. prompt_config](#117-prompt_config)
    - [1.1.8. webui_config](#118-webui_config)
    - [1.1.9. history_config](#119-history_config)
  - [P.S.](#ps)
    - **Prompt 目錄**
    - **OS 目錄**


# 1. 增強檢索大型語言模型 (RAG LLM) .json 配置檔
此配置檔案旨在提供增強檢索大型語言模型 (RAG LLM) 需要的各項配置參數，以便於使用者快速部署和調整模型性能。


## 1.1. 配置區塊 (Configuration Blocks)
配置文件包含以下幾個主要配置區塊，每個區塊都扮演著不同的角色，針對不同的模型和應用需求提供配置支持。

### 1.1.1. saveload_config
- **外部領域知識 (External Domain Knowledge):** 用於指定如何載入和儲存模型，以及如何與外部知識領域進行交互。

### 1.1.2. embedding_config
- **嵌入模型 (Embedding Model):** 配置與嵌入向量生成相關的模型參數，以支持增強檢索功能。

### 1.1.3. database_config
- **增強檢索向量資料庫參數 (RAG Vector Database):** 設定用於儲存和檢索向量的資料庫參數。
  - P.S. distance_fn 可設定參數："l2" 或 "ip" 或 "cosine"

### 1.1.4. gpt_config
- **大型語言模型-預訓練模型載入 (Loading):** 配置如何載入預訓練的大型語言模型。
  - P.S. gpu_max_memory 與 cpu_max_memory 可設定參數類型："1GiB" 或 "1024MiB" 其中的數值可自行做設定

### 1.1.5. 大型語言模型-量化模型載入 (Loading)
- `bitsandbites_config`
  - P.S. nb_4bit_quant_type 可設定參數："nf4" 或 "fp4"
- `gptq_config`
- `autogptq_config`
- `ctransformers_config`
  - P.S. model_type 可設定參數：null 或 "llama"
- `exllama_config`
- `exllama2_config`
- `llamacpp_config`

這些配置區塊用於細節化控制量化模型的載入過程，以優化模型的記憶體和運算效率。

### 1.1.6. inference_config
- **大型語言模型-推理 (Inference):** 配置模型推理過程中的參數，包括批次大小、記憶體管理等。
  - P.S. load_type 可設定參數："gpt" 或 "gptc" 或 "exllama" 或 "exllama2" 或 "llamacpp"
  - P.S. gpt_type 可設定參數："gpt" 或 "autogptq"

### 1.1.7. prompt_config
- **大型語言模型-提示 (Prompt):** 定義如何生成和使用提示，以引導模型產生特定類型的回應。

### 1.1.8. webui_config
- **使用者介面參數設定:** 配置與使用者介面相關的參數，如主機 IP、語言設定等。

### 1.1.9. history_config
- **歷史訊息資料庫 (History Database):** 設定用於存儲歷史訊息的資料庫名稱和集合名稱。


## P.S.
- **Prompt 目錄:** 此目錄中包含不同大型語言模型-提示的範式。使用者可以根據不同的模型需求，選擇相應的提示範式並將其配置區塊更新到原始`.json`配置檔中。這樣的設計旨在提供一種靈活的方式來自定義和優化模型的提示策略，從而改善模型的輸出質量和相關性。

- **OS 目錄:** 包含不同作業系統 (OS) 檔案路徑的範式。不同的作業系統對檔案路徑的表示方式可能有所不同，因此這個目錄旨在提供一套統一的指南，幫助使用者根據他們的操作系統調整`.json`配置檔中的檔案路徑設定。這確保了配置檔的跨平台兼容性，使得無論在哪種作業系統上，用戶都能夠順利部署和使用增強檢索大型語言模型 (RAG LLM)。
