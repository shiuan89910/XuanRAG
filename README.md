# 0. 目錄
- [1. XuanRAG 專案簡介](#1-xuanrag-專案簡介)
  - [1.1. 專案概念圖](#11-專案概念圖)
  - [1.2. 使用者介面介紹](#12-使用者介面介紹)
  - [1.3. Text Embedding 概念圖](#13-text-embedding-概念圖)
  - [1.4. Text Embedding 使用者介面介紹](#14-text-embedding-使用者介面介紹)
  - [1.5. Semantic Search 概念圖](#15-semantic-search-概念圖)
    - [1.5.1. Chunk Strategy & Embedding](#151-chunk-strategy--embedding)
    - [1.5.2. Vector Database](#152-vector-database)
    - [1.5.3. Re-Ranking & Semantic Search](#153-re-ranking--semantic-search)
  - [1.6. Semantic Search 使用者介面介紹](#16-semantic-search-使用者介面介紹)
- [2. 專案流程圖 & 目錄結構圖](#2-專案流程圖--目錄結構圖)
  - [2.1. XuanRAG 專案流程圖](#21-xuanrag-專案流程圖)
    - [2.1.1. 模組描述](#211-模組描述)
    - [2.1.2. 數據流向](#212-數據流向)
    - [2.1.3. 配置](#213-配置)
    - [2.1.4. 使用方式](#214-使用方式)
  - [2.2. XuanRAG 目錄結構圖](#22-xuanrag-目錄結構圖)
- [3. 安裝與入門指南](#3-安裝與入門指南)
  - [3.1. 安裝 Conda](#31-安裝-conda)
  - [3.2. 建立 conda 環境](#32-建立-conda-環境)
  - [3.3. 安裝 git 與 pytorch](#33-安裝-git-與-pytorch)
  - [3.4. 下載XuanRAG專案，並安裝 requirements 中的套件](#34-下載xuanrag專案並安裝-requirements-中的套件)
  - [3.5. 下載 Embedding 與 GPT 模型](#35-下載-embedding-與-gpt-模型)
  - [3.6. 配置 .json 配置檔](#36-配置-json-配置檔)
  - [3.7. 數據相關檔案 Embedding](#37-數據相關檔案-embedding)
  - [3.8. 啟動 WebUI](#38-啟動-webui)
- [4. Docker 的安裝方式](#4-docker-的安裝方式)
- [5. 致謝](#5-致謝)


# 1. XuanRAG 專案簡介
實現了基於檢索增強的生成，提高了問答系統的準確性和靈活性。


# 1.1. 專案概念圖
![XuanRAG Concept Map](https://github.com/shiuan89910/XuanProjectData/blob/main/XuanRAG/xuanrag_concept_map.png)

[回到目錄](#0-目錄)


## 1.2. 使用者介面介紹
![XuanRAG UI Introduction](https://github.com/shiuan89910/XuanProjectData/blob/main/XuanRAG/xuanrag_ui_introduction.png)

[回到目錄](#0-目錄)


## 1.3. Text Embedding 概念圖
![Text Embedding Concept Map](https://github.com/shiuan89910/XuanProjectData/blob/main/XuanRAG/text_embedding_concept_map.png)

[回到目錄](#0-目錄)


## 1.4. Text Embedding 使用者介面介紹
![Text Embedding UI Introduction](https://github.com/shiuan89910/XuanProjectData/blob/main/XuanRAG/text_embedding_ui_introduction.png)

[回到目錄](#0-目錄)


## 1.5. Semantic Search 概念圖
### 1.5.1. Chunk Strategy & Embedding
![Semantic Search Concept Map 1](https://github.com/shiuan89910/XuanProjectData/blob/main/XuanRAG/semantic_search_concept_map_1.png)

[回到目錄](#0-目錄)

### 1.5.2. Vector Database
![Semantic Search Concept Map 2](https://github.com/shiuan89910/XuanProjectData/blob/main/XuanRAG/semantic_search_concept_map_2.png)

[回到目錄](#0-目錄)

### 1.5.3. Re-Ranking & Semantic Search
![Semantic Search Concept Map 3](https://github.com/shiuan89910/XuanProjectData/blob/main/XuanRAG/semantic_search_concept_map_3.png)

[回到目錄](#0-目錄)


## 1.6. Semantic Search 使用者介面介紹
![Semantic Search UI Introduction](https://github.com/shiuan89910/XuanProjectData/blob/main/XuanRAG/semantic_search_ui_introduction.png)

[回到目錄](#0-目錄)



# 2. 專案流程圖 & 目錄結構圖
## 2.1. XuanRAG 專案流程圖
![XuanRAG Project Flowchart](https://github.com/shiuan89910/XuanProjectData/blob/main/XuanRAG/xuanrag_project_flowchart.png)
此圖展示 LLM RAG（檢索增強式語言模型生成）專案的架構。每個模組都旨在處理系統的不同方面，確保模組化和易於維護。

[回到目錄](#0-目錄)

### 2.1.1. 模組描述
- `param.py`：定義整個專案中使用的參數。
- `gpt.py`：管理 GPT (Tansformer、CTransformer、AutoGPTQ) 模型的加載與推論。
- `llama.py`：管理 GPT (Exllama、LLamaCpp) 模型的加載與推論。
- `database.py`：處理數據庫操作。
- `history.py`：管理互動歷史的追踪和儲存。
- `embedding.py`：處理嵌入模型操作。
- `webui.py`：用於與系統互動的網頁用戶介面模組。
- `saveload.py`：包含儲存和加載檔案或數據的功能。
- `util.py`：提供整個專案中使用的實用功能。

[回到目錄](#0-目錄)

### 2.1.2. 數據流向
箭頭指示模組之間的數據流向和依賴關係。例如：param.py 是一個基礎模組，它向 gpt.py 和 llama.py 提供參數，這兩個模組又向 webui.py 提供問答互動的結果、embedding.py 模組與數據庫緊密合作以管理嵌入。

[回到目錄](#0-目錄)

### 2.1.3. 配置
.json config 文件被用來儲存配置細節，這些配置細節會被相應的模組加載和使用。

[回到目錄](#0-目錄)

### 2.1.4. 使用方式
透過 webui.py 來進行，它協調 gpt.py 和 llama.py 模型的使用，以及由 database.py 和 embedding.py 提供的必要數據處理。

[回到目錄](#0-目錄)


## 2.2. XuanRAG 目錄結構圖
```
XuanRAG
├── data                              # 用於 Embedding 的數據相關檔案
├── database                          # 數據集和數據相關檔案 Embedding 後儲存於數據庫的檔案
├── docker                            # Docker 配置檔，包含 XuanRAG 和 docker-compose 腳本
│   ├── XuanRAG
│   ├── docker-compose.yml
├── model                             # 存放模型相關檔案
│   ├── embedding                     # Embedding 模型檔案
│   ├── gpt                           # GPT 模型檔案
├── module                            # 專案核心模組
│   ├── templates                     # 網頁模板檔案
│   │   ├── databaseindexpage.html    # 數據庫索引頁面
│   │   ├── languagepage.html         # 語言設置頁面
│   │   ├── loadpage.html             # 加載頁面
│   │   ├── modepage.html             # 模式設置頁面
│   │   ├── semanticsearchpage.html   # 語義搜尋頁面
│   │   ├── webui.html                # 主要 Web UI 頁面
│   ├── database.py                   # 數據庫操作模組
│   ├── embedding.py                  # Embedding 處理模組
│   ├── gpt.py                        # GPT (Tansformer、CTransformer、AutoGPTQ) 模型操作模組
│   ├── history.py                    # 歷史追踪模組
│   ├── llama.py                      # GPT (Exllama、LLamaCpp) 模型操作模組
│   ├── param.py                      # 參數設置模組
│   ├── saveload.py                   # 儲存和加載檔案或數據功能模組
│   ├── util.py                       # 工具功能模組
│   ├── webui.py                      # Web 使用者界面模組
├── offload                           # 模型參數卸載檔案夾
├── setting                           # 配置檔案夾
│   ├── Default.json                  # 預設配置檔
│   ├── Chat.json                     # Chat 模式配置檔
│   ├── QA.json                       # QA 模式配置檔
├── requirements.txt                  # 專案依賴的 Python 套件列表
```
此目錄結構提供專案組織的直觀視圖，每個組件都被整理在合適的位置，方便開發者快速導航和理解專案的構建。

[回到目錄](#0-目錄)



# 3. 安裝與入門指南
## 3.1. 安裝 Conda
首先，安裝 Conda 環境管理器。推薦使用 Miniconda，因為它比 Anaconda 更輕量。可以從以下連結下載安裝：
[Miniconda](https://docs.anaconda.com/free/miniconda/index.html)

[回到目錄](#0-目錄)


## 3.2. 建立 conda 環境
接著，使用以下命令建立一個新的 conda 環境並啟動他。此處以`XuanRAG`做為環境名稱，並安裝了 Python 3.10.9 版本。
```bash
conda create -n XuanRAG python=3.10.9
conda activate XuanRAG
```

[回到目錄](#0-目錄)


## 3.3. 安裝 git 與 pytorch
透過以下命令在環境中安裝 Git 和 PyTorch。這裡安裝的是 PyTorch 2.0.1 版本，並確保相容於 CUDA 11.8。
P.S. 如果你需要安裝最新版本的 PyTorch，可以使用註解掉的命令行。
```bash
conda install -c anaconda git
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia
#conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

[回到目錄](#0-目錄)


## 3.4. 下載`XuanRAG`專案，並安裝 requirements 中的套件
下載以下連結的專案，並置於根目錄底下：
[XuanRAG 專案](https://github.com/shiuan89910/XuanRAG/archive/refs/heads/main.zip)
>根目錄的位置
>Windows: C:\Users\使用者名稱
>Ubuntu: /home/使用者名稱

再透過以下命令進入專案目錄，此處為`XuanRAG`，並安裝所有依賴。
```bash
cd XuanRAG
pip install -r requirements.txt
```

[回到目錄](#0-目錄)


## 3.5. 下載 Embedding 與 GPT 模型
Embedding 請參照：[Embedding 模型](https://github.com/shiuan89910/XuanRAG/blob/main/model/gpt/README.md)
GPT 請參照：[GPT 模型](https://github.com/shiuan89910/XuanRAG/blob/main/model/embedding/README.md)

[回到目錄](#0-目錄)


## 3.6. 配置 .json 配置檔
.json 配置檔請參照：[.json 配置檔](https://github.com/shiuan89910/XuanRAG/blob/main/setting/README.md)

[回到目錄](#0-目錄)


## 3.7. 數據相關檔案 Embedding
數據相關檔案 Embedding 的方式請參照：[檔案 Embedding 的方式](https://github.com/shiuan89910/XuanRAG/blob/main/data/README.md)

[回到目錄](#0-目錄)


## 3.8. 啟動 WebUI
最後，透過以下命令進入包含 WebUI 的模塊目錄並啟動 Web 界面。
```bash
cd module
python webui.py
```

[回到目錄](#0-目錄)



# 4. Docker 的安裝方式
Docker 安裝方式請參照：[Docker 安裝方式](https://github.com/shiuan89910/XuanRAG/blob/main/docker/README.md)

[回到目錄](#0-目錄)



# 5. 致謝
本專案 `gpt.py`、`llama.py`、`util.py` 部分程式碼參考了 [oobabooga](https://github.com/oobabooga/text-generation-webui) 的實作，特此致謝。原始程式碼採用 [AGPL-3.0 license](https://github.com/shiuan89910/XuanRAG/tree/main?tab=GPL-3.0-1-ov-file#GPL-3.0-1-ov-file) 許可證。

[回到目錄](#0-目錄)
