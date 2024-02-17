# 安裝與入門指南

## 1. 安裝 Conda
首先，安裝 Conda 環境管理器。推薦使用 Miniconda，因為它比 Anaconda 更輕量。可以從以下連結下載安裝：
[Miniconda](https://docs.anaconda.com/free/miniconda/index.html)

## 2. 建立 conda 環境
接著，使用以下命令建立一個新的 conda 環境並啟動他。此處以`XuanRAG`做為環境名稱，並安裝了 Python 3.10.9 版本。

```bash
conda create -n XuanRAG python=3.10.9
conda activate XuanRAG
```

## 3. 安裝 git 與 pytorch
透過以下命令在環境中安裝 Git 和 PyTorch。這裡安裝的是 PyTorch 2.0.1 版本，並確保相容於 CUDA 11.8。
P.S. 如果你需要安裝最新版本的 PyTorch，可以使用註解掉的命令行。

```bash
conda install -c anaconda git
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia
#conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

## 4. 下載`XuanRAG`專案，並安裝 requirements 中的套件
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

## 5. 啟動 WebUI
最後，透過以下命令進入包含 WebUI 的模塊目錄並啟動 Web 界面。

```bash
cd module
python webui.py
```
