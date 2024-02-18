# 0. 目錄
- [1. Docker 配置](#1-docker-配置)
  - [1.1. 安裝 Nvidia CUDA 驅動於 Windows](#11-安裝-nvidia-cuda-驅動於-windows)
  - [1.2. 透過終端安裝 WSL 於 Windows](#12-透過終端安裝-wsl-於-windows)
  - [1.3. 在 WSL 中安裝 CUDA Toolkit](#13-在-wsl-中安裝-cuda-toolkit)
  - [1.4. 在 WSL 中進行後續步驟](#14-在-wsl-中進行後續步驟)
  - [1.5. 在 Windows 中安裝 Docker](#15-在-windows-中安裝-docker)
  - [1.6. 在 WSL 中安裝 Docker 和 NVIDIA 容器工具包](#16-在-wsl-中安裝-docker-和-nvidia-容器工具包)
  - [1.7. 在 Windows 中創建 Dockerfile 並在 WSL 中構建鏡像](#17-在-windows-中創建-dockerfile-並在-wsl-中構建鏡像)
- [2. XuanRAG 專案設定指南](#2-xuanrag-專案設定指南)
  - [2.1. 將 XuanRAG 專案放置到指定路徑](#21-將-xuanrag-專案放置到指定路徑)
  - [2.2. 修改 docker-compose.yml](#22-修改-docker-composeyml)
  - [2.3. 開啟終端機](#23-開啟終端機)


# 1. Docker 配置
## 1.1. 安裝 Nvidia CUDA 驅動於 Windows
- [Nvidia CUDA 驅動下載頁面](https://www.nvidia.com/Download/index.aspx?lang=en-us)
- [Nvidia 顯卡驅動下載頁面](https://www.nvidia.com.tw/download/driverResults.aspx/193749/tw)
  - 選擇正確的驅動版本（11.8）進行安裝。
 
[回到目錄](#0-目錄)


## 1.2. 透過終端安裝 WSL 於 Windows
- [WSL 安裝指南官方頁面](https://learn.microsoft.com/zh-tw/windows/wsl/install)
  - 開啟終端。
  - 執行以下命令：
    ```bash
    wsl --install
    ```
  - 或者，透過（進階）設定，設置預設版本為 2，列出可用發行版，指定發行版安裝，執行以下命令：
    ```bash
    wsl --set-default-version 2
    wsl --list --online
    wsl --install -d "DistroName"
    ```
  - 創建 WSL（Ubuntu）用戶名與密碼。
  - 檢查結果（新終端），執行以下命令：
    ```bash
    wsl --list --verbose
    ```

[回到目錄](#0-目錄)


## 1.3. 在 WSL 中安裝 CUDA Toolkit
- [CUDA Toolkit 下載頁面](https://developer.nvidia.com/cuda-11-8-0-download-archive)
  - 下載並安裝 CUDA Toolkit，執行以下命令：
    ```bash
    wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin
    sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600
    wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-repo-wsl-ubuntu-11-8-local_11.8.0-1_amd64.deb
    sudo dpkg -i cuda-repo-wsl-ubuntu-11-8-local_11.8.0-1_amd64.deb
    sudo cp /var/cuda-repo-wsl-ubuntu-11-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
    sudo apt-get update
    sudo apt-get -y install cuda
    ```

[回到目錄](#0-目錄)


## 1.4. 在 WSL 中進行後續步驟
- 將 CUDA 路徑添加至 `.bashrc`，執行以下命令與添加複製路徑 (請逐步執行)：
  ```bash
  cd /usr/local/cuda/bin/
  pwd
  # 複製顯示的路徑
  cd
  nano .bashrc
  # 在文件末尾添加 export PATH=$PATH:(複製的路徑)
  # 使用 Ctrl+X，然後 Y 並按 Enter 離開 nano
  sudo reboot
  # 重啟終端
  wsl
  ```
- 檢查結果，執行以下命令：
  ```bash
  echo $PATH
  nvidia-smi
  nvcc --version
  ```

[回到目錄](#0-目錄)


## 1.5. 在 Windows 中安裝 Docker
- [Docker 在 Windows 的安裝指南](https://learn.microsoft.com/zh-tw/windows/wsl/tutorials/wsl-containers)
  - 安裝 Docker Desktop。
  - 在 Settings -> General -> 勾選 "Use the WSL 2 based engine"
  - 在 Settings -> Resources -> WSL Integration -> 啟用 "Ubuntu (Distro Name)"

[回到目錄](#0-目錄)


## 1.6. 在 WSL 中安裝 Docker 和 NVIDIA 容器工具包
- 相關鏈接：
  - [Docker 容器使用 GPU 教程](https://saturncloud.io/blog/how-to-use-gpu-from-a-docker-container-a-guide-for-data-scientists-and-software-engineers/)
  - [Docker 安裝 Ubuntu 教程](https://docs.docker.com/engine/install/ubuntu/)
  - [NVIDIA 容器工具包安裝指南](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
  - [NVIDIA CUDA Docker 鏡像](https://hub.docker.com/r/nvidia/cuda/tags)
    - [11.8.0-runtime-ubuntu22.04](https://hub.docker.com/layers/nvidia/cuda/11.8.0-runtime-ubuntu22.04/images/sha256-0b654e1fcb503532817c27b4088d0f121b86eaeb7676b2268dd4e448df333bea?context=explore)
    - [11.8.0-devel-ubuntu22.04](https://hub.docker.com/layers/nvidia/cuda/11.8.0-devel-ubuntu22.04/images/sha256-e395e2d30c63b6bd9ced510c8d09b1beb76bedaca4c527f3196348cce16a0da2?context=explore)
  - 安裝 Docker 和 NVIDIA 容器工具包，並配置用戶組，執行以下命令：
    ```bash
    # 重啟終端
    wsl

    sudo apt-get update
    sudo apt-get install ca-certificates curl gnupg
    sudo install -m 0755 -d /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
    sudo chmod a+r /etc/apt/keyrings/docker.gpg
    echo \
      "deb [arch="$(dpkg --print-architecture)" signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
      "$(. /etc/os-release && echo "$VERSION_CODENAME")" stable" | \
      sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
    sudo apt-get update
    sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
    ```
  - 檢查結果，執行以下命令：
    ```bash
    sudo docker run hello-world
    ```
  - 安裝 NVIDIA 容器工具包，配置 Docker 和 Containerd 以支援 GPU，執行以下命令：
    ```bash
    # 重啟終端
    wsl

    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
      && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
        sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
        sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list \
      && \
        sudo apt-get update
    sudo apt-get install -y nvidia-container-toolkit
    sudo nvidia-ctk runtime configure --runtime=docker
    sudo systemctl restart docker
    sudo nvidia-ctk runtime configure --runtime=containerd
    sudo systemctl restart containerd
     
    # 以下為可選配置
    sudo nvidia-ctk runtime configure --runtime=crio
    sudo systemctl restart crio
    ```
  - 將當前用戶添加至 Docker 群組，以免每次都需要 sudo，執行以下命令：
    ```bash
    sudo groupadd docker
    sudo usermod -aG docker $USER
    newgrp docker
    ```
  - 檢查 GPU 支援，執行以下命令：
    ```bash
    docker run --gpus all nvidia/cuda:11.8.0-runtime-ubuntu22.04 nvidia-smi
    ```

[回到目錄](#0-目錄)

    
## 1.7. 在 Windows 中創建 Dockerfile 並在 WSL 中構建鏡像
- 將`XuanRAG`專案放置於 Windows 路徑 C:\Users\(User Name)。
- 使用以下命令啟動：
  ```bash
  # 重啟終端
  wsl
  
  cd XuanRAG
  ln -s docker/{XuanRAG,docker-compose.yml} .
  docker compose up
  ```
- 將鏡像上傳至 Hub（可選）。

[回到目錄](#0-目錄)



# 2. XuanRAG 專案設定指南
本指南將引導您如何將 XuanRAG 專案放置到指定路徑，並透過 Docker 啟動專案。

[回到目錄](#0-目錄)


## 2.1. 將 XuanRAG 專案放置到指定路徑
首先，將 XuanRAG 專案放置到 `C:\Users\(User Name)` 路徑下。`(User Name)` 為您電腦的實際使用者名稱。

[回到目錄](#0-目錄)


## 2.2. 修改 docker-compose.yml
在進行下一步之前，需要對 `docker-compose.yml` 進行簡單的修改。使用文字編輯器開啟 `docker-compose.yml`，並找到以下行：
```yaml
      - ./module:/XuanRAG/module
```

在該行前加上 # 進行註解，如下所示：
```yaml
      # - ./module:/XuanRAG/module
```

[回到目錄](#0-目錄)


## 2.3. 開啟終端機
在終端機中，請依照以下指令操作
```bash
wsl # 切換到 Windows Subsystem for Linux 環境
cd XuanRAG # 切換到 XuanRAG 專案目錄
ln -s docker/{XuanRAG,docker-compose.yml} . # 建立到 docker 目錄中 XuanRAG 和 docker-compose.yml 的符號連結
docker compose up # 啟動 Docker 容器
```

[回到目錄](#0-目錄)
