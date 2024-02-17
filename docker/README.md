# 1. Docker 配置
## 1.1. 安裝 Nvidia CUDA 驅動於 Windows
- [Nvidia CUDA 驅動下載頁面](https://www.nvidia.com/Download/index.aspx?lang=en-us) 或 [Nvidia 顯卡驅動下載頁面](https://www.nvidia.com.tw/download/driverResults.aspx/193749/tw)
  - 選擇正確的驅動版本（11.8）進行安裝。


## 1.2. 透過終端安裝 WSL 於 Windows
- [WSL 安裝指南官方頁面](https://learn.microsoft.com/zh-tw/windows/wsl/install)
  - 開啟終端。
  - 執行 `wsl --install`。
  - 或者（進階設定）設置預設版本為 2，列出可用發行版，指定發行版安裝：
    ``` bash
    wsl --set-default-version 2
    wsl --list --online
    wsl --install -d "DistroName"
    ```
  - 創建 WSL（Ubuntu）用戶名與密碼。
  - 檢查（新終端）`wsl --list --verbose`。


## 1.3. 在 WSL 中安裝 CUDA Toolkit
- [CUDA Toolkit 下載頁面](https://developer.nvidia.com/cuda-11-8-0-download-archive)
  - 下載並安裝 CUDA Toolkit：
    ```bash
    wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin
    sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600
    wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-repo-wsl-ubuntu-11-8-local_11.8.0-1_amd64.deb
    sudo dpkg -i cuda-repo-wsl-ubuntu-11-8-local_11.8.0-1_amd64.deb
    sudo cp /var/cuda-repo-wsl-ubuntu-11-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
    sudo apt-get update
    sudo apt-get -y install cuda
    ```

## 1.4. 在 WSL 中進行後續步驟
- 將 CUDA 路徑添加至 `.bashrc`：
  ```bash
  cd /usr/local/cuda/bin/
  pwd # 複製顯示的路徑
  cd
  nano .bashrc # 在文件末尾添加：export PATH=$PATH:(複製的路徑)
  # 使用 Ctrl+X，然後 Y 並按 Enter 離開 nano
  sudo reboot # 重啟
  ```



# 2. XuanRAG 專案設定指南
本指南將引導您如何將 XuanRAG 專案放置到指定路徑，並透過 Docker 啟動專案。


## 2.1. 將 XuanRAG 專案放置到指定路徑
首先，將 XuanRAG 專案放置到 `C:\Users\(User Name)` 路徑下。`(User Name)` 為您電腦的實際使用者名稱。


## 2.2. 修改 `docker-compose.yml`
在進行下一步之前，需要對 `docker-compose.yml` 進行簡單的修改。使用文字編輯器開啟 `docker-compose.yml`，並找到以下行：
```yaml
      - ./module:/XuanRAG/module
```

在該行前加上 # 進行註解，如下所示：
```yaml
      # - ./module:/XuanRAG/module
```


## 2.3. 開啟終端機
在終端機中，請依照以下指令操作
```bash
wsl # 切換到 Windows Subsystem for Linux 環境
cd XuanRAG # 切換到 XuanRAG 專案目錄
ln -s docker/{XuanRAG,docker-compose.yml} . # 建立到 docker 目錄中 XuanRAG 和 docker-compose.yml 的符號連結
docker compose up # 啟動 Docker 容器
```
