# XuanRAG 專案設定指南

本指南將引導您如何將 XuanRAG 專案放置到指定路徑，並透過 Docker 啟動專案。

## 將 XuanRAG 專案放置到指定路徑

首先，將 XuanRAG 專案放置到 `C:\Users\(User Name)` 路徑下。`(User Name)` 為您電腦的實際使用者名稱。

## 修改 `docker-compose.yml`

在進行下一步之前，需要對 `docker-compose.yml` 進行簡單的修改。使用文字編輯器開啟 `docker-compose.yml`，並找到以下行：

```yaml
      - ./module:/XuanRAG/module
```

在該行前加上 # 進行註解，如下所示：

```yaml
      # - ./module:/XuanRAG/module
```

## 開啟終端機

在終端機中，請依照以下指令操作

```bash
wsl # 切換到 Windows Subsystem for Linux 環境
cd XuanRAG # 切換到 XuanRAG 專案目錄
ln -s docker/{XuanRAG,docker-compose.yml} . # 建立到 docker 目錄中 XuanRAG 和 docker-compose.yml 的符號連結
docker compose up # 啟動 Docker 容器
```
