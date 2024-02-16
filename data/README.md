# 外部領域知識 (External Domain Knowledge) 的檔案路徑

外部領域知識是通過嵌入模型 (Embedding Model) 轉換成向量資料的，這一過程允許模型理解和處理這些知識。以下介紹了如何將外部領域知識的文本轉換為模型能夠處理的向量資料。

## 轉換方式

### 1. WebUI 轉換流程

在透過WebUI進行轉換之前，請注意以下步驟：

- **注意:** 轉換前請先備份將要取代的同檔名向量資料庫。檔名為目錄`XuanRAG/setting/.json`檔中的`"database_config"`的`"query_database_name"`。

**步驟:**

1. 將外部領域知識庫存成 `.txt` 檔，檔案名稱需與目錄`XuanRAG/setting/.json`檔中的`"saveload_config"`的`"data_name"`相同。
2. 在 WebUI 中選擇 `"Load Page"` 頁面。
3. 點擊 `"Load All Settings"` 進行設定刷新。
4. 點擊 `"Create Load Collection"` 進行轉換。

### 2. XuanRAG 模組中的 `database.py` 檔轉換流程

同樣，在進行轉換之前，請確保已備份將要取代的同檔名向量資料庫。檔名為目錄`XuanRAG/setting/.json`檔中的`"database_config"`的`"query_database_name"`。

再將外部領域知識庫存成 `.txt` 檔，檔案名稱需與目錄`XuanRAG/setting/.json`檔中的`"saveload_config"`的`"data_name"`相同。

**執行以下 Python 腳本進行轉換:**

```python
from embedding import load_embedding_model
from database import create_load_query_collection, query_collection_add
from saveload import load_all_settings

load_all_settings()
load_embedding_model()
create_load_query_collection()
query_collection_add()
```
