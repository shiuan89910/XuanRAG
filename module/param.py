import getpass
import os

all_setting_config = {}
proj_name = "XuanRAG"
try:
    user_name = os.getlogin()
    setting_save_folder_path = f"C:\\Users\\{user_name}\\{proj_name}\\setting" # Windows (Modify the path in Default.json)
except:
    user_name = getpass.getuser()
    if user_name == "root":
        setting_save_folder_path = f"/{proj_name}/setting" # Docker (Modify the path in Default.json)
    else:
        setting_save_folder_path = f"/home/{user_name}/{proj_name}/setting" # Ubuntu (Modify the path in Default.json)
setting_save_name = "Default.json"
setting_save_path = os.path.join(setting_save_folder_path, setting_save_name)

saveload_config = {}

embedding_config = {}
embedding_model = None
embedding_tokenizer = None

database_config = {}
query_client = None
query_1st_collection = None
query_2nd_collection = None

gpt_config = {}
bitsandbites_config = {}
autogptq_config = {}
ctransformers_config = {}
gpt_model = None
max_memory = {}
stop_all = False

exllama_config = {}

exllama2_config = {}

llamacpp_config = {}

inference_config = {}
gpt_model_path = None
load_type = None
gpt_type = None
bitsandbites = None
gptq = None
offload = None
gpt_inference_params = {}

prompt_config = {}
instruction = None
prompt = None
completion = None
context = None
conversation_list = []
prompt_all = None

webui_config = {}
host_ip = None
chat_mode = None
lang_code = None

history_config = {}
history_client = None
history_collection = None
history_subcollection = None
question_client = None
question_collection = None