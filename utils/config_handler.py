import yaml

from utils.path_tool import get_abs_path


def load_rag_config(file_path: str="config/rag.yml",encoding="utf-8"):
    file_path = get_abs_path(file_path)
    with open(file_path, "r",encoding = encoding) as f:
        return yaml.load(f,Loader=yaml.FullLoader)

def load_chroma_config(file_path: str="config/chroma.yml",encoding="utf-8"):
    file_path = get_abs_path(file_path)
    with open(file_path, "r",encoding = encoding) as f:
        return yaml.load(f,Loader=yaml.FullLoader)

def load_prompts_config(file_path: str="config/prompts.yml",encoding="utf-8"):
    file_path = get_abs_path(file_path)
    with open(file_path, "r",encoding = encoding) as f:
        return yaml.load(f,Loader=yaml.FullLoader)

def load_agent_config(file_path: str="config/agent.yml",encoding="utf-8"):
    file_path = get_abs_path(file_path)
    with open(file_path, "r",encoding = encoding) as f:
        return yaml.load(f,Loader=yaml.FullLoader)

def load_tools_config(file_path: str="config/tools.yml",encoding="utf-8"):
    file_path = get_abs_path(file_path)
    with open(file_path, "r",encoding = encoding) as f:
        return yaml.load(f,Loader=yaml.FullLoader)

rag_conf = load_rag_config()
chroma_conf = load_chroma_config()
prompts_conf = load_prompts_config()
agent_conf = load_agent_config()
tool_conf = load_tools_config()

if __name__ == '__main__':
    print(rag_conf["embedding_model_name"])