from langchain.chat_models import init_chat_model
from langchain_community.llms import Tongyi
from env_utils import deepseek_api_key, deepseek_api_baseurl, dashscope_api_key, dashscope_baseurl

deepseek_llm = init_chat_model(
    model="deepseek-chat",
    model_provider="deepseek",
    api_key = deepseek_api_key,
    base_url = deepseek_api_baseurl
)

qwen_llm = Tongyi(
    model="qwen-chat",
    api_key = dashscope_api_key,
    base_url = dashscope_baseurl
)