import os
from dotenv import load_dotenv

load_dotenv(override=True)
dashscope_api_key = os.getenv("DASHSCOPE_API_KEY")
dashscope_baseurl = os.getenv("DASHSCOPE_BASE_URL")

deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
deepseek_api_baseurl = os.getenv("DEEPSEEK_BASE_URL")