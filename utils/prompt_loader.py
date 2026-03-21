from config_handler import prompts_conf
from utils.logger_handler import logger


def get_system_prompt():
    try:
        SYSTEM_PROMPT_PATH = prompts_conf["system_prompt_path"]
    except KeyError as e:
        logger.error(f"[获取系统提示]系统提示配置项不存在，请检查配置文件")
        raise e
    try:
        return open(SYSTEM_PROMPT_PATH,"r",encoding="utf-8").read()
    except Exception as e:
        logger.error(f"[获取系统提示]解析系统提示词失败")
        raise e


def get_rag_prompt():
    try:
        RAG_PROMPT_PATH = prompts_conf["rag_prompt_path"]
    except KeyError as e:
        logger.error(f"[获取系统提示]系统提示配置项不存在，请检查配置文件")
        raise e
    try:
        return open(RAG_PROMPT_PATH,"r",encoding="utf-8").read()
    except Exception as e:
        logger.error(f"[获取系统提示]解析RAG提示词失败")
        raise e


def get_report_prompt():
    try:
        REPORT_PROMPT_PATH = prompts_conf["report_prompt_path"]
    except KeyError as e:
        logger.error(f"[获取系统提示]系统提示配置项不存在，请检查配置文件")
        raise e
    try:
        return open(REPORT_PROMPT_PATH,"r",encoding="utf-8").read()
    except Exception as e:
        logger.error(f"[获取系统提示]解析生成报告提示词失败")
        raise e