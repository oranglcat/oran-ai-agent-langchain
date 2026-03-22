import os
import requests
from langchain_core.tools import tool
from rag.rag_service import RagSummarizeService
import random
from utils.config_handler import tool_conf, agent_conf
from utils.logger_handler import logger
from utils.path_tool import get_abs_path

rag = RagSummarizeService()

user_ids = ["1001", "1002", "1003", "1004", "1005", "1006", "1007", "1008", "1009", "1010"]

external_data = {}


@tool(description="从向量数据库中检索参考资料")
def ragSummarize(query: str) -> str:
    return rag.ragSummarize(query)


@tool(description="获取指定城市的天气信息，以字符串的形式输出")
def get_weather(city: str) -> str:
    api_key = os.getenv("AMAP_API_KEY")
    if not api_key:
        return "无法获取天气信息：缺少API密钥配置"

    url = "https://restapi.amap.com/v3/weather/weatherInfo"
    params = {
        "city": city,
        "key": api_key,
        "extensions": "base",
        "output": "JSON"
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data["status"] == "1" and data["lives"]:
                weather_info = data["lives"][0]
                return f"{weather_info['city']}天气：{weather_info['weather']}，温度{weather_info['temperature']}°C，风力{weather_info['windpower']}级"
            else:
                return f"无法获取{city}的天气信息"
        else:
            return f"获取天气信息失败，HTTP状态码：{response.status_code}"
    except Exception as e:
        return f"获取天气信息时发生错误：{str(e)}"


@tool(description="获取用户所在的城市名称，返回字符串形式")
def get_location() -> str:
    """
   获取用户所在的城市位置
   Returns:
       城市名称字符串
   """
    return random.choice(["北京", "上海", "广州", "深圳"])


@tool(description="获取用户的id，以纯字符串形式返回")
def get_user_id() -> str:
    return random.choice(user_ids)


def generate_external_data():
    """
    {
        "user_id": {
            "month" : {"特征": xxx, "效率": xxx, ...}
            "month" : {"特征": xxx, "效率": xxx, ...}
            "month" : {"特征": xxx, "效率": xxx, ...}
            ...
        },
        "user_id": {
            "month" : {"特征": xxx, "效率": xxx, ...}
            "month" : {"特征": xxx, "效率": xxx, ...}
            "month" : {"特征": xxx, "效率": xxx, ...}
            ...
        },
        "user_id": {
            "month" : {"特征": xxx, "效率": xxx, ...}
            "month" : {"特征": xxx, "效率": xxx, ...}
            "month" : {"特征": xxx, "效率": xxx, ...}
            ...
        },
        ...
    }
    :return:
    """
    if not external_data:
        external_data_path = get_abs_path(agent_conf["external_data_path"])

        if not os.path.exists(external_data_path):
            raise FileNotFoundError(f"外部数据文件{external_data_path}不存在")

        with open(external_data_path, "r", encoding="utf-8") as f:
            for line in f.readlines()[1:]:
                arr: list[str] = line.strip().split(",")

                user_id: str = arr[0].replace('"', "")
                feature: str = arr[1].replace('"', "")
                efficiency: str = arr[2].replace('"', "")
                consumables: str = arr[3].replace('"', "")
                comparison: str = arr[4].replace('"', "")
                time: str = arr[5].replace('"', "")

                if user_id not in external_data:
                    external_data[user_id] = {}

                external_data[user_id][time] = {
                    "特征": feature,
                    "效率": efficiency,
                    "耗材": consumables,
                    "对比": comparison,
                }


@tool(description="从外部系统中获取指定用户在指定月份的使用记录，以纯字符串形式返回， 如果未检索到返回空字符串")
def fetch_external_data(user_id: str, month: str) -> str:
    generate_external_data()

    try:
        return external_data[user_id][month]
    except KeyError:
        logger.warning(f"[fetch_external_data]未能检索到用户：{user_id}在{month}的使用记录数据")
        return ""


@tool(description="不接收任何参数，只返回一句话，调用后触发中间件自动为报告注入场景动态上下文，为后续提示词切换提供上下文信息")
def fill_context_for_report():
    return "【fill_context_for_report】 已被调用"



tools = [ragSummarize,fill_context_for_report,get_weather,get_location,get_user_id,fetch_external_data]


