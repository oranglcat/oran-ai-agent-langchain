from langchain.chat_models import init_chat_model
from langchain_community.llms import Tongyi
from env_utils import deepseek_api_key, deepseek_api_baseurl, dashscope_api_key, dashscope_baseurl


class ModelFactory:
    """
    模型工厂类，用于创建和管理不同的语言模型实例
    """

    @staticmethod
    def create_model(model_type: str, model_name: str = None):
        """
        根据模型类型创建相应的模型实例

        Args:
            model_type (str): 模型类型，支持 'deepseek' 和 'qwen'
            model_name (str, optional): 模型名称，默认为 None

        Returns:
            模型实例

        Raises:
            ValueError: 当不支持的模型类型时抛出异常
        """
        if model_type == "deepseek":
            return init_chat_model(
                model=model_name or "deepseek-chat",
                model_provider="deepseek",
                api_key=deepseek_api_key,
                base_url=deepseek_api_baseurl
            )
        elif model_type == "qwen":
            return Tongyi(
                model=model_name or "qwen-max",
                api_key=dashscope_api_key,
                base_url=dashscope_baseurl
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")


if __name__ == '__main__':
    deepseek_model = ModelFactory.create_model("deepseek")
    qwen_model = ModelFactory.create_model("qwen")
    print(qwen_model.invoke("你是谁"))
