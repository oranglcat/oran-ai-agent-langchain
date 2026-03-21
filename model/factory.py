from abc import ABC, abstractmethod
from typing import Optional
from langchain.chat_models import init_chat_model
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

from env_utils import deepseek_api_baseurl, deepseek_api_key
from utils.config_handler import rag_conf


class BaseModelFactory(ABC):
    @abstractmethod
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        pass


class ChatModelFactory(BaseModelFactory):
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        return init_chat_model(model="deepseek-chat",
                               model_provider="deepseek",
                               api_key=deepseek_api_key,
                               base_url=deepseek_api_baseurl)


class EmbeddingsFactory(BaseModelFactory):
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        return DashScopeEmbeddings(model=rag_conf["embedding_model_name"])


chat_model = ChatModelFactory().generator()

embed_model = EmbeddingsFactory().generator()
