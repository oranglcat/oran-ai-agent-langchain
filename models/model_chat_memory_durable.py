import json
import os
from typing import Sequence

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, message_to_dict, messages_from_dict
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from langchain_core.runnables import RunnableWithMessageHistory

from model_init import deepseek_llm


class FileBasedChatMessageHistory(BaseChatMessageHistory):
    """
    基于文件的聊天消息历史记录
    """
    def __init__(self, storage_path: str, session_id: str):
        self.storage_path = storage_path
        self.session_id = session_id
        self.file_path = os.path.join(self.storage_path, self.session_id)
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)

    storage_path: str
    session_id: str
    def add_message(self, messages: Sequence[BaseMessage]):
        all_messages = list(self.messages)
        all_messages.append(messages)
        #将消息列表转换为字典列表
        messages_dict = [message_to_dict(message) for message in all_messages]
        #写入文件
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(messages_dict, f)

    @property
    def messages(self) -> list[BaseMessage]:
        # 读取文件，获取历史消息
        try:
            with open(self.file_path, "r", encoding="utf-8")as f:
                messages_dict =  json.load(f)
                history_message = messages_from_dict(messages_dict)
        except FileNotFoundError:
            return []
        return history_message

    def clear(self) -> None:
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump([], f)



prompt =  ChatPromptTemplate.from_messages(
    [
        ("system","请你根据历史对话内容回答用户问题，对话历史："),
        MessagesPlaceholder("history"),
        ("human","请回答如下问题：{input}")
    ]
)

def print_prompt(full_prompt):
    print("="*20 + full_prompt.to_string() + "="*20)
    return full_prompt

base_chain = prompt | print_prompt | deepseek_llm | StrOutputParser()


def get_history(session_id):
    return FileBasedChatMessageHistory("./chat_history",session_id)


memory_chain = RunnableWithMessageHistory(
    base_chain,
    get_history,
    input_messages_key="input",
    history_messages_key="history"
)



if __name__ == '__main__':
    session_id_config = {
        "configurable":{
            "session_id":"user001"
        }
    }

    res =  memory_chain.invoke({"input":"小明养了两只猫"},config=session_id_config)
    print("第1次执行：" + res)
    res = memory_chain.invoke({"input":"小霞养了三只兔子"},config=session_id_config)
    print("第2次执行：" + res)
    res = memory_chain.invoke({"input":"他们俩一共养了几只动物？"},config=session_id_config)
    print("第3次执行：" + res)