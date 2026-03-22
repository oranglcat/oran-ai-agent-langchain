from langchain.agents import create_agent
from langchain_core.stores import InMemoryStore
from langgraph.checkpoint.memory import InMemorySaver
from uuid import uuid4 as uuid4_str
from model.factory import chat_model
from utils.prompt_loader import get_system_prompt
from agent.tools.agent_tools import tools
from agent.tools.middleware import *


class ReactAgent:

    def __init__(self):
        self.agent = create_agent(
            model= chat_model,
            system_prompt= get_system_prompt(),
            tools= tools,
            middleware= [report_prompt_switch,log_before_model,monitor_tool],
            checkpointer= InMemorySaver()
        )


    def excute_stream(self,query: str, thread_id: str = None):

        # 如果没有提供thread_id，生成一个新的
        if thread_id is None:
            thread_id = str(uuid4_str())

        user_input = {
            "messages":[
                {
                    "role": "user",
                    "content": query
                }
            ]
        }


        for chunk in  self.agent.stream(user_input,stream_mode="values",context={"report":False},config={"thread_id":thread_id}):
            lasted_msg =  chunk["messages"][-1]
            if lasted_msg:
                # 检查消息类型，只返回助手的回复
                msg_type = type(lasted_msg).__name__
                if msg_type == "AIMessage" or msg_type == "AssistantMessage":
                    yield lasted_msg.content.strip() + "\n"


if __name__ == '__main__':
    agent = ReactAgent()
    for res in agent.excute_stream("扫地机器人在我所在地区的气温下如何保养，请为我输出结果报告"):
        print(res,end="",flush=True)