from langchain.agents import create_agent
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
            middleware= [report_prompt_switch,log_before_model,monitor_tool]
        )


    def excute_stream(self,query: str):

        user_input = {
            "messages":[
                {
                    "role": "user",
                    "content": query
                }
            ]
        }


        for chunk in  self.agent.stream(user_input,stream_mode="values",context={"report":False}):
            lasted_msg =  chunk["messages"][-1]
            if lasted_msg:
                yield lasted_msg.content.strip() + "\n"


if __name__ == '__main__':
    agent = ReactAgent()
    for res in agent.excute_stream("扫地机器人在我所在地区的气温下如何保养"):
        print(res,end="",flush=True)