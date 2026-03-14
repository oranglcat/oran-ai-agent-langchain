from langchain_core.messages import SystemMessage, HumanMessage
from pyexpat.errors import messages

from model_init import deepseek_llm

system_prompt = SystemMessage(content="你是一个旅游规划助手，专业提供旅游建议和规划服务。"
                      "你可以通过逐步提问，引导用户提供旅游地点和预算，根据用户输入的城市，推荐著名的旅游景点,根据用户的预算，调整旅游规划和建议。"
                      "在对话一开始，首先表明自己的身份,随后根据城市名称，列举出著名的旅游景点并根据用户预算调整住宿、餐饮、交通等建议，确保整体费用在预算范围内")
messages = []
messages.append(system_prompt)

user_prompt = HumanMessage(content="你好，我是oran")

messages.append(user_prompt)

resp = deepseek_llm.stream(messages)
print(type(resp))
for chunk in resp:
    print(chunk,end="",flush=True)
