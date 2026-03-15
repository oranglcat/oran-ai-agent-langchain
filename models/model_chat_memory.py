from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from typing_extensions import TypedDict, Annotated
from langgraph.graph.message import add_messages
from model_init import deepseek_llm


# 1.定义状态
class AgentState(TypedDict):
    """
    定义图的状态结构
    message：消息列表，使用add_message更新消息
    """
    messages: Annotated[list, add_messages]


# 2.提示词模板
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "请你根据历史对话回答问题,对话历史："),
        MessagesPlaceholder("messages")
    ]
)


# 3.定义节点函数
def chat_node(state: AgentState):
    """ai对话节点"""
    chain = prompt | deepseek_llm
    return {"messages": [chain.invoke(state["messages"])]}


# 4.定义graph工作流
def create_graph():
    """
   创建并编译 StateGraph，添加 checkpointer 实现记忆功能
   """
    # 创建图构建器
    graph_builder = StateGraph(AgentState)

    # 添加聊天机器人节点
    graph_builder.add_node("chatbot", chat_node)

    # 添加起始边
    graph_builder.add_edge(START, "chatbot")

    # 添加结束边
    graph_builder.add_edge("chatbot", END)

    # ==================== 关键：添加 MemorySaver 检查点 ====================
    memory = MemorySaver()

    # 编译图时传入 checkpointer
    return graph_builder.compile(checkpointer=memory)

if __name__ == '__main__':
    graph = create_graph()

    config = {
        "configurable":{
            "thread_id":"user001"
        }
    }

    user_input = "小明养了2只兔子"
    graph.invoke({"messages": [HumanMessage(content=user_input)]},config=  config)

    user_input = "小刚养了3只兔子"
    graph.invoke({"messages": [HumanMessage(content=user_input)]},config=  config)

    user_input = "我的名字是oran"
    graph.invoke({"messages": [HumanMessage(content=user_input)]},config=  config)

    user_input = "我的名字是什么？我刚刚告诉你了"
    res = graph.invoke({"messages": [HumanMessage(content=user_input)]},config= config)
    print(res["messages"][-1].content)