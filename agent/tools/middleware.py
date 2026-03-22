from typing import Callable
from langchain.agents import AgentState
from langchain.agents.middleware import wrap_tool_call, before_model, dynamic_prompt, ModelRequest
from langchain_core.messages import ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.runtime import Runtime
from langgraph.types import Command
from utils.logger_handler import logger
from utils.prompt_loader import get_report_prompt,get_system_prompt

@wrap_tool_call
def monitor_tool(
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command]
)-> ToolMessage | Command:
    logger.info(f"【tool_monitor】调用工具：{request.tool_call["name"]}")
    logger.info(f"【tool_monitor】传入参数：{request.tool_call["args"]}")

    if request.tool_call["name"] == "fill_context_for_report":
        request.runtime.context["report"] = True
    try:
        result = handler(request)
        logger.info(f"【tool_monitor】工具{request.tool_call["name"]}调用成功")
        return result
    except Exception as e:
        logger.error(f"【tool_monitor】调用工具{request.tool_call["name"]}异常：{str(e)}")
        raise e


@before_model
def log_before_model(
        state: AgentState,
        runtime: Runtime
):
    logger.info(f"【before_model】即将调用模型，带有{len(state['messages'])}条消息")

    logger.debug(f"【before_model】{type(state['messages'][-1]).__name__} | 消息：{state['messages'][-1]}")

    return None

@dynamic_prompt  #每一次生成提示词前，调用该函数 判断是否需要生成报告 动态修改提示词
def report_prompt_switch(request: ModelRequest):
    if request.runtime.context.get("report",False) == True:
        return get_report_prompt()

    return get_system_prompt()
