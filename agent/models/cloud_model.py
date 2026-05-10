"""
Cloud model wrapper for ChatTongyi / DashScope.
Used when lora.use_lora_adapter is false (the default) in agent_config.yaml.
Wraps the Qwen model init, keeping the same interface as LocalChefModel.

Requirements:
  Set the DASHSCOPE_API_KEY environment variable
"""

import os
from langchain_community.chat_models import ChatTongyi
from utils.logger_handler import get_logger

logger = get_logger("ai_chef.cloud_model")


class CloudChefModel:
    """
    Cloud model wrapper based on ChatTongyi (Qwen).
    After init, pass the .llm attribute to the agent.

    Usage:
        cloud = CloudChefModel(llm_config)
        agent_executor = create_agent(model=cloud.llm, tools=all_tools, ...)
    """

    def __init__(self, llm_config: dict):
        api_key = os.environ.get("DASHSCOPE_API_KEY")
        if not api_key:
            raise ValueError("DASHSCOPE_API_KEY is missing from environment variables!")

        model_name  = llm_config.get("main_model", "qwen3-max")
        temperature = llm_config.get("main_temperature", 0.7)

        self.llm = ChatTongyi(
            model_name=model_name,
            dashscope_api_key=api_key,
            temperature=temperature,
        )
        logger.info(f"[Cloud mode] ChatTongyi ready, model: {model_name}")
