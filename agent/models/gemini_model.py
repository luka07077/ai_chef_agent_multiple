"""
Gemini cloud model wrapper (Google Gemini version).
Used when gemini.use_gemini is true in agent_config.yaml.
Uses ChatGoogleGenerativeAI from langchain-google-genai,
same structure as CloudChefModel / LocalChefModel.

Requirements:
  pip install -U langchain-google-genai
  Set the GEMINI_API_KEY environment variable
"""

import os
from langchain_google_genai import ChatGoogleGenerativeAI
from utils.logger_handler import get_logger

logger = get_logger("ai_chef.gemini_model")


class GeminiChefModel:
    """
    Cloud model wrapper based on ChatGoogleGenerativeAI (Google Gemini).
    After init, pass the .llm attribute to the agent.

    Usage:
        gemini = GeminiChefModel(gemini_config)
        agent_executor = create_agent(model=gemini.llm, tools=all_tools, ...)
    """

    def __init__(self, gemini_config: dict):
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY is missing from environment variables!")

        model_name  = gemini_config.get("main_model", "gemini-2.5-flash")
        temperature = gemini_config.get("main_temperature", 0.7)

        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=temperature,
        )
        logger.info(f"[Gemini mode] ChatGoogleGenerativeAI ready, model: {model_name}")
