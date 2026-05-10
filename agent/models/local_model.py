"""
Local model wrapper using langchain-ollama.
Used when lora.use_lora_adapter is true in agent_config.yaml.
Calls a local GGUF fine-tuned model via ChatOllama, same interface as CloudChefModel.

Requirements:
  1. Ollama must be installed and running (ollama serve)
  2. Register the model with Ollama:
       ollama create chef-lora -f lora_tuning/data/Modelfile
     Or use the base model directly:
       ollama pull qwen2.5:7b

Install dependency: pip install langchain-ollama
"""

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from utils.logger_handler import get_logger

logger = get_logger("ai_chef.local_model")

DEFAULT_SYSTEM_PROMPT = (
    "You are the dedicated AI chef assistant for this cooking app, "
    "specializing in Chinese and Western cuisine, nutrition, and food safety. "
    "Give users personalized cooking advice in a helpful, friendly tone."
)


class LocalChefModel:
    """
    Local model wrapper based on ChatOllama.

    Usage:
        model = LocalChefModel(lora_config)
        reply = model.chat("How do I make chicken breast tender?")
    """

    def __init__(self, lora_config: dict):
        model_name  = lora_config.get("ollama_model", "chef-lora")
        base_url    = lora_config.get("ollama_base_url", "http://localhost:11434")
        temperature = lora_config.get("temperature", 0.7)
        max_tokens  = lora_config.get("max_new_tokens", 512)

        self.llm = ChatOllama(
            model=model_name,
            base_url=base_url,
            temperature=temperature,
            num_predict=max_tokens,
        )
        logger.info(f"[Local mode] ChatOllama ready, model: {model_name} @ {base_url}")
