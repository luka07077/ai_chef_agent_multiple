import os
import sys
from typing import Literal

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from langchain_community.chat_models import ChatTongyi
from langchain_core.messages import SystemMessage, HumanMessage

from conf import get_prompt_config, get_agent_config
from utils.logger_handler import get_logger

"""
Multi-Agent Router (Orchestrator & Router)
==========================================
Implements a layered routing architecture: Orchestrator → specialized agent

┌─────────────────────────────────────────┐
│           User Input                     │
└──────────────┬──────────────────────────┘
               ↓
┌─────────────────────────────────────────┐
│   Orchestrator (intent classification)   │
│   Model: qwen-turbo (lightweight+fast)   │
│   Output: recipe / health / fridge / general│
└──────┬───────┬──────────┬───────────────┘
       ↓       ↓          ↓          ↓
  ┌────────┐ ┌───────┐ ┌───────┐ ┌─────────┐
  │Recipe  │ │Health │ │Fridge │ │General  │
  │Expert  │ │Advisor│ │Helper │ │Chef     │
  └────────┘ └───────┘ └───────┘ └─────────┘
       ↓       ↓          ↓          ↓
  (each uses its own system prompt, all share the same tools and agent_executor)

Design notes:
  - The underlying execution layer (MCP tools, RAG, agent_executor) is fully shared — no duplication
  - Only the system prompt differs per agent, reflecting each one's specialty
  - Orchestrator uses a lightweight LLM to keep routing fast and cheap
  - All routing decisions are logged for observability
"""

logger = get_logger("ai_chef.router")

# Intent type definition
IntentType = Literal["recipe", "health", "fridge", "general"]

# Intent → config key in prompt_config.yaml
_INTENT_PROMPT_MAP: dict[str, str] = {
    "recipe":  "recipe_agent_prompt",
    "health":  "health_agent_prompt",
    "fridge":  "fridge_agent_prompt",
    "general": "chef_system_prompt",
}

# Intent → user-visible agent label
INTENT_LABELS: dict[str, str] = {
    "recipe":  "🍳 Recipe Expert",
    "health":  "💊 Health & Nutrition Advisor",
    "fridge":  "❄️ Fridge Manager",
    "general": "👨‍🍳 Chef Assistant",
}

# Global singleton for the router LLM (lazy-loaded to avoid repeated init)
_router_llm: ChatTongyi | None = None


def get_router_llm() -> ChatTongyi:
    """
    Get (or lazy-load) the lightweight Orchestrator LLM.
    Uses evaluator_model (qwen-turbo): fast, cheap, good enough for intent classification.
    temperature=0 makes the classification deterministic.
    """
    global _router_llm
    if _router_llm is None:
        config = get_agent_config()
        api_key = os.environ.get("DASHSCOPE_API_KEY")
        if not api_key:
            logger.warning("[Router] DASHSCOPE_API_KEY not set — router LLM unavailable, will use fallback")
            return None
        _router_llm = ChatTongyi(
            model_name=config["llm"]["evaluator_model"],  # qwen-turbo
            dashscope_api_key=api_key,
            temperature=0,
        )
        logger.info(f"[Router] Orchestrator LLM initialized: {config['llm']['evaluator_model']}")
    return _router_llm


def classify_intent(user_input: str) -> IntentType:
    """
    Core Orchestrator method: classify the user's intent to decide where to route.

    Classification logic:
      - recipe  → recipe recommendations, cooking methods, ingredient combos, what to eat today
      - health  → nutrition queries, healthy eating, allergens, health reports, special diets
      - fridge  → fridge inventory, expiring items, grocery orders, adding ingredients
      - general → general greetings or unclear requests

    Args:
        user_input: raw user input (already passed through guardrails)

    Returns:
        IntentType: "recipe" | "health" | "fridge" | "general"
    """
    prompts = get_prompt_config()
    system_prompt = prompts.get(
        "intent_classifier_prompt",
        "Classify the user input as: recipe/health/fridge/general. Only output the label, nothing else."
    )

    llm = get_router_llm()
    if llm is None:
        logger.warning("[Router] No LLM available, defaulting intent to 'general'")
        return "general"

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_input[:300]),  # truncate to save tokens — full length not needed for classification
    ]

    try:
        response = llm.invoke(messages)
        raw = response.content.strip().lower()

        for key in ["recipe", "health", "fridge", "general"]:
            if key in raw:
                logger.info(
                    f"[Router] Routing decision: '{user_input[:40]}' "
                    f"→ {INTENT_LABELS[key]}"
                )
                return key  # type: ignore[return-value]

        logger.warning(f"[Router] Intent not recognized (raw='{raw}'), falling back to general")

    except Exception as e:
        logger.warning(f"[Router] Orchestrator call failed, falling back to general: {e}")

    return "general"


def has_order_intent(user_input: str) -> bool:
    """
    Use the LLM to check if the user wants to buy/order groceries — used to trigger the HITL popup.
    Uses qwen-turbo for lightweight classification, returns True/False.
    Returns False on any error (conservative — don't block unrelated requests by mistake).
    """
    llm = get_router_llm()
    if llm is None:
        return False  # fail-open: no key, skip order detection

    messages = [
        SystemMessage(content=(
            "Does the user input contain an intent to buy, order, or purchase ingredients? "
            "Only output yes or no, nothing else."
        )),
        HumanMessage(content=user_input[:200]),
    ]
    try:
        raw = llm.invoke(messages).content.strip().lower()
        result = "yes" in raw
        logger.info(f"[Router] Order intent check: '{user_input[:40]}' → {'yes' if result else 'no'}")
        return result
    except Exception as e:
        logger.warning(f"[Router] Order intent check failed, defaulting to no: {e}")
        return False


def get_specialized_prompt(intent: IntentType) -> str:
    """
    Return the system prompt for the specialized agent that matches the given intent.
    Falls back to the general chef system prompt if the specialized one doesn't exist.

    Args:
        intent: intent type returned by classify_intent()

    Returns:
        system prompt string
    """
    prompts = get_prompt_config()
    prompt_key = _INTENT_PROMPT_MAP.get(intent, "chef_system_prompt")

    specialized = prompts.get(prompt_key)
    if specialized:
        return specialized

    # Fallback: use the general prompt
    logger.warning(f"[Router] Specialized prompt '{prompt_key}' not found, using general prompt")
    return prompts.get("chef_system_prompt", "You are a professional AI chef assistant.")
