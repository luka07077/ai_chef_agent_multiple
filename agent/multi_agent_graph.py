import os
import sys
from typing import Annotated
from typing_extensions import TypedDict

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from langchain.agents import create_agent
from langchain_core.messages import BaseMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from agent.router import classify_intent, get_specialized_prompt
from agent.middleware import all_middleware
from utils.logger_handler import get_logger

"""
Multi-Agent LangGraph graph (multi_agent_graph)
=============================================
Architecture: Orchestrator intent routing → 4 specialized agent nodes

                 ┌────────────────────────────┐
                 │       MultiAgentState       │
                 │  messages + intent          │
                 └─────────────┬──────────────┘
                               ↓
                 ┌─────────────────────────────┐
                 │  Orchestrator node           │
                 │  · if intent pre-filled → pass through  │
                 │  · else call classify_intent() │
                 └────┬────┬────┬────┬──────────┘
                      ↓    ↓    ↓    ↓
               recipe  health fridge  general
               (7 tools)(5 tools)(5 tools)(all tools)
                      ↓    ↓    ↓    ↓
                           END

Design notes:
  · LLM and MCP tools are loaded once and shared across all 4 agents
  · Each agent has its own tool subset (least privilege principle)
  · Each agent uses a dedicated system prompt from prompt_config.yaml
  · Orchestrator accepts a pre-filled intent from the UI layer (avoids extra LLM call)
"""

logger = get_logger("ai_chef.multi_agent_graph")


# ==========================================
# Multi-Agent state definition
# ==========================================
class MultiAgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    intent: str   # "recipe" | "health" | "fridge" | "general"


# ==========================================
# Tool subset whitelists
# ==========================================
_RECIPE_TOOL_NAMES = {
    "get_fridge_inventory", "check_fridge_warnings", "check_allergen_safety",
    "order_fresh_groceries", "get_local_weather",
    "text_to_speech_tool", "search_private_knowledge",
}

_HEALTH_TOOL_NAMES = {
    "get_nutrition_info", "check_allergen_safety",
    "search_private_knowledge", "read_file", "list_directory",
}

_FRIDGE_TOOL_NAMES = {
    "get_fridge_inventory", "check_fridge_warnings",
    "add_food_to_fridge", "remove_food_from_fridge", "clear_fridge_inventory",
    "order_fresh_groceries", "text_to_speech_tool",
}


def _filter_tools(all_tools: list, allowed: set) -> list:
    """Filter the full tool list down to just the allowed subset by name."""
    return [t for t in all_tools if t.name in allowed]


# ==========================================
# Main build function
# ==========================================
def build_multi_agent_graph(llm, all_tools: list):
    """
    Build and compile a Multi-Agent LangGraph StateGraph.

    Args:
        llm:       initialized LLM instance (ChatTongyi or ChatOllama)
        all_tools: full list of tools (MCP tools + local tools)

    Returns:
        compiled LangGraph CompiledGraph
    """
    # Create 4 specialized agent instances, each with its own tool subset
    recipe_tools  = _filter_tools(all_tools, _RECIPE_TOOL_NAMES)
    health_tools  = _filter_tools(all_tools, _HEALTH_TOOL_NAMES)
    fridge_tools  = _filter_tools(all_tools, _FRIDGE_TOOL_NAMES)

    recipe_executor  = create_agent(model=llm, tools=recipe_tools,  middleware=all_middleware)
    health_executor  = create_agent(model=llm, tools=health_tools,  middleware=all_middleware)
    fridge_executor  = create_agent(model=llm, tools=fridge_tools,  middleware=all_middleware)
    general_executor = create_agent(model=llm, tools=all_tools,     middleware=all_middleware)

    logger.info(
        "[MultiAgent] 4 specialized agents ready | "
        f"recipe={len(recipe_tools)}, health={len(health_tools)}, "
        f"fridge={len(fridge_tools)}, general={len(all_tools)} tools"
    )

    # Orchestrator node
    def orchestrator_node(state: MultiAgentState) -> dict:
        """
        Intent routing node.
        If the UI layer already pre-filled the intent, just pass it through (no extra LLM call).
        Otherwise, call classify_intent() to figure it out.
        """
        current_intent = state.get("intent", "")
        if current_intent in ("recipe", "health", "fridge", "general"):
            logger.info(f"[Orchestrator] Using pre-filled intent: {current_intent}")
            return {}

        human_msgs = [m for m in state["messages"] if m.type == "human"]
        query = human_msgs[-1].content if human_msgs else ""
        intent = classify_intent(query)
        logger.info(f"[Orchestrator] Intent classified: '{query[:40]}' → {intent}")
        return {"intent": intent}

    # Factory function for specialized agent nodes
    def _make_agent_node(executor, intent_key: str):
        """
        Factory: creates an async node for each specialized agent.
        Injects the specialized system prompt, replacing any existing SystemMessage.
        """
        async def agent_node(state: MultiAgentState) -> dict:
            system_prompt = get_specialized_prompt(intent_key)
            non_system = [m for m in state["messages"] if m.type != "system"]
            messages = [SystemMessage(content=system_prompt)] + non_system

            response = await executor.ainvoke({"messages": messages})
            return {"messages": response["messages"]}

        agent_node.__name__ = f"{intent_key}_agent_node"
        return agent_node

    recipe_node  = _make_agent_node(recipe_executor,  "recipe")
    health_node  = _make_agent_node(health_executor,  "health")
    fridge_node  = _make_agent_node(fridge_executor,  "fridge")
    general_node = _make_agent_node(general_executor, "general")

    # Routing function
    def route_by_intent(state: MultiAgentState) -> str:
        intent = state.get("intent", "general")
        return intent if intent in ("recipe", "health", "fridge") else "general"

    # Build the StateGraph
    graph = StateGraph(MultiAgentState)

    graph.add_node("orchestrator", orchestrator_node)
    graph.add_node("recipe",       recipe_node)
    graph.add_node("health",       health_node)
    graph.add_node("fridge",       fridge_node)
    graph.add_node("general",      general_node)

    graph.add_edge(START, "orchestrator")
    graph.add_conditional_edges(
        "orchestrator",
        route_by_intent,
        {"recipe": "recipe", "health": "health", "fridge": "fridge", "general": "general"},
    )
    graph.add_edge("recipe",  END)
    graph.add_edge("health",  END)
    graph.add_edge("fridge",  END)
    graph.add_edge("general", END)

    compiled = graph.compile()
    logger.info("[MultiAgent] LangGraph StateGraph compiled successfully")
    return compiled
