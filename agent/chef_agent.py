import os
import sys
import asyncio

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from langchain_core.messages import HumanMessage, SystemMessage
from langchain.agents import create_agent

from agent.mcp_client_tools import load_mcp_tools
from agent.middleware import all_middleware
from agent.models.local_model import LocalChefModel
from agent.models.cloud_model import CloudChefModel
from agent.models.gemini_model import GeminiChefModel
from multimodal.audio_handler import text_to_speech_tool
from agent.state_manager import StateManager
from rag.agentic_rag_core import search_private_knowledge
from conf import get_agent_config, get_prompt_config
from utils.logger_handler import get_logger

"""
Main brain of the AI chef agent.
Builds a ReAct workflow using LangChain's create_agent,
integrating MCP tools, multimodal input, Agentic RAG, official middleware, and a persistent state machine.

The run mode is determined by lora.use_lora_adapter in conf/agent_config.yaml:
  false (default) → cloud mode: ChatTongyi (qwen3-max), voice/image available
  true            → local mode: ChatOllama (qwen2.5:7b), MCP/RAG available, voice/image not available
Both modes share the same agent path — only the LLM is different.
"""

logger = get_logger("ai_chef.agent")


def load_system_prompt() -> str:
    """Load the system prompt from prompt_config.yaml."""
    prompts = get_prompt_config()
    return prompts.get("chef_system_prompt", "You are a professional AI chef assistant.")


async def init_agent_executor():
    """
    Initialize the agent executor:
    1. Pick the LLM based on config (cloud ChatTongyi / local ChatOllama)
    2. Load MCP tools + local tools
    3. Build the agent with create_agent and attach the official middleware
    """
    logger.info("Waking up the AI chef brain...")

    config        = get_agent_config()
    lora_config   = config.get("lora", {})
    gemini_config = config.get("gemini", {})
    use_lora      = lora_config.get("use_lora_adapter", False)
    use_gemini    = gemini_config.get("use_gemini", False)
    llm_config    = config["llm"]

    # 1. Pick the LLM (priority: local > Gemini > Qwen cloud)
    if use_lora:
        llm = LocalChefModel(lora_config).llm
        logger.info(f"[Local mode] LLM: {lora_config.get('ollama_model', 'qwen2.5:7b')}")
    elif use_gemini:
        llm = GeminiChefModel(gemini_config).llm
        logger.info(f"[Gemini mode] LLM: {gemini_config.get('main_model', 'gemini-2.5-flash')}")
    else:
        llm = CloudChefModel(llm_config).llm
        logger.info(f"[Cloud mode] LLM: {llm_config.get('main_model', 'qwen-max-latest')}")

    # 2. Load MCP and local tools (available in both modes)
    mcp_tools, mcp_client = await load_mcp_tools()
    all_tools = mcp_tools + [text_to_speech_tool, search_private_knowledge]
    logger.info(f"Agent ready! Loaded {len(all_tools)} tools total.")

    # 3. Build the agent with create_agent and attach middleware
    #    @wrap_tool_call  — tool monitoring + warning state
    #    @before_model    — pre-model logging
    #    @dynamic_prompt  — dynamic prompt switching
    agent_executor = create_agent(
        model=llm,
        tools=all_tools,
        middleware=all_middleware
    )

    return agent_executor, mcp_client


async def interactive_chat():
    """
    Terminal chat loop for testing. Cloud and local modes share the same path.
    Switching modes only changes the LLM inside init_agent_executor().
    """
    config      = get_agent_config()
    lora_config = config.get("lora", {})
    use_lora    = lora_config.get("use_lora_adapter", False)
    llm_config  = config.get("llm", {})

    logger.info("=" * 55)
    logger.info(f"  Mode       : {'Local (Ollama)' if use_lora else 'Cloud (DashScope API)'}")
    if use_lora:
        logger.info(f"  Local model: {lora_config.get('ollama_model', 'qwen2.5:7b')}")
        logger.info(f"  Ollama URL : {lora_config.get('ollama_base_url', 'http://localhost:11434')}")
        logger.info("  Voice/Image: not available (API quota limitation)")
    else:
        logger.info(f"  Cloud model: {llm_config.get('main_model', 'qwen3-max')}")
    logger.info("=" * 55)

    print("\n" + "=" * 50)
    if use_lora:
        print("  AI Chef Terminal [Local Mode · Ollama]")
        print("  MCP / RAG available | Voice / Image not available")
    else:
        print("  AI Chef Terminal [Cloud Mode]")
    print("  Type 'quit' to exit, 'clear' to reset memory")
    print("=" * 50 + "\n")

    memory_manager      = StateManager()
    current_session_id  = config["user"]["default_session_id"]

    try:
        agent_executor, mcp_client = await init_agent_executor()
        system_msg = SystemMessage(content=load_system_prompt())
        logger.info(f"Loaded history for session [{current_session_id}]")

        while True:
            user_input = input("\nYou: ")

            if user_input.lower() in ['quit', 'exit']:
                print("Chef: See you next time! Enjoy your meal!")
                break

            if user_input.lower() == 'clear':
                memory_manager.clear_memory(current_session_id)
                continue

            if not user_input.strip():
                continue

            try:
                chat_history = memory_manager.load_history(current_session_id)
                messages = [system_msg] + chat_history + [HumanMessage(content=user_input)]

                response = await agent_executor.ainvoke({"messages": messages})
                bot_reply = response["messages"][-1].content
                print(f"\nChef: {bot_reply}")

                memory_manager.add_conversation(
                    session_id=current_session_id,
                    human_text=user_input,
                    ai_text=bot_reply
                )

            except Exception as e:
                logger.error(f"Agent error: {e}")
                print(f"\nSomething went wrong: {str(e)}")

    finally:
        if 'mcp_client' in locals():
            logger.info("Safely disconnecting from MCP services...")


async def get_chef_response(agent_executor, messages) -> str:
    """
    Non-streaming call: get the full agent reply at once.
    Middleware is attached via the middleware param and runs automatically.
    """
    response = await agent_executor.ainvoke({"messages": messages})
    return response["messages"][-1].content


async def get_chef_response_stream(agent_executor, messages):
    """
    Hybrid approach: use astream to get each step's output,
    then yield the final AI text reply character by character.
    """
    final_content = ""

    async for chunk in agent_executor.astream(
            {"messages": messages},
            stream_mode="updates"
    ):
        if not isinstance(chunk, dict):
            continue
        for node_name, node_output in chunk.items():
            if not node_output or not isinstance(node_output, dict):
                continue
            if "messages" not in node_output:
                continue
            for msg in node_output["messages"]:
                if hasattr(msg, "content") and msg.content and not getattr(msg, "tool_calls", None):
                    final_content = msg.content

    if final_content:
        for char in final_content:
            yield char
            await asyncio.sleep(0.02)  # control typing speed


async def init_multi_agent_graph():
    """
    Initialize the Multi-Agent LangGraph graph.
    Compatible signature with init_agent_executor(): both return (graph, mcp_client)
    so app_ui.py can switch between them seamlessly.
    """
    from agent.multi_agent_graph import build_multi_agent_graph

    logger.info("Building Multi-Agent LangGraph graph...")

    config        = get_agent_config()
    lora_config   = config.get("lora", {})
    gemini_config = config.get("gemini", {})
    use_lora      = lora_config.get("use_lora_adapter", False)
    use_gemini    = gemini_config.get("use_gemini", False)
    llm_config    = config["llm"]

    if use_lora:
        llm = LocalChefModel(lora_config).llm
        logger.info(f"[MultiAgent] Local mode LLM: {lora_config.get('ollama_model', 'qwen2.5:7b')}")
    elif use_gemini:
        llm = GeminiChefModel(gemini_config).llm
        logger.info(f"[MultiAgent] Gemini mode LLM: {gemini_config.get('main_model', 'gemini-2.5-flash')}")
    else:
        llm = CloudChefModel(llm_config).llm
        logger.info(f"[MultiAgent] Cloud mode LLM: {llm_config.get('main_model', 'qwen-max-latest')}")

    mcp_tools, mcp_client = await load_mcp_tools()
    all_tools = mcp_tools + [text_to_speech_tool, search_private_knowledge]

    graph = build_multi_agent_graph(llm, all_tools)
    logger.info(f"[MultiAgent] System ready, {len(all_tools)} tools loaded")
    return graph, mcp_client


async def get_multi_agent_response_stream(graph, messages, intent: str = "general"):
    """
    Streaming response generator for the Multi-Agent graph.
    intent is pre-computed by the UI layer and passed in — the Orchestrator node
    will use it directly without calling the LLM again.

    Args:
        graph:    the compiled LangGraph returned by build_multi_agent_graph()
        messages: [SystemMessage, ...history..., HumanMessage]
        intent:   pre-computed intent label (empty string = let Orchestrator classify)
    """
    initial_state = {"messages": messages, "intent": intent}
    final_content = ""

    async for chunk in graph.astream(initial_state, stream_mode="updates"):
        if not isinstance(chunk, dict):
            continue
        for node_name, node_output in chunk.items():
            if node_name == "orchestrator" or not isinstance(node_output, dict):
                continue
            if "messages" not in node_output:
                continue
            for msg in node_output["messages"]:
                if (hasattr(msg, "content") and msg.content
                        and not getattr(msg, "tool_calls", None)
                        and getattr(msg, "type", "") in ("ai", "assistant")):
                    content = msg.content
                    # Gemini returns a list of dicts, Qwen returns a plain string — normalize both
                    if isinstance(content, list):
                        content = "".join(
                            p.get("text", "") if isinstance(p, dict) else str(p)
                            for p in content
                        )
                    final_content = content

    if final_content:
        for char in final_content:
            yield char
            await asyncio.sleep(0.02)


if __name__ == "__main__":
    asyncio.run(interactive_chat())
