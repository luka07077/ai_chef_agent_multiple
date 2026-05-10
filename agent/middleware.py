import time

from langchain.agents import AgentState
from langchain.agents.middleware import before_model, wrap_tool_call, dynamic_prompt
from langgraph.runtime import Runtime

from utils.logger_handler import get_logger
from conf import get_prompt_config

"""
Agent middleware layer (LangChain Official Middleware)
Uses LangChain 1.0's official middleware decorators to intercept model calls and tool calls
inside the agent loop.

Three core middlewares:
1. @wrap_tool_call   — monitors all tool calls (including MCP tools), logs args/result/time/errors
                       also detects warning-type tool calls and sets the warning_mode flag
2. @before_model     — logs context info before each model call (message count, turn)
3. @dynamic_prompt   — dynamically switches the system prompt based on runtime state (e.g. safety alert mode)

Note: all middleware functions are async to work with the agent's async call mode (astream / ainvoke).
"""

logger = get_logger("ai_chef.middleware")

# Tools that trigger warning_mode when they return a risk result
_WARNING_TOOLS = {"check_fridge_warnings", "check_allergen_safety"}

# Shared state across all three middlewares.
# Each middleware gets a different runtime instance, so we use a module-level dict
# as the single source of truth for shared data.
_shared_context: dict = {}


# ==========================================
# Middleware 1: tool call monitoring + warning state
# ==========================================
@wrap_tool_call
async def monitor_tool(request, handler):
    """
    Intercepts every tool call inside the agent loop.
    - Logs tool name, input args, output, and time taken
    - If a warning-type tool returns a risk result, sets warning_mode
      so @dynamic_prompt switches to the safety alert prompt on the next model call
    """
    tool_name = request.tool_call['name']
    tool_args = request.tool_call['args']

    logger.info(f"┌─ TOOL CALL: {tool_name}")
    logger.info(f"│  args: {tool_args}")

    start_time = time.time()
    try:
        result = await handler(request)
        elapsed = time.time() - start_time

        result_preview = str(result.content if hasattr(result, 'content') else result)[:150]
        logger.info(f"└─ OK ({elapsed:.2f}s): {result_preview}")

        # Warning trigger: if the warning tool returned a risk message, write to shared context
        # Note: MCP tool result.content might be a list of text chunks, so we force str() conversion
        if tool_name in _WARNING_TOOLS:
            result_text = str(result.content) if hasattr(result, 'content') else str(result)
            # Match the actual warning prefixes from tool output to avoid false triggers
            if "[EXPIRED]" in result_text or "[EXPIRING SOON]" in result_text or "[SAFETY ALERT]" in result_text:
                _shared_context['warning_mode'] = True
                logger.info("│  >> Risk signal detected, warning_mode activated")

        return result
    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"└─ FAILED ({elapsed:.2f}s): {type(e).__name__}: {e}")
        raise


# ==========================================
# Middleware 2: pre-model logging
# ==========================================
@before_model
async def log_before_model(state: AgentState, runtime: Runtime) -> None:
    """
    Logs context info before each model call.
    Resets warning_mode on the first call of a new conversation turn
    so alerts from the previous turn don't bleed into the next one.
    """
    msg_count = len(state['messages'])

    # <= 2 messages (system + user) means this is the first call of a new turn, reset warning state
    if msg_count <= 2:
        _shared_context['warning_mode'] = False

    mode = "safety alert mode" if _shared_context.get('warning_mode', False) else "normal mode"
    logger.info(f"── MODEL CALL: about to call model [{mode}], context has {msg_count} messages")


# ==========================================
# Middleware 3: dynamic prompt switching
# ==========================================
@dynamic_prompt
async def chef_dynamic_prompt(request):
    """
    Dynamically switches the system prompt based on runtime state.
    When warning_mode is active (set by monitor_tool when a risk signal is detected),
    appends the safety alert instructions to the prompt so the model focuses on allergy
    and expiry risks in its response.
    """
    # 1. Load all prompts from config
    prompts = get_prompt_config()

    # 2. Get the base chef persona prompt, with a fallback just in case
    base_prompt = prompts.get("chef_system_prompt", "You are a professional AI chef assistant.")

    # 3. Read warning_mode from the shared context (set by monitor_tool when risk is detected)
    is_warning_mode = _shared_context.get('warning_mode', False)

    if is_warning_mode:
        logger.info("── PROMPT SWITCH: switching to safety alert mode prompt")

        # 4. Get the alert-specific addition from config, with a short fallback
        warning_addition = prompts.get(
            "chef_warning_prompt_addition",
            "[SAFETY ALERT MODE] Focus on ingredient risks and give clear safety advice."
        )

        # 5. Combine base prompt + warning instructions and return
        return f"{base_prompt}\n\n{warning_addition}"

    # Normal mode: just return the base prompt
    return base_prompt


# Export the middleware list so chef_agent.py can use it
all_middleware = [monitor_tool, log_before_model, chef_dynamic_prompt]
