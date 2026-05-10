import os
import re
import logging
from typing import Tuple

"""
Guardrails module for the AI chef system.
Provides two layers of safety to meet "Responsible Agentic AI" requirements:

Layer 1 · LLM-as-Judge (Input Safety Filter)
  - Uses qwen-turbo to check if user input is harmful
  - Catches violence, drugs, explosives, illegal content, etc. — even with rephrasing
  - If the LLM call fails, we fail-open (let it through) so normal use isn't broken

Layer 2 · PII Redaction
  - Regex-based detection and masking of: email, phone, ID card, bank card numbers
  - Protects both directions: input (prevent PII from entering the LLM) + output (prevent leaks to frontend)

Usage:
    from agent.guardrails import apply_input_guardrails, apply_output_guardrails

    # Handle user input
    is_safe, safe_input, reason = apply_input_guardrails(user_text)
    if not is_safe:
        return reason  # show the block message

    # Handle agent output
    safe_output = apply_output_guardrails(raw_output)
"""

logger = logging.getLogger("ai_chef.guardrails")

# ==========================================
# Layer 1: LLM-as-Judge safety check
# ==========================================

_judge_llm = None


def _get_judge_llm():
    """Lazy-load the LLM judge model (qwen-turbo, same spec as router.py)."""
    global _judge_llm
    if _judge_llm is None:
        from langchain_community.chat_models import ChatTongyi
        from conf import get_agent_config
        config = get_agent_config()
        api_key = os.environ.get("DASHSCOPE_API_KEY")
        if not api_key:
            logger.warning("[Guardrails] DASHSCOPE_API_KEY not set — LLM-as-Judge unavailable, will fail-open")
            return None
        _judge_llm = ChatTongyi(
            model_name=config["llm"]["evaluator_model"],  # qwen-turbo
            dashscope_api_key=api_key,
            temperature=0,
        )
        logger.info("[Guardrails] LLM-as-Judge initialized")
    return _judge_llm


def check_input_safety(text: str) -> Tuple[bool, str]:
    """
    LLM-as-Judge safety check: uses qwen-turbo to judge if the input is harmful.

    Args:
        text: raw user input

    Returns:
        (is_safe, block_reason)
        - is_safe=True: input is fine, continue processing
        - is_safe=False: blocked, block_reason is a friendly message to show the user
    """
    from langchain_core.messages import SystemMessage, HumanMessage

    try:
        from conf import get_prompt_config
        judge_prompt = get_prompt_config().get(
            "safety_judge_prompt",
            "Determine if the user input is harmful. Only output SAFE or UNSAFE: <reason>."
        )
        llm = _get_judge_llm()
        if llm is None:
            return True, ""  # fail-open: no key, let it through
        response = llm.invoke([
            SystemMessage(content=judge_prompt),
            HumanMessage(content=text[:300]),
        ])
        raw = response.content.strip()

        if raw.upper().startswith("UNSAFE"):
            reason = raw.split(":", 1)[-1].strip() if ":" in raw else "inappropriate content"
            logger.warning(f"[Guardrails] LLM flagged as unsafe | reason: {reason} | input: '{text[:60]}'")
            return (
                False,
                f"⚠️ Sorry, your request contains inappropriate content ({reason}) and can't be processed. "
                "Feel free to ask about food or cooking!"
            )

        logger.info(f"[Guardrails] LLM says safe: '{text[:40]}'")
        return True, ""

    except Exception as e:
        # fail-open: if the safety LLM is down, let it through so normal use isn't affected
        logger.warning(f"[Guardrails] Safety check failed, letting it through: {e}")
        return True, ""


# ==========================================
# Layer 2: PII regex patterns
# ==========================================
_PII_PATTERNS: list[tuple[re.Pattern, str]] = [
    # Email address (standard format)
    (
        re.compile(r'\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b'),
        "[email redacted]"
    ),
    # International phone number: +<country code> <local number>
    # Covers formats like +65 8982 7664 / +1 555-123-4567 / +44 7700 900123
    (
        re.compile(r'\+\d{1,3}[\s\-]?\(?\d{1,5}\)?(?:[\s\-]?\d{3,5}){1,3}(?=\s|$|[^\d])'),
        "[phone redacted]"
    ),
    # Chinese mainland phone number (starts with 1, second digit 3-9, 11 digits total)
    (
        re.compile(r'(?<!\d)1[3-9]\d{9}(?!\d)'),
        "[phone redacted]"
    ),
    # Chinese national ID (18 digits, last digit can be X)
    (
        re.compile(
            r'\b\d{6}(?:19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{3}[\dX]\b',
            re.IGNORECASE
        ),
        "[ID redacted]"
    ),
    # Bank / credit card (16 digits, may have spaces or dashes)
    (
        re.compile(r'\b(?:\d{4}[- ]?){3}\d{4}\b'),
        "[card number redacted]"
    ),
]


# ==========================================
# Core functions
# ==========================================

def redact_pii(text: str) -> str:
    """
    Scan text for PII and replace it with placeholder strings.
    Handles emails, phone numbers, ID cards, and bank cards.

    Args:
        text: input text (can be user input or agent output)

    Returns:
        the text with sensitive info masked
    """
    original = text
    for pattern, replacement in _PII_PATTERNS:
        text = pattern.sub(replacement, text)
    if text != original:
        logger.info(f"[Guardrails] PII redacted | original: '{original[:80]}' → '{text[:80]}'")
    else:
        logger.debug(f"[Guardrails] PII scan: no sensitive data found in '{original[:60]}'")
    return text


def apply_input_guardrails(user_input: str) -> Tuple[bool, str, str]:
    """
    Main input guard: runs LLM safety check + PII redaction.

    Args:
        user_input: raw user input

    Returns:
        (is_safe, sanitized_input, block_reason)
        - is_safe=False: request blocked, block_reason has the message to show
        - is_safe=True: sanitized_input is the cleaned version, block_reason is empty
    """
    # Layer 1: LLM semantic safety check
    is_safe, reason = check_input_safety(user_input)
    if not is_safe:
        return False, user_input, reason

    # Layer 2: PII redaction (stop private data from entering the agent context / LLM)
    sanitized = redact_pii(user_input)
    return True, sanitized, ""


def apply_output_guardrails(agent_output: str) -> str:
    """
    Output guard: redact PII from agent replies before showing them to the user.

    Args:
        agent_output: raw agent output text

    Returns:
        the cleaned output
    """
    return redact_pii(agent_output)
