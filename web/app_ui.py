import os
import sys
import asyncio
import queue
import threading
import time
from datetime import datetime

import streamlit as st

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from agent.chef_agent import (
    init_agent_executor, load_system_prompt, get_chef_response, get_chef_response_stream,
    init_multi_agent_graph, get_multi_agent_response_stream,
)
from agent.state_manager import StateManager
from agent.router import classify_intent, get_specialized_prompt, INTENT_LABELS, has_order_intent
from agent.guardrails import apply_input_guardrails, apply_output_guardrails
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from fridge_manager import fridge_db
from multimodal.audio_handler import speech_to_text
from multimodal.vision_parser import parse_fridge_image
from rag.vector_stores import ingest_single_document, is_document_ingested
from conf import get_agent_config
from utils.logger_handler import get_logger

logger = get_logger("ai_chef.web")

# ==========================================
# Page global config
# ==========================================
st.set_page_config(
    page_title="AI Chef & Smart Ingredient Manager",
    page_icon="👨‍🍳",
    layout="wide"
)

config         = get_agent_config()
lora_config    = config.get("lora", {})
gemini_config  = config.get("gemini", {})
USE_LORA       = lora_config.get("use_lora_adapter", False)
USE_GEMINI     = gemini_config.get("use_gemini", False)
LOCAL_MODEL    = lora_config.get("ollama_model", "qwen2.5:7b")
CLOUD_MODEL    = config.get("llm", {}).get("main_model", "qwen-max-latest")
GEMINI_MODEL   = gemini_config.get("main_model", "gemini-2.5-flash")

memory_manager = StateManager()
USER_ID    = config["user"]["default_user_id"]
SESSION_ID = config["user"]["default_session_id"]

# Debounce + state persistence
for key, default in [
    ("processed_audio_id", None),
    ("processed_img_id", None),
    ("latest_audio_path", None),
    ("processed_kb_id", None),
    ("agent_executor", None),    # kept for backwards compatibility
    ("multi_agent_graph", None), # the Multi-Agent LangGraph graph
    ("mcp_client", None),
    # Multi-agent routing state
    ("active_agent_label", "👨‍🍳 Chef Assistant"),
    # HITL 1: order confirmation (per-message)
    ("pending_order_input", None),
    ("order_approved", False),
    # HITL 2: fridge warning on page load (session-level, once per session)
    ("fridge_warning_checked", False),
    # HITL 3: fridge warning on fridge operations (per-message, resets each turn)
    ("fridge_op_warned", False),         # True = already shown for this message
    ("fridge_op_pending_input", None),   # pending input waiting after fridge op dialog
    ("fridge_op_saved_intent", None),    # saved intent to avoid double LLM call on rerun
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ==========================================
# HITL: purchase intent detection + confirmation dialog
# ==========================================

@st.dialog("🛒 Human-in-the-Loop · Order Confirmation")
def show_order_confirmation():
    """
    Human confirmation dialog: requires the user to explicitly approve before AI
    calls any external ordering API.
    Demonstrates the Human-in-the-Loop principle in Responsible Agentic AI.
    """
    st.markdown("#### ⚠️ Purchase intent detected")
    st.markdown("The AI is about to call an **external supplier API** to place an order — this will trigger a real purchase.")
    st.info(f"💬 Your instruction: **{st.session_state.pending_order_input}**")
    st.divider()
    st.caption("🔒 Per the Human-in-the-Loop safety policy, high-risk actions require human approval before proceeding.")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ Confirm & place order", use_container_width=True, type="primary"):
            st.session_state.order_approved = True
            st.rerun()
    with col2:
        if st.button("❌ Cancel", use_container_width=True):
            st.session_state.order_approved = False
            st.session_state.pending_order_input = None
            st.rerun()


@st.dialog("🧊 Human-in-the-Loop · Fridge Safety Check")
def show_fridge_warning_dialog(warnings: dict, mode: str = "session"):
    """
    Human acknowledgment dialog for fridge safety alerts.
    mode="session"   → page-load check (sets fridge_warning_checked)
    mode="operation" → pre-fridge-op check (sets fridge_op_warned)
    The "Discard expired" button removes expired items from the DB immediately.
    """
    expired       = warnings.get("expired", [])
    expiring_soon = warnings.get("expiring_soon", [])

    st.markdown("#### 🚨 Fridge issue detected")
    st.markdown("Your fridge has items that need attention. Please review before continuing.")

    if expired:
        names = ", ".join(f"**{i['item_name']}** ({i['quantity']}{i['unit']})" for i in expired)
        st.error(f"❌ **Expired (throw away immediately):** {names}")

    if expiring_soon:
        names = ", ".join(f"**{i['item_name']}** ({i['days_left']} day(s) left)" for i in expiring_soon)
        st.warning(f"⏰ **Expiring soon (use first):** {names}")

    st.divider()
    st.caption("🔒 Per the Human-in-the-Loop safety policy, food safety alerts require your acknowledgment before proceeding.")

    def _mark_done():
        if mode == "session":
            st.session_state.fridge_warning_checked = True
        else:
            st.session_state.fridge_op_warned = True

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("✅ Acknowledged, proceed", use_container_width=True, type="primary"):
            _mark_done()
            st.rerun()
    with col2:
        if expired and st.button("🗑️ Discard expired & proceed", use_container_width=True):
            for item in expired:
                fridge_db.consume_food_item(USER_ID, item["item_name"])
            _mark_done()
            st.rerun()
    with col3:
        if st.button("🔕 Dismiss", use_container_width=True):
            _mark_done()
            st.rerun()


# ==========================================
# Agent initialization (cached to avoid reconnecting MCP on every rerun)
# Cloud and local modes both use this — LLM is picked inside init_agent_executor
# ==========================================
async def _ensure_agent():
    """Lazy-initialize the Multi-Agent LangGraph graph (avoids reconnecting MCP on every rerun)."""
    if st.session_state.multi_agent_graph is None:
        graph, client = await init_multi_agent_graph()
        st.session_state.multi_agent_graph = graph
        st.session_state.mcp_client = client
    return st.session_state.multi_agent_graph


def _get_agent_sync():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(_ensure_agent())
    finally:
        loop.close()


# ==========================================
# Agent response (multi-agent routing + streaming output)
# ==========================================
async def _stream_agent_response(user_input: str, intent: str = "general"):
    """
    Async generator: passes user input + intent into the Multi-Agent graph and streams the reply.

    Args:
        user_input: user input already cleaned by guardrails
        intent:     intent pre-classified by the UI layer (Orchestrator node will use it directly)
    """
    graph        = await _ensure_agent()
    chat_history = memory_manager.load_history(SESSION_ID)
    messages     = chat_history + [HumanMessage(content=user_input)]

    full_reply = ""
    async for token in get_multi_agent_response_stream(graph, messages, intent):
        full_reply += token
        yield token

    # Output guard: redact PII from agent reply before saving to memory
    safe_reply = apply_output_guardrails(full_reply)
    memory_manager.add_conversation(SESSION_ID, user_input, safe_reply)


def stream_agent_response(user_input: str, intent: str = "general"):
    """
    Sync wrapper: converts the async generator to a sync generator for st.write_stream.
    Runs the entire async stream in a daemon thread with its own fresh event loop so that
    MCP stdio subprocesses and anyio contexts stay alive for the full duration (avoids
    uvloop / nested-loop incompatibilities on Streamlit Cloud).
    """
    # Reset cached graph so MCP is always initialised inside the same event loop as the stream.
    st.session_state.multi_agent_graph = None
    st.session_state.mcp_client = None

    token_queue: queue.Queue = queue.Queue()

    async def _produce():
        try:
            async for token in _stream_agent_response(user_input, intent):
                token_queue.put(token)
        except Exception as exc:
            token_queue.put(exc)
        finally:
            token_queue.put(None)  # sentinel

    threading.Thread(target=asyncio.run, args=(_produce(),), daemon=True).start()

    while True:
        item = token_queue.get(timeout=120)
        if item is None:
            break
        if isinstance(item, Exception):
            raise item
        yield item


# ==========================================
# Sidebar: control panel + fridge monitor + knowledge base upload
# ==========================================
with st.sidebar:
    st.header("⚙️ Control Panel")
    if USE_LORA:
        st.info(f"🔌 Local mode · `{LOCAL_MODEL}`")
        st.caption("Voice and image recognition not available (API quota limitation)")
    elif USE_GEMINI:
        st.success(f"🌀 Gemini mode · `{GEMINI_MODEL}`")
    else:
        st.success(f"☁️ Cloud mode (Qwen) · `{CLOUD_MODEL}`")
    st.write(f"Current user: `{SESSION_ID}`")

    if st.button("🧹 Clear memory", use_container_width=True):
        memory_manager.clear_memory(SESSION_ID)
        st.session_state.latest_audio_path = None
        st.session_state.multi_agent_graph = None
        st.session_state.agent_executor = None
        st.session_state.mcp_client = None
        st.success("Memory cleared! Chef is ready for a fresh start.")
        st.rerun()

    st.divider()

    # Real-time fridge inventory
    st.header("❄️ Fridge Inventory")
    inventory = fridge_db.get_active_inventory(USER_ID)
    if not inventory:
        st.info("The fridge is empty — ask the chef to order some groceries!")
    else:
        today = datetime.now().date()
        for item in inventory:
            expiration_date = datetime.strptime(item['expiration_date'], "%Y-%m-%d").date()
            days_to_expire = (expiration_date - today).days
            name = item['item_name']
            quantity = item['quantity']
            unit = item['unit']

            if days_to_expire < 0:
                st.error(f"❌ {name} ({quantity}{unit}) - Expired! ({abs(days_to_expire)} days ago)")
            elif days_to_expire <= 2:
                st.warning(f"⚠️ {name} ({quantity}{unit}) - Expires in {days_to_expire} day(s)!")
            elif days_to_expire <= 5:
                st.info(f"🔸 {name} ({quantity}{unit}) - {days_to_expire} days left")
            else:
                st.success(f"✅ {name} ({quantity}{unit}) - {days_to_expire} days left")

    st.divider()

    # Knowledge base upload (supports txt / pdf / docx, with dedup check)
    st.header("📚 Upload Knowledge Base")
    st.caption("Upload recipes or nutrition documents and the chef will remember them.")

    kb_file = st.file_uploader("Supports .txt / .pdf / .docx", type=["txt", "pdf", "docx"])

    if kb_file is not None and kb_file.file_id != st.session_state.processed_kb_id:
        st.session_state.processed_kb_id = kb_file.file_id

        upload_dir = os.path.join(PROJECT_ROOT, "data", "uploads")
        os.makedirs(upload_dir, exist_ok=True)
        kb_path = os.path.join(upload_dir, kb_file.name)

        with open(kb_path, "wb") as f:
            f.write(kb_file.getbuffer())

        if is_document_ingested(kb_path):
            st.warning(f"'{kb_file.name}' has already been indexed — no need to upload it again!")
        else:
            with st.spinner(f"🧠 Parsing {kb_file.name} and adding it to the knowledge base..."):
                success = ingest_single_document(kb_path)
                if success:
                    st.success(f"'{kb_file.name}' indexed successfully!")
                else:
                    st.error("Indexing failed — please check the file format or content.")

# ==========================================
# Main area: multimodal interaction
# ==========================================
st.title("👨‍🍳 AI Chef & Smart Ingredient Manager")
st.caption("A full-stack AI project integrating LLM Agent, MCP microservices, Agentic RAG, multimodal input, and a state machine.")

# ==========================================
# Proactive fridge safety check — runs once per session on page load
# ==========================================
if not st.session_state.fridge_warning_checked:
    from fridge_manager.warning_system import check_expiring_items
    _fridge_warnings = check_expiring_items(USER_ID)
    if _fridge_warnings["expired"] or _fridge_warnings["expiring_soon"]:
        show_fridge_warning_dialog(_fridge_warnings)
        st.stop()
    else:
        st.session_state.fridge_warning_checked = True  # nothing to warn, skip forever this session

# Render chat history
chat_history = memory_manager.load_history(SESSION_ID)
for msg in chat_history:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user", avatar="👤"):
            st.write(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant", avatar="👨‍🍳"):
            st.write(msg.content)

# Render persisted voice reply (auto-play)
if st.session_state.latest_audio_path and os.path.exists(st.session_state.latest_audio_path):
    st.divider()
    st.subheader("🔊 Chef Voice Reply")
    try:
        with open(st.session_state.latest_audio_path, "rb") as f:
            audio_bytes = f.read()
        st.audio(audio_bytes, format="audio/wav", autoplay=True)
    except Exception as e:
        st.warning(f"Audio playback failed: {str(e)}")
    st.divider()

# Multimodal input area (voice and image only available in cloud mode)
user_text = st.chat_input("What do you want to eat? Or what can I help you with?")

if USE_LORA:
    st.caption("🔌 Voice and image recognition not available in local mode (API quota limitation)")
    audio_file = None
    img_file   = None
else:
    col1, col2 = st.columns(2)
    with col1:
        audio_file = st.file_uploader("🎙️ Upload voice command (wav/mp3)", type=["wav", "mp3", "m4a"])
    with col2:
        img_file = st.file_uploader("📸 Upload fridge photo", type=["jpg", "png", "jpeg"])

# ==========================================
# Core logic: handle user input
# ==========================================
final_input = ""

# Priority 1: text input
if user_text:
    final_input = user_text

# Priority 2: voice input (cloud mode only)
elif audio_file is not None and audio_file.file_id != st.session_state.processed_audio_id:
    st.session_state.processed_audio_id = audio_file.file_id

    audio_dir = os.path.join(PROJECT_ROOT, "data", "audio")
    os.makedirs(audio_dir, exist_ok=True)
    temp_audio_path = os.path.join(audio_dir, "temp_upload.wav")

    with open(temp_audio_path, "wb") as f:
        f.write(audio_file.getbuffer())

    with st.spinner("🎧 Listening to your voice..."):
        recognized_text = speech_to_text(temp_audio_path)
        if recognized_text:
            final_input = f"[Voice command]: {recognized_text}"
            st.toast(f"Recognized: {recognized_text}", icon="✅")

# Priority 3: image input (cloud mode only)
elif img_file is not None and img_file.file_id != st.session_state.processed_img_id:
    st.session_state.processed_img_id = img_file.file_id

    upload_dir = os.path.join(PROJECT_ROOT, "data", "uploads")
    os.makedirs(upload_dir, exist_ok=True)
    temp_img_path = os.path.join(upload_dir, "temp_fridge_upload.jpg")

    with open(temp_img_path, "wb") as f:
        f.write(img_file.getbuffer())

    with st.spinner("👁️ Scanning your fridge with the vision model..."):
        try:
            extracted_items = parse_fridge_image(temp_img_path)

            if extracted_items:
                st.toast(f"Recognized {len(extracted_items)} ingredients!", icon="✅")

                added_items_str = ""
                for item in extracted_items:
                    fridge_db.add_food_item(
                        user_id=USER_ID,
                        item_name=item.get("item_name", "unknown"),
                        quantity=item.get("quantity", 1),
                        unit=item.get("unit", "pcs"),
                        days_to_expire=item.get("days_to_expire", 3)
                    )
                    added_items_str += (
                        f"- {item.get('item_name')}: "
                        f"{item.get('quantity')}{item.get('unit')} "
                        f"(estimated shelf life: {item.get('days_to_expire')} days)\n"
                    )

                final_input = (
                    f"[Vision scan complete] I just uploaded a fridge photo. "
                    f"The vision module recognized the following ingredients and added them to the virtual fridge:\n\n"
                    f"{added_items_str}\n"
                    f"Based on everything currently in my fridge, please recommend tonight's dinner."
                )
            else:
                st.error("The vision module couldn't recognize any valid ingredients.")
        except Exception as e:
            st.error(f"Vision parsing error: {str(e)}")

# Restore pending input after any HITL dialog rerun (chat_input is empty after rerun)
_restored_from_order = False
if st.session_state.order_approved and st.session_state.pending_order_input:
    final_input = st.session_state.pending_order_input
    _restored_from_order = True   # flag: skip fridge op check for this order-flow turn
elif st.session_state.fridge_op_warned and st.session_state.fridge_op_pending_input:
    final_input = st.session_state.fridge_op_pending_input

# ==========================================
# Send to agent (with Guardrails + HITL + multi-agent routing)
# ==========================================
if final_input:

    # Layer 1: Guardrails — input safety check + PII redaction
    is_safe, safe_input, block_reason = apply_input_guardrails(final_input)

    if not is_safe:
        # Input blocked: show the guardrail message, don't pass to agent
        with st.chat_message("user", avatar="👤"):
            st.write(final_input)
        with st.chat_message("assistant", avatar="🛡️"):
            st.warning(f"🚫 **Blocked by safety guardrails** · {block_reason}")
        st.stop()

    # Layer 2: HITL — purchasing actions require human approval
    if not st.session_state.order_approved and has_order_intent(safe_input):
        st.session_state.pending_order_input = safe_input
        show_order_confirmation()
        st.stop()

    # Reset order flags after agent executes
    st.session_state.order_approved = False
    st.session_state.pending_order_input = None

    # Orchestrator: intent classification (use cached intent if coming back from fridge op dialog)
    if st.session_state.fridge_op_saved_intent:
        intent      = st.session_state.fridge_op_saved_intent
        agent_label = INTENT_LABELS.get(intent, "👨‍🍳 Chef Assistant")
        st.session_state.active_agent_label = agent_label
    elif not USE_LORA:
        with st.spinner("🔀 Orchestrator is analyzing your intent..."):
            intent = classify_intent(safe_input)
        agent_label = INTENT_LABELS.get(intent, "👨‍🍳 Chef Assistant")
        st.session_state.active_agent_label = agent_label
    else:
        intent      = "general"
        agent_label = "👨‍🍳 Chef Assistant (local mode)"

    # Layer 3: HITL — fridge operation safety check (skip during order flow)
    if intent == "fridge" and not st.session_state.fridge_op_warned and not _restored_from_order:
        from fridge_manager.warning_system import check_expiring_items
        fridge_op_warnings = check_expiring_items(USER_ID)
        if fridge_op_warnings["expired"] or fridge_op_warnings["expiring_soon"]:
            st.session_state.fridge_op_pending_input = safe_input
            st.session_state.fridge_op_saved_intent  = intent
            show_fridge_warning_dialog(fridge_op_warnings, mode="operation")
            st.stop()

    # Reset fridge op flags after agent executes
    st.session_state.fridge_op_warned       = False
    st.session_state.fridge_op_pending_input = None
    st.session_state.fridge_op_saved_intent  = None

    # Show user message
    with st.chat_message("user", avatar="👤"):
        st.write(final_input)   # show original input (with PII — user sees their own message)

    # Stream the specialized agent's response
    with st.chat_message("assistant", avatar="👨‍🍳"):
        # Routing label: tell the user which agent is handling this
        st.caption(f"Routed to · **{agent_label}**")

        spinner_text = (
            f"🔌 Local model ({LOCAL_MODEL}) is thinking..."
            if USE_LORA else
            f"🧠 {agent_label} is working on your request..."
        )
        with st.spinner(spinner_text):
            reply = st.write_stream(stream_agent_response(safe_input, intent))

        # Voice output check (only available in cloud mode)
        if not USE_LORA:
            audio_dir = os.path.join(PROJECT_ROOT, "data", "audio")
            if os.path.exists(audio_dir):
                now = time.time()
                wav_files = [
                    os.path.join(audio_dir, f)
                    for f in os.listdir(audio_dir)
                    if f.endswith(".wav") and f != "temp_upload.wav"
                ]
                if wav_files:
                    latest_audio = max(wav_files, key=os.path.getctime)
                    if now - os.path.getctime(latest_audio) < 30:
                        st.session_state.latest_audio_path = latest_audio
                        with open(latest_audio, "rb") as f:
                            audio_bytes = f.read()
                        st.audio(audio_bytes, format="audio/wav", autoplay=True)

    st.rerun()
