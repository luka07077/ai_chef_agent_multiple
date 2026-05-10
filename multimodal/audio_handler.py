import os
import time
import dashscope
from langchain_community.chat_models import ChatTongyi
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool

from conf import get_project_root, get_agent_config, get_prompt_config
from utils.logger_handler import get_logger

"""
Audio handler module.
Handles voice input (ASR) and voice output (TTS) for the multimodal features.
ASR uses LCEL chain pattern: model | StrOutputParser (qwen-audio-turbo, multilingual)
TTS uses DashScope CosyVoice v2 (tts_v2 API, multilingual Chinese + English)
All generated audio files are saved under data/audio/.
"""

logger = get_logger("ai_chef.audio")

AUDIO_DIR = os.path.join(get_project_root(), "data", "audio")
os.makedirs(AUDIO_DIR, exist_ok=True)


def speech_to_text(audio_path: str, api_key: str = None) -> str:
    """
    ASR (speech to text): calls ChatTongyi via an LCEL chain to transcribe audio.

    Chain structure: multimodal audio message -> ChatTongyi(qwen-audio) -> StrOutputParser
    """
    if not os.path.exists(audio_path):
        logger.error(f"Audio file not found: {audio_path}")
        return ""

    actual_api_key = api_key or os.environ.get("DASHSCOPE_API_KEY")
    if not actual_api_key:
        raise ValueError("DASHSCOPE_API_KEY not found — please set it in your environment variables.")

    config = get_agent_config()
    prompts = get_prompt_config()

    logger.info(f"Running ASR on audio file: {os.path.basename(audio_path)}")

    # 1. Build the LCEL chain: model | str_parser
    chat_audio = ChatTongyi(
        model_name=config["llm"]["audio_asr_model"],
        dashscope_api_key=actual_api_key,
        temperature=0.1,
    )
    str_parser = StrOutputParser()
    chain = chat_audio | str_parser

    # 2. Build the multimodal audio message
    abs_audio_path = os.path.abspath(audio_path)
    file_uri = f"file://{abs_audio_path}"

    messages = [
        SystemMessage(content=prompts["asr_system_prompt"]),
        HumanMessage(
            content=[
                {"text": prompts["asr_user_prompt"]},
                {"audio": file_uri}
            ]
        )
    ]

    try:
        # 3. Run the chain, get back a plain string
        result_text = chain.invoke(messages)
        logger.info(f"ASR done: {result_text[:50]}...")
        return result_text.strip()

    except Exception as e:
        logger.error(f"Speech recognition error: {e}")
        return ""


@tool
def text_to_speech_tool(text: str) -> str:
    """
    Tool that converts text to a speech audio file.
    Call this when you need to read out a recipe, remind the user about expiring items,
    or have a spoken conversation.

    Args:
        text: the text to convert to speech.
    """
    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        return "Failed: DASHSCOPE_API_KEY not found in environment variables."

    config = get_agent_config()
    llm_cfg = config["llm"]
    tts_model = llm_cfg["audio_tts_model"]     # e.g. "cosyvoice-v2"
    tts_voice = llm_cfg.get("audio_tts_voice", "longxiaochun")
    sample_rate = llm_cfg["audio_sample_rate"]  # 22050 for CosyVoice v2

    dashscope.api_key = api_key
    logger.info(f"Synthesizing speech [{tts_model}/{tts_voice}]: '{text[:30]}...'")

    try:
        # CosyVoice v2 uses the newer tts_v2 API; sambert models used the old tts API.
        # We branch on model name so the config switch is seamless.
        if "cosyvoice" in tts_model.lower():
            from dashscope.audio.tts_v2 import SpeechSynthesizer as SpeechSynthesizerV2, AudioFormat
            # format must be an AudioFormat enum, NOT a plain string.
            synthesizer = SpeechSynthesizerV2(
                model=tts_model,
                voice=tts_voice,
                format=AudioFormat.WAV_22050HZ_MONO_16BIT,
            )
            audio_data = synthesizer.call(text)
        else:
            # Legacy sambert path kept for backward compatibility
            from dashscope.audio.tts import SpeechSynthesizer
            result = SpeechSynthesizer.call(
                model=tts_model,
                text=text,
                sample_rate=sample_rate,
                format='wav'
            )
            audio_data = result.get_audio_data() if result.get_audio_data() is not None else None

        if audio_data:
            timestamp = int(time.time())
            filename = f"chef_reply_{timestamp}.wav"
            final_output_path = os.path.join(AUDIO_DIR, filename)
            with open(final_output_path, 'wb') as f:
                f.write(audio_data)
            logger.info(f"Speech synthesis successful: {final_output_path}")
            return f"Speech synthesis successful, file saved at: {final_output_path}"
        else:
            return "Speech synthesis failed: empty audio data returned."

    except Exception as e:
        logger.error(f"TTS error: {e}")
        return f"Tool execution error: {str(e)}"


if __name__ == "__main__":
    test_text = "Hello! I'm your smart AI chef assistant."
    print("--- TTS test ---")
    tts_result = text_to_speech_tool.invoke({"text": test_text})
    print(tts_result)
