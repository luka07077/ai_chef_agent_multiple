import json
import os

from langchain_community.chat_models import ChatTongyi
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser

from conf import get_agent_config, get_prompt_config
from utils.logger_handler import get_logger

"""
Vision parser module.
Uses qwen-vl-max to recognize ingredients in fridge photos and extract them as structured JSON.
Uses LCEL chain pattern: model | output_parser
"""

logger = get_logger("ai_chef.vision")


def _extract_text_from_response(raw_content) -> str:
    """Extract plain text from a multimodal model response."""
    if isinstance(raw_content, list):
        parts = []
        for block in raw_content:
            if isinstance(block, dict) and "text" in block:
                parts.append(block["text"])
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts)
    return str(raw_content)


def _clean_json_output(text: str) -> str:
    """Strip Markdown code block tags from the model output to get pure JSON."""
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


def parse_fridge_image(image_path: str, api_key: str = None) -> list:
    """
    Core vision function: reads an image and calls Qwen-VL via LCEL chain to recognize ingredients.

    Chain structure: build multimodal message -> ChatTongyi(qwen-vl) -> StrOutputParser -> JSON parse

    Args:
        image_path: path to the fridge or food image
        api_key: DashScope API key (falls back to env variable if not passed)

    Returns:
        list of dicts with ingredient info
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    actual_api_key = api_key or os.environ.get("DASHSCOPE_API_KEY")
    if not actual_api_key:
        raise ValueError("DASHSCOPE_API_KEY not found — please set it in your environment.")

    config = get_agent_config()
    llm_config = config["llm"]
    prompts = get_prompt_config()

    # 1. Build the LCEL chain components
    chat_model = ChatTongyi(
        model_name=llm_config["vision_model"],
        dashscope_api_key=actual_api_key,
        max_tokens=llm_config["vision_max_tokens"],
        temperature=llm_config["vision_temperature"],
    )
    str_parser = StrOutputParser()

    # 2. Build the multimodal message (vision models need direct messages — can't use ChatPromptTemplate)
    abs_image_path = os.path.abspath(image_path)
    file_uri = f"file://{abs_image_path}"

    messages = [
        SystemMessage(content=prompts["vision_system_prompt"]),
        HumanMessage(
            content=[
                {"text": "Please identify all the ingredients in this image and return the result as JSON."},
                {"image": file_uri}
            ]
        )
    ]

    try:
        logger.info("Calling vision model to recognize ingredients...")

        # 3. LCEL chain call: model -> str_parser
        # Vision model's multimodal messages can't go through a prompt template,
        # so we chain (model | parser) directly
        chain = chat_model | str_parser
        raw_output = chain.invoke(messages)

        # 4. Clean and parse the JSON output
        clean_json = _clean_json_output(raw_output)
        parsed_data = json.loads(clean_json)

        logger.info(f"Vision recognition done — found {len(parsed_data)} ingredients")
        return parsed_data

    except json.JSONDecodeError:
        logger.error(f"JSON parse failed, model output was: {raw_output}")
        return []
    except Exception as e:
        logger.error(f"Vision parsing error: {e}")
        return []


if __name__ == "__main__":
    from conf import get_project_root

    test_img_path = os.path.join(get_project_root(), "multimodal", "test_fridge.jpg")

    if os.path.exists(test_img_path):
        print("--- Starting vision parser test ---")
        items = parse_fridge_image(test_img_path)
        print(json.dumps(items, indent=4, ensure_ascii=False))
    else:
        print(f"Test image not found: {test_img_path}")
