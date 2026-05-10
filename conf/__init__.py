import os
import yaml

"""
Config loader: reads YAML config files and makes them available to other modules.
"""

# Autoload .env file from project root if it exists
try:
    from dotenv import load_dotenv
    _env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    load_dotenv(_env_path)
except ImportError:
    pass  # python-dotenv not installed, just use system env vars

# Streamlit Cloud: pull secrets into env vars so the rest of the code needs no changes
try:
    import streamlit as st
    _SECRETS_KEYS = ("DASHSCOPE_API_KEY", "GEMINI_API_KEY", "SPOONACULAR_API_KEY")
    for _k in _SECRETS_KEYS:
        if _k in st.secrets and not os.environ.get(_k):
            os.environ[_k] = st.secrets[_k]
except Exception:
    pass

_CONF_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_CONF_DIR)

# Cache to avoid re-reading files every time
_agent_config = None
_prompt_config = None


def get_project_root() -> str:
    return _PROJECT_ROOT


def get_agent_config() -> dict:
    global _agent_config
    if _agent_config is None:
        path = os.path.join(_CONF_DIR, "agent_config.yaml")
        with open(path, "r", encoding="utf-8") as f:
            _agent_config = yaml.safe_load(f)
    return _agent_config


def get_prompt_config() -> dict:
    global _prompt_config
    if _prompt_config is None:
        path = os.path.join(_CONF_DIR, "prompt_config.yaml")
        with open(path, "r", encoding="utf-8") as f:
            _prompt_config = yaml.safe_load(f)
    return _prompt_config
