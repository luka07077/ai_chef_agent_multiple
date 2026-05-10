import os
import json
from langchain_core.messages import HumanMessage, AIMessage, messages_from_dict, messages_to_dict

from conf import get_project_root, get_agent_config
from utils.logger_handler import get_logger

"""
State and memory manager for the AI chef agent.
Handles saving/loading chat history per session, and trims the context window when it gets too long.
"""

logger = get_logger("ai_chef.state")


class StateManager:

    def __init__(self, memory_dir: str = None):
        if memory_dir is None:
            self.memory_dir = os.path.join(get_project_root(), "memory_sessions")
        else:
            self.memory_dir = memory_dir
        os.makedirs(self.memory_dir, exist_ok=True)

    def _get_session_file(self, session_id: str) -> str:
        return os.path.join(self.memory_dir, f"{session_id}.json")

    def load_history(self, session_id: str) -> list:
        """Load the chat history for a given user session."""
        file_path = self._get_session_file(session_id)
        if not os.path.exists(file_path):
            return []

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                dicts = json.load(f)
                return messages_from_dict(dicts)
        except Exception as e:
            logger.warning(f"Failed to read memory, starting fresh: {e}")
            return []

    def save_history(self, session_id: str, messages: list):
        """Serialize and save the message list to disk."""
        file_path = self._get_session_file(session_id)
        dicts = messages_to_dict(messages)

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(dicts, f, ensure_ascii=False, indent=2)

    def add_conversation(self, session_id: str, human_text: str, ai_text: str, max_keep: int = None):
        """Add a new turn to the session history, then trim if it's too long."""
        if max_keep is None:
            config = get_agent_config()
            max_keep = config["state"]["max_history_messages"]

        history = self.load_history(session_id)
        history.append(HumanMessage(content=human_text))
        history.append(AIMessage(content=ai_text))

        # Sliding window: keep only the most recent messages
        if len(history) > max_keep:
            history = history[-max_keep:]

        self.save_history(session_id, history)

    def clear_memory(self, session_id: str):
        """Delete all saved memory for a given user session."""
        file_path = self._get_session_file(session_id)
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"Cleared all memory for session {session_id}.")
