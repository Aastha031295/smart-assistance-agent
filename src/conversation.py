"""
Conversation management module for maintaining chat history and memory.
"""

import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List

import streamlit as st
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import BaseMessage

from src.config import settings

# Configure logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=settings.log_level.value)


class SessionManager:
    """
    Manages session state and conversation memory.

    This class handles creating and updating the conversation memory
    that maintains context between user interactions.
    """

    def __init__(self):
        """Initialize session manager."""
        self.max_history_length = settings.max_history_length

    def initialize_session(self) -> None:
        """
        Initialize session state variables.

        This method sets up all necessary session state variables if they
        don't already exist.
        """
        # Initialize UI message history
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Initialize conversation memory
        if "conversation_memory" not in st.session_state:
            st.session_state.conversation_memory = ConversationBufferMemory(
                memory_key="chat_history", return_messages=True
            )

        # Initialize detected part information
        if "detected_part" not in st.session_state:
            st.session_state.detected_part = None

        # Initialize vector store reference
        if "vector_store" not in st.session_state:
            st.session_state.vector_store = None

        # Initialize session metadata
        if "session_data" not in st.session_state:
            st.session_state.session_data = {
                "session_id": self._generate_session_id(),
                "created_at": datetime.now(),
                "last_activity": datetime.now(),
                "message_count": 0,
            }

    def add_user_message(self, message: str) -> None:
        """
        Add a user message to the conversation.

        This updates both the UI message history and the conversation memory.

        Args:
            message: User message content
        """
        # Add to UI message history
        st.session_state.messages.append({"role": "user", "content": message})

        # Add to conversation memory
        st.session_state.conversation_memory.chat_memory.add_user_message(message)

        # Update session metadata
        self._update_session_activity()

    def add_assistant_message(self, message: str) -> None:
        """
        Add an assistant message to the conversation.

        This updates both the UI message history and the conversation memory.

        Args:
            message: Assistant message content
        """
        # Add to UI message history
        st.session_state.messages.append({"role": "assistant", "content": message})

        # Add to conversation memory
        st.session_state.conversation_memory.chat_memory.add_ai_message(message)

        # Update session metadata
        self._update_session_activity()

    def get_chat_history(self) -> List[BaseMessage]:
        """
        Get the full conversation history.

        Returns:
            List of conversation messages
        """
        return st.session_state.conversation_memory.chat_memory.messages

    def clear_conversation(self) -> None:
        """Clear the conversation history."""
        # Clear UI message history
        st.session_state.messages = []

        # Clear conversation memory
        st.session_state.conversation_memory = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True
        )

        # Reset session metadata
        st.session_state.session_data["message_count"] = 0

        logger.info("Conversation cleared")

    def _update_session_activity(self) -> None:
        """Update session activity timestamp and message count."""
        st.session_state.session_data["last_activity"] = datetime.now()
        st.session_state.session_data["message_count"] += 1

        # Prune history if it exceeds maximum length
        self._prune_history_if_needed()

    def _prune_history_if_needed(self) -> None:
        """
        Prune conversation history if it exceeds the maximum length.

        This prevents the context window from growing too large.
        """
        messages = st.session_state.conversation_memory.chat_memory.messages

        if len(messages) > self.max_history_length:
            # Remove oldest messages (keep the most recent)
            excess = len(messages) - self.max_history_length
            st.session_state.conversation_memory.chat_memory.messages = messages[
                excess:
            ]
            logger.info(f"Pruned {excess} messages from conversation history")

    def is_session_expired(self) -> bool:
        """
        Check if the current session has expired.

        Returns:
            True if session has expired, False otherwise
        """
        if "session_data" not in st.session_state:
            return False

        last_activity = st.session_state.session_data["last_activity"]
        expiry_minutes = settings.session_expiry_minutes

        # Check if session has been inactive for too long
        elapsed = (datetime.now() - last_activity).total_seconds() / 60
        return elapsed > expiry_minutes

    def _generate_session_id(self) -> str:
        """
        Generate a unique session ID.

        Returns:
            Unique session ID
        """
        return str(uuid.uuid4())

    def get_session_info(self) -> Dict[str, Any]:
        """
        Get information about the current session.

        Returns:
            Dictionary with session information
        """
        if "session_data" not in st.session_state:
            return {}

        session_data = st.session_state.session_data
        message_count = session_data["message_count"]
        created_at = session_data["created_at"]
        last_activity = session_data["last_activity"]

        # Calculate session duration
        duration_seconds = (datetime.now() - created_at).total_seconds()
        hours, remainder = divmod(duration_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)

        return {
            "session_id": session_data["session_id"],
            "message_count": message_count,
            "created_at": created_at.strftime("%Y-%m-%d %H:%M:%S"),
            "last_activity": last_activity.strftime("%Y-%m-%d %H:%M:%S"),
            "duration": f"{int(hours)}h {int(minutes)}m {int(seconds)}s",
        }


# Create a global instance
session_manager = SessionManager()


def initialize_conversation_memory() -> None:
    """
    Initialize conversation memory in the session state.

    This is a helper function for use in Streamlit apps.
    """
    session_manager.initialize_session()
