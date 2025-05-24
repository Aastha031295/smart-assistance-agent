"""
Test configuration for the application.
"""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import streamlit as st
from langchain.memory import ConversationBufferMemory

# Set environment variables for testing
os.environ["GROQ_API_KEY"] = "test_groq_key"
os.environ["SEARCH_API_KEY"] = "test_search_key"
os.environ["MODEL_NAME"] = "llama3-8b-8192"  # Use smaller model for tests
os.environ["DEBUG"] = "true"
os.environ["VECTOR_DB_PATH"] = "./test_chroma_db"


@pytest.fixture(autouse=True)
def mock_streamlit():
    """Mock Streamlit session state and other functions."""
    # Create a mock for st.session_state
    with patch.object(st, "session_state", {}) as mock_session_state:
        # Initialize required session state variables
        mock_session_state["messages"] = []
        mock_session_state["detected_part"] = None
        mock_session_state["vector_store"] = None
        mock_session_state["conversation_memory"] = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True
        )
        mock_session_state["session_data"] = {
            "session_id": "test-session-id",
            "created_at": "2023-01-01T00:00:00",
            "last_activity": "2023-01-01T00:05:00",
            "message_count": 0,
        }

        yield mock_session_state

    # Clean up after tests
    if os.path.exists("./test_chroma_db"):
        import shutil

        shutil.rmtree("./test_chroma_db")


@pytest.fixture
def sample_documents():
    """Provide sample documents for testing."""
    from langchain_core.documents import Document

    return [
        Document(
            page_content="Headlights not working can be caused by: burned out bulbs, faulty wiring, blown fuse, bad relay, or alternator problems.",
            metadata={
                "category": "Electrical",
                "part": "Headlight",
                "issue": "Not Working",
            },
        ),
        Document(
            page_content="Noise from wheel area while driving could indicate worn wheel bearings, damaged CV joints, brake issues, or suspension problems.",
            metadata={"category": "Suspension", "part": "Wheel", "issue": "Noise"},
        ),
        Document(
            page_content="Car not starting but lights work often indicates a starter motor issue, ignition switch problem, or fuel system failure.",
            metadata={
                "category": "Electrical",
                "part": "Starter",
                "issue": "Not Starting",
            },
        ),
    ]


@pytest.fixture
def sample_image():
    """Provide a sample image for testing."""
    import numpy as np
    from PIL import Image

    # Create a simple 100x100 RGB image
    img_array = np.zeros((100, 100, 3), dtype=np.uint8)
    img_array[:, :, 0] = 255  # Red channel

    return Image.fromarray(img_array)


@pytest.fixture
def mock_groq_llm():
    """Mock Groq LLM for testing."""
    llm = MagicMock()
    llm.invoke.return_value = "This is a test response from the mock LLM."
    return llm


@pytest.fixture
def mock_chroma_db():
    """Mock Chroma vector store for testing."""
    vector_store = MagicMock()

    # Mock similarity search
    vector_store.similarity_search_with_score.return_value = [
        (MagicMock(page_content="Test document content"), 0.8)
    ]

    # Mock as_retriever
    retriever = MagicMock()
    retriever.get_relevant_documents.return_value = [
        MagicMock(page_content="Test document content")
    ]
    vector_store.as_retriever.return_value = retriever

    return vector_store
