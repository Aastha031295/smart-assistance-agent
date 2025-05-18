"""
Main application file for the Car Repair Assistant chatbot.
"""

import os
import time
import logging
import traceback

import streamlit as st
from PIL import Image

from src.config import settings
from src.conversation import session_manager, initialize_conversation_memory
from src.image_processor import identify_car_part
from src.knowledge_base import knowledge_base
from src.rag_engine import setup_rag_chain
from src.search_engine import search_engine

# Configure logging
logging.basicConfig(
    level=settings.log_level.value,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set page config
st.set_page_config(
    page_title=settings.app_name,
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Cache for app performance
@st.cache_resource
def load_resources():
    """Initialize and load resources needed by the application."""
    try:
        # Load vector database
        kb = knowledge_base.load()
        logger.info("Vector database loaded successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to load resources: {str(e)}")
        return False


def main():
    """Main application entry point."""
    try:
        # Initialize session state
        initialize_conversation_memory()
        
        # Check for session expiry
        if session_manager.is_session_expired():
            logger.info("Session expired, resetting conversation")
            session_manager.clear_conversation()
        
        # Load resources
        resources_loaded = load_resources()
        
        # Main title
        st.title("🚗 Car Repair Assistant")
        st.markdown("Upload a car part image or ask questions about car problems directly.")
        
        # Two columns for layout
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Image upload section
            display_image_upload()
            
        with col2:
            # Chat interface
            display_chat_interface(resources_loaded)
        
        # Admin sidebar
        display_admin_sidebar()
        
        # Add a footer
        st.markdown("---")
        st.markdown("Car Repair Assistant | Powered by Groq AI")
    
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        logger.error(traceback.format_exc())
        st.error(f"An error occurred: {str(e)}")


def display_image_upload():
    """Display the image upload section and process uploaded images."""
    # Image upload
    uploaded_image = st.file_uploader("Upload a photo of a car part", type=["jpg", "jpeg", "png"])
    
    if uploaded_image:
        try:
            # Display the uploaded image
            image = Image.open(uploaded_image)
            st.image(image, caption="Uploaded Image", width=300)
            
            # Process the image
            with st.spinner("Analyzing image..."):
                result = identify_car_part(image)
                detected_part = result["name"]
                st.session_state.detected_part = detected_part
                
                # Display detected part info
                st.success(f"✅ Detected: {detected_part}")
                st.info(f"Common issues with this part:")
                for issue in result["common_issues"]:
                    st.markdown(f"- {issue}")
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            st.error("Unable to process the uploaded image. Please try another image.")


def display_chat_interface(resources_loaded: bool):
    """
    Display the chat interface and handle user interactions.
    
    Args:
        resources_loaded: Whether resources were loaded successfully
    """
    # Chat section header
    st.subheader("Chat with the Car Assistant")
    
    # Quick help section
    with st.expander("💡 Try asking about...", expanded=False):
        st.markdown("""
        - Why is my car making noise when I turn?
        - How do I know if my brake pads are worn?
        - What causes a car to overheat?
        - My headlights aren't working, what should I check?
        - How long should a car battery last?
        - What are signs of alternator problems?
        """)
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Show resources not loaded warning
    if not resources_loaded:
        st.warning("Some resources failed to load. Functionality may be limited.")
    
    # Chat input
    if prompt := st.chat_input("Ask about car repairs or issues..."):
        process_user_message(prompt)


def process_user_message(prompt: str):
    """
    Process a user message and generate a response.
    
    Args:
        prompt: User message
    """
    try:
        # Add user message to conversation
        session_manager.add_user_message(prompt)
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            
            # Create enhanced prompt with detected part information if available
            if st.session_state.detected_part:
                enhanced_prompt = f"[Image uploaded shows a {st.session_state.detected_part}] {prompt}"
            else:
                enhanced_prompt = prompt
            
            # Setup RAG chain
            chain = setup_rag_chain(settings.model_name.value)
            
            if chain:
                with st.spinner("Thinking..."):
                    # Get response from RAG chain with internet fallback
                    try:
                        response = chain.invoke({
                            "question": enhanced_prompt,
                            "chat_history": session_manager.get_chat_history()
                        })
                        
                        # Display streaming response
                        display_streaming_response(response, message_placeholder)
                        
                        # Add assistant response to conversation
                        session_manager.add_assistant_message(response)
                        
                    except Exception as e:
                        logger.error(f"Error generating response: {str(e)}")
                        error_message = "I'm sorry, I encountered an error while generating a response. Please try again or rephrase your question."
                        message_placeholder.markdown(error_message)
                        session_manager.add_assistant_message(error_message)
            else:
                # Fallback message if RAG chain setup fails
                error_msg = ("I'm currently unable to access my knowledge base. " 
                            "Please ensure API keys are configured properly.")
                message_placeholder.markdown(error_msg)
                session_manager.add_assistant_message(error_msg)
    
    except Exception as e:
        logger.error(f"Error processing message: {str(e)}")
        logger.error(traceback.format_exc())
        st.error(f"An error occurred: {str(e)}")


def display_streaming_response(full_response: str, message_placeholder):
    """
    Display a streaming effect for the response.
    
    Args:
        full_response: Complete response text
        message_placeholder: Streamlit container for the message
    """
    response_chunks = full_response.split()
    displayed_response = ""
    
    for i in range(0, len(response_chunks), 3):  # Process in chunks for efficiency
        chunk = " ".join(response_chunks[i:i+3])
        displayed_response += chunk + " "
        message_placeholder.markdown(displayed_response + "▌")
        time.sleep(0.01)
    
    # Display full response
    message_placeholder.markdown(full_response)


def display_admin_sidebar():
    """Display admin settings in the sidebar."""
    with st.sidebar:
        st.title("Admin Settings")
        st.markdown("These settings are for administrators only.")
        
        # API key configuration
        with st.expander("API Configuration"):
            groq_api_key = st.text_input("Groq API Key", type="password")
            if groq_api_key and st.button("Set Groq API Key"):
                os.environ["GROQ_API_KEY"] = groq_api_key
                st.success("API key set")
            
            search_api_key = st.text_input("Search API Key", type="password")
            if search_api_key and st.button("Set Search API Key"):
                os.environ["SEARCH_API_KEY"] = search_api_key
                st.success("Search API key set")
            
            # Model selection
            model_options = [m.value for m in settings.GroqModel]
            selected_model = st.selectbox(
                "Select Groq Model", 
                options=model_options, 
                index=model_options.index(settings.model_name.value)
            )
            if selected_model != settings.model_name.value and st.button("Set Model"):
                os.environ["MODEL_NAME"] = selected_model
                st.success(f"Model changed to {selected_model}")
                st.info("Please refresh the page for changes to take effect")
        
        # Knowledge base management
        with st.expander("Knowledge Base Management"):
            st.markdown("### Update Knowledge Base")
            st.markdown("In a production system, this would be an admin-only feature.")
            
            uploaded_files = st.file_uploader(
                "Upload car repair PDFs (admin only)", 
                accept_multiple_files=True,
                type=["pdf"],
                key="admin_upload"
            )
            
            if uploaded_files and st.button("Update Knowledge Base"):
                with st.spinner("Processing documents..."):
                    try:
                        # In a real implementation, process and add to knowledge base
                        st.info("In a production implementation, these documents would update the vector store.")
                        st.success(f"Processed {len(uploaded_files)} documents")
                    except Exception as e:
                        logger.error(f"Error updating knowledge base: {str(e)}")
                        st.error(f"Error: {str(e)}")
            
            if st.button("Reset Knowledge Base"):
                with st.spinner("Resetting knowledge base..."):
                    try:
                        knowledge_base.reset()
                        st.success("Knowledge base reset to default")
                        st.info("Please refresh the page")
                    except Exception as e:
                        logger.error(f"Error resetting knowledge base: {str(e)}")
                        st.error(f"Error: {str(e)}")
        
        # Session management
        with st.expander("Session Management"):
            st.markdown("### Conversation History")
            
            # Show session info
            session_info = session_manager.get_session_info()
            if session_info:
                st.write(f"Session ID: {session_info.get('session_id', 'N/A')}")
                st.write(f"Messages: {session_info.get('message_count', 0)}")
                st.write(f"Started: {session_info.get('created_at', 'N/A')}")
                st.write(f"Duration: {session_info.get('duration', 'N/A')}")
            
            # Clear conversation button
            if st.button("Clear Conversation"):
                session_manager.clear_conversation()
                st.success("Conversation cleared")
                st.info("Please refresh the page")
        
        # Debug information
        if settings.debug:
            with st.expander("Debug Information"):
                st.write("Environment:", {k: v for k, v in os.environ.items() if k in ["MODEL_NAME", "DEBUG", "LOG_LEVEL"]})
                st.write("Session State Keys:", list(st.session_state.keys()))
                st.write("Detected Part:", st.session_state.detected_part)
                st.write("Message Count:", len(st.session_state.messages))


# Run the app
if __name__ == "__main__":
    main()
