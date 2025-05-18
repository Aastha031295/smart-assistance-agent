"""
RAG engine module for the Car Repair Assistant.

Combines knowledge base, internet search, and LLM to provide answers
to user queries about car repair.
"""

import logging
from typing import Any, Dict, List, Optional, Callable

from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain.llms.fake import FakeListLLM
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableBranch

from src.config import settings
from src.knowledge_base import knowledge_base
from src.search_engine import search_engine, SearchResult


# Configure logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=settings.log_level.value)


class RAGEngine:
    """
    Retrieval Augmented Generation engine for the Car Repair Assistant.
    
    This class handles combining the knowledge base, internet search,
    and language model to provide helpful responses.
    """
    
    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize the RAG engine.
        
        Args:
            model_name: Name of the Groq model to use
        """
        self.model_name = model_name or settings.model_name.value
        self.llm = self._setup_llm()
    
    def _setup_llm(self):
        """
        Set up the language model.
        
        Returns:
            Configured language model
        """
        try:
            api_key = settings.groq_api_key.get_secret_value()
            return ChatGroq(
                model_name=self.model_name,
                temperature=0.2,
                groq_api_key=api_key
            )
        except Exception as e:
            logger.error(f"Failed to initialize Groq LLM: {str(e)}")
            logger.warning("Using fake LLM for demonstration")
            
            # Use a fake LLM for demonstration purposes
            return FakeListLLM(responses=[
                "I need a Groq API key to provide detailed assistance. Based on general knowledge, if your headlights aren't working, first check the fuses, then inspect the bulbs and wiring connections.",
                "To diagnose that wheel noise, safely jack up the car and try to move the wheel. Grinding or play in the wheel often indicates worn wheel bearings that should be replaced soon.",
                "For optimal vehicle maintenance, regularly check fluid levels, tire pressure, and listen for unusual noises. Address small issues before they become major problems."
            ])
    
    def setup_rag_chain(self, conversation_memory: Optional[ConversationBufferMemory] = None):
        """
        Set up the RAG chain with internet search fallback.
        
        Args:
            conversation_memory: ConversationBufferMemory instance
            
        Returns:
            Configured RAG chain
        """
        # Make sure the knowledge base is loaded
        if not knowledge_base.vector_store:
            knowledge_base.load()
        
        # Setup memory
        memory = conversation_memory or ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # Setup prompt for knowledge base responses
        kb_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert car mechanic assistant that helps identify car parts 
            and provide repair guidance. Use the information from the knowledge base to provide 
            accurate and helpful responses. If you're not sure about something, admit it rather 
            than making up information. If the user has uploaded an image of a car part, refer to 
            the detected part name in your response.
            
            When responding about a car part or issue:
            1. Briefly explain what the part does or the nature of the issue
            2. List possible causes of the problem
            3. Provide troubleshooting steps in a logical order
            4. Give repair or replacement guidance including difficulty level and tools needed
            5. Mention safety precautions when relevant
            
            Keep your responses practical and focused on helping the user fix their problem.
            Use professional but accessible language.
            
            IMPORTANT: When responding, refer to the conversation history to maintain context
            and avoid repeating information already provided. If the user has referenced something from 
            earlier in the conversation, make sure to address it appropriately."""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
            ("system", "Here is information from the knowledge base that may be helpful:\n\n{context}")
        ])
        
        # Setup prompt for internet search responses
        web_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert car mechanic assistant that helps identify car parts 
            and provide repair guidance. The knowledge base did not have relevant information for this query,
            so you are using information from internet search results. Synthesize this information to provide
            a helpful response, and mention that this information comes from online sources.
            
            When responding about a car part or issue:
            1. Briefly explain what the part does or the nature of the issue
            2. List possible causes of the problem
            3. Provide troubleshooting steps in a logical order
            4. Give repair or replacement guidance including difficulty level and tools needed
            5. Mention safety precautions when relevant
            
            Keep your responses practical and focused on helping the user fix their problem.
            Use professional but accessible language.
            
            IMPORTANT: When responding, refer to the conversation history to maintain context
            and avoid repeating information already provided. If the user has referenced something from 
            earlier in the conversation, make sure to address it appropriately."""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
            ("system", "Here is information from internet search results that may be helpful:\n\n{search_results}")
        ])
        
        # Create RAG chain with knowledge base
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        # Chain for knowledge base responses
        kb_chain = (
            {"context": lambda x: format_docs(knowledge_base.get_relevant_documents(x["question"])), 
             "question": RunnablePassthrough(), 
             "chat_history": RunnablePassthrough()}
            | kb_prompt
            | self.llm
            | StrOutputParser()
        )
        
        # Chain for internet search responses
        def format_search_results(results: List[SearchResult]) -> str:
            formatted = ""
            for i, result in enumerate(results):
                formatted += f"Result {i+1}:\nTitle: {result.title}\nContent: {result.snippet}\nURL: {result.url}\n\n"
            return formatted
        
        def search_and_format(query):
            results = search_engine.search(query)
            return format_search_results(results)
        
        web_chain = (
            {"search_results": lambda x: search_and_format(x["question"]), 
             "question": RunnablePassthrough(), 
             "chat_history": RunnablePassthrough()}
            | web_prompt
            | self.llm
            | StrOutputParser()
        )
        
        # Decision function to determine which chain to use
        def determine_chain(inputs):
            query = inputs["question"]
            
            # Check if knowledge base has relevant information
            has_info, _ = knowledge_base.has_relevant_info(query)
            
            if has_info:
                logger.info(f"Using knowledge base for query: {query}")
                return kb_chain
            else:
                logger.info(f"Using internet search for query: {query}")
                return web_chain
        
        # Create a branch to decide which chain to use
        chain = RunnableBranch(
            (lambda x: determine_chain(x)),
            kb_chain,  # Default to knowledge base if the branch function fails
        )
        
        return chain


# Function to get a configured RAG chain
def setup_rag_chain(model_name: Optional[str] = None):
    """
    Set up a RAG chain with the given model name.
    
    Args:
        model_name: Name of the Groq model to use
        
    Returns:
        Configured RAG chain
    """
    engine = RAGEngine(model_name)
    return engine.setup_rag_chain(st.session_state.conversation_memory)
