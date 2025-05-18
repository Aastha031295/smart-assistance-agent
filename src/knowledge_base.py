"""
Knowledge base module for managing the vector database.
"""

import logging
import os
import shutil
from typing import List, Tuple, Optional

import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from src.config import settings

# Configure logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=settings.log_level.value)


class KnowledgeBase:
    """
    Vector database for car repair knowledge.
    
    Manages creating, updating, and querying the vector database
    of car repair knowledge.
    """
    
    def __init__(self, persist_directory: Optional[str] = None):
        """
        Initialize the knowledge base.
        
        Args:
            persist_directory: Directory to store the vector database
        """
        self.persist_directory = persist_directory or settings.vector_db_path
        self.embeddings = self._get_embeddings()
        self.vector_store = None
    
    def _get_embeddings(self):
        """
        Get the embeddings model.
        
        Returns:
            Embeddings model
        """
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    
    def load(self) -> Chroma:
        """
        Load the vector database from disk.
        
        Returns:
            Loaded vector store
        """
        try:
            logger.info(f"Loading vector database from {self.persist_directory}")
            self.vector_store = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
            return self.vector_store
        except Exception as e:
            logger.warning(f"Failed to load vector database: {str(e)}")
            logger.info("Creating a new vector database with sample data")
            return self.create_sample_db()
    
    def create_from_documents(self, documents: List[Document]) -> Chroma:
        """
        Create a vector database from documents.
        
        Args:
            documents: List of Document objects
            
        Returns:
            Created vector store
        """
        logger.info(f"Creating vector database with {len(documents)} documents")
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = text_splitter.split_documents(documents)
        logger.info(f"Split into {len(chunks)} chunks")
        
        # Create and persist vector store
        self.vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )
        
        return self.vector_store
    
    def create_from_directory(self, directory: str) -> Chroma:
        """
        Create a vector database from a directory of PDF files.
        
        Args:
            directory: Directory containing PDF files
            
        Returns:
            Created vector store
        """
        # Load documents
        logger.info(f"Loading documents from {directory}")
        loader = DirectoryLoader(directory, glob="**/*.pdf", loader_cls=PyPDFLoader)
        documents = loader.load()
        
        return self.create_from_documents(documents)
    
    def create_sample_db(self) -> Chroma:
        """
        Create a sample vector database with common car issues.
        
        Returns:
            Created vector store
        """
        logger.info("Creating sample vector database")
        
        # Sample car repair knowledge
        car_knowledge = [
            Document(
                page_content="Headlights not working can be caused by: burned out bulbs, faulty wiring, blown fuse, bad relay, or alternator problems. First check the fuse box for blown fuses, then inspect bulbs for damage, test the electrical connections, and check the relay.",
                metadata={"category": "Electrical", "part": "Headlight", "issue": "Not Working"}
            ),
            Document(
                page_content="Noise from wheel area while driving could indicate worn wheel bearings, damaged CV joints, brake issues, or suspension problems. Jack up the car safely, spin the wheel, and listen for grinding or humming. Check for play in the wheel by grabbing at 12 and 6 o'clock positions and rocking.",
                metadata={"category": "Suspension", "part": "Wheel", "issue": "Noise"}
            ),
            Document(
                page_content="Car not starting but lights work often indicates a starter motor issue, ignition switch problem, or fuel system failure. Check for clicking sounds when turning the key, test battery voltage, inspect starter connections, and ensure fuel pump is running.",
                metadata={"category": "Electrical", "part": "Starter", "issue": "Not Starting"}
            ),
            Document(
                page_content="Engine overheating is commonly caused by coolant leaks, failed water pump, blocked radiator, faulty thermostat, or broken fan. Check coolant level, inspect for leaks, test radiator and cooling fan operation, and verify thermostat function.",
                metadata={"category": "Cooling", "part": "Radiator", "issue": "Overheating"}
            ),
            Document(
                page_content="Brake pads typically last between 30,000 to 70,000 miles depending on driving habits and conditions. Signs of worn brake pads include squealing or grinding noises, vibration when braking, longer stopping distances, and brake warning light. Replace in pairs (both front or both rear).",
                metadata={"category": "Braking", "part": "Brake Pads", "issue": "Worn"}
            ),
            Document(
                page_content="A car battery typically lasts 3-5 years. Signs of a failing battery include slow engine crank, dim headlights, electrical issues, swollen battery case, and need for frequent jump starts. Test battery voltage with a multimeter - should be around 12.6V when off and 13.7-14.7V when running.",
                metadata={"category": "Electrical", "part": "Battery", "issue": "Failing"}
            ),
            Document(
                page_content="Check engine light illumination can be caused by oxygen sensor failure, loose gas cap, catalytic converter failure, mass airflow sensor failure, or spark plug/wire issues. Use an OBD-II scanner to retrieve the specific error code for accurate diagnosis.",
                metadata={"category": "Engine", "part": "Check Engine Light", "issue": "Illuminated"}
            ),
            Document(
                page_content="Alternator failures can cause battery drain, dim/flickering lights, strange noises, warning lights, and dead battery. To test an alternator, check battery voltage while engine is running - should be 13.7 to 14.7V. Lower voltage indicates alternator problems.",
                metadata={"category": "Electrical", "part": "Alternator", "issue": "Failing"}
            ),
            Document(
                page_content="Transmission fluid should be checked regularly and replaced according to manufacturer's schedule. Look for fluid that is bright red and doesn't smell burnt. Low fluid can cause hard shifting, slipping gears, surging, and overheating. Change fluid if it's dark, cloudy, or smells burnt.",
                metadata={"category": "Transmission", "part": "Transmission Fluid", "issue": "Maintenance"}
            ),
            Document(
                page_content="Serpentine belt squealing or chirping indicates the belt is worn, loose, or misaligned. Inspect for cracks, missing chunks, glazing, or fraying. Most modern belts last 60,000-100,000 miles. Belt tensioner issues can also cause noise and should be checked.",
                metadata={"category": "Engine", "part": "Serpentine Belt", "issue": "Noise"}
            ),
            Document(
                page_content="Tire pressure should be checked monthly and maintained at manufacturer's recommended PSI (typically found on driver's door jamb). Underinflated tires cause poor fuel economy, handling issues, and accelerated wear on outer edges. Overinflated tires cause harsh ride and accelerated wear in center of tread.",
                metadata={"category": "Tires", "part": "Tire Pressure", "issue": "Maintenance"}
            ),
            Document(
                page_content="Oil changes are typically needed every 3,000-7,500 miles for conventional oil and 7,500-15,000 miles for synthetic oil. Check oil level when engine is cold by removing dipstick, wiping clean, reinserting, and checking level. Oil should be amber to light brown - dark or gritty oil needs changing.",
                metadata={"category": "Engine", "part": "Engine Oil", "issue": "Maintenance"}
            ),
        ]
        
        # Create vector store with sample knowledge
        return self.create_from_documents(car_knowledge)
    
    def add_documents(self, documents: List[Document]) -> None:
        """
        Add documents to the vector database.
        
        Args:
            documents: List of Document objects
        """
        if not self.vector_store:
            self.load()
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = text_splitter.split_documents(documents)
        
        # Add chunks to vector store
        self.vector_store.add_documents(chunks)
    
    def has_relevant_info(self, query: str, threshold: Optional[float] = None) -> Tuple[bool, List[Document]]:
        """
        Check if the knowledge base has relevant information for the query.
        
        Args:
            query: Search query
            threshold: Similarity threshold (lower is more similar)
            
        Returns:
            Tuple of (has_relevant_info, relevant_documents)
        """
        if not self.vector_store:
            self.load()
        
        threshold = threshold or settings.similarity_threshold
        
        # Get most relevant documents
        results = self.vector_store.similarity_search_with_score(query, k=2)
        
        if not results:
            return False, []
        
        # Check if the most relevant document is above threshold
        most_relevant_doc, score = results[0]
        
        # Lower score means higher similarity in some implementations
        is_relevant = score < threshold
        
        # Get documents without scores
        docs = [doc for doc, _ in results]
        
        return is_relevant, docs
    
    def get_relevant_documents(self, query: str, k: int = 3) -> List[Document]:
        """
        Get relevant documents for a query.
        
        Args:
            query: Search query
            k: Number of documents to retrieve
            
        Returns:
            List of relevant Document objects
        """
        if not self.vector_store:
            self.load()
        
        retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k}
        )
        
        return retriever.get_relevant_documents(query)
    
    def reset(self) -> None:
        """Reset the vector database by deleting and recreating it."""
        if self.vector_store:
            self.vector_store = None
        
        # Delete the persist directory if it exists
        if os.path.exists(self.persist_directory):
            shutil.rmtree(self.persist_directory)
        
        # Create a new sample database
        self.create_sample_db()


# Create a cached instance for Streamlit
@st.cache_resource
def load_vector_database() -> Chroma:
    """
    Load the vector database for use in Streamlit.
    
    Returns:
        Loaded vector store
    """
    kb = KnowledgeBase()
    return kb.load()


# Provide a global instance
knowledge_base = KnowledgeBase()
