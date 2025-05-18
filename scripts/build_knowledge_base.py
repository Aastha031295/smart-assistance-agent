"""
Script to build or update the knowledge base from a directory of documents.
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

from langchain_community.document_loaders import (CSVLoader, DirectoryLoader,
                                                  PyPDFLoader, TextLoader)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import settings
from src.knowledge_base import KnowledgeBase

# Configure logging
logging.basicConfig(
    level=settings.log_level.value,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def get_loader_by_extension(file_path: str):
    """
    Get the appropriate document loader based on file extension.

    Args:
        file_path: Path to the document

    Returns:
        Document loader instance
    """
    ext = Path(file_path).suffix.lower()

    if ext == ".pdf":
        return PyPDFLoader(file_path)
    elif ext == ".txt":
        return TextLoader(file_path)
    elif ext in [".csv", ".tsv"]:
        return CSVLoader(file_path)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")


def build_knowledge_base(
    input_dir: str, output_dir: Optional[str] = None, reset: bool = False
):
    """
    Build or update the knowledge base from documents.

    Args:
        input_dir: Directory containing documents
        output_dir: Directory to store the vector database
        reset: Whether to reset the existing database
    """
    start_time = time.time()
    logger.info(f"Building knowledge base from {input_dir}")

    # Initialize knowledge base
    kb = KnowledgeBase(output_dir or settings.vector_db_path)

    # Reset if requested
    if reset and kb.vector_store:
        logger.info("Resetting existing knowledge base")
        kb.reset()

    # Check if directory exists
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    # Load documents
    logger.info("Loading documents...")

    # Check for different file types
    pdf_loader = DirectoryLoader(input_dir, glob="**/*.pdf", loader_cls=PyPDFLoader)
    txt_loader = DirectoryLoader(input_dir, glob="**/*.txt", loader_cls=TextLoader)
    csv_loader = DirectoryLoader(input_dir, glob="**/*.csv", loader_cls=CSVLoader)

    # Load all document types
    documents = []
    documents.extend(pdf_loader.load())
    documents.extend(txt_loader.load())
    documents.extend(csv_loader.load())

    # Check if any documents were found
    if not documents:
        logger.warning(f"No supported documents found in {input_dir}")
        return

    logger.info(f"Loaded {len(documents)} documents")

    # Process and add to vector store
    if kb.vector_store and not reset:
        # Update existing vector store
        logger.info("Updating existing knowledge base")
        kb.add_documents(documents)
    else:
        # Create new vector store
        logger.info("Creating new knowledge base")
        kb.create_from_documents(documents)

    elapsed_time = time.time() - start_time
    logger.info(f"Knowledge base built successfully in {elapsed_time:.2f} seconds")


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Build or update the knowledge base")
    parser.add_argument(
        "--input_dir", type=str, required=True, help="Directory containing documents"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to store the vector database",
    )
    parser.add_argument(
        "--reset", action="store_true", help="Reset the existing database"
    )

    args = parser.parse_args()

    try:
        build_knowledge_base(
            input_dir=args.input_dir, output_dir=args.output_dir, reset=args.reset
        )
    except Exception as e:
        logger.error(f"Error building knowledge base: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
