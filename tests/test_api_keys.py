"""
Test script for API keys to verify they are working correctly.
"""

import argparse
import logging
import os
import sys

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import SearchProvider, settings
from src.rag_engine import RAGEngine
from src.search_engine import SearchEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_groq_api_key(api_key: str = None, model_name: str = None):
    """
    Test the Groq API key by making a simple query.

    Args:
        api_key: Groq API key to test
        model_name: Model name to use for testing

    Returns:
        bool: Whether the API key is working
    """
    if api_key:
        os.environ["GROQ_API_KEY"] = api_key

    if model_name:
        os.environ["MODEL_NAME"] = model_name

    logger.info("Testing Groq API key...")

    try:
        engine = RAGEngine()
        llm = engine._setup_llm()

        # Make a simple query
        response = llm.invoke("What is a car alternator?")

        logger.info(f"Groq API response received, length: {len(str(response))}")
        logger.info("Groq API key is working ✓")
        return True

    except Exception as e:
        logger.error(f"Groq API key test failed: {str(e)}")
        return False


def test_search_api_key(api_key: str = None, provider: str = None):
    """
    Test the search API key by making a simple query.

    Args:
        api_key: Search API key to test
        provider: Search provider to use

    Returns:
        bool: Whether the API key is working
    """
    if api_key:
        os.environ["SEARCH_API_KEY"] = api_key

    if provider:
        os.environ["SEARCH_PROVIDER"] = provider

    logger.info(
        f"Testing search API key for provider: {settings.search_provider.value}..."
    )

    try:
        search_engine = SearchEngine()
        results = search_engine.search("car repair headlight", 2)

        if not results:
            logger.error("Search API returned no results")
            return False

        logger.info(f"Search API returned {len(results)} results")
        for i, result in enumerate(results):
            logger.info(f"Result {i+1}: {result.title[:50]}...")

        logger.info("Search API key is working ✓")
        return True

    except Exception as e:
        logger.error(f"Search API key test failed: {str(e)}")
        return False


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Test API keys")
    parser.add_argument("--groq", action="store_true", help="Test Groq API key")
    parser.add_argument("--search", action="store_true", help="Test search API key")
    parser.add_argument("--groq-key", type=str, help="Groq API key to test")
    parser.add_argument("--search-key", type=str, help="Search API key to test")
    parser.add_argument("--model", type=str, help="Groq model to use")
    parser.add_argument(
        "--provider",
        type=str,
        choices=[p.value for p in SearchProvider],
        help="Search provider to use",
    )
    parser.add_argument("--all", action="store_true", help="Test all API keys")

    args = parser.parse_args()

    # Default to testing all if no specific test is selected
    if not (args.groq or args.search) and not args.all:
        logger.info("No specific test selected, testing all API keys")
        args.all = True

    status = {"groq": None, "search": None}

    # Test Groq API key
    if args.groq or args.all:
        status["groq"] = test_groq_api_key(args.groq_key, args.model)

    # Test search API key
    if args.search or args.all:
        status["search"] = test_search_api_key(args.search_key, args.provider)

    # Print summary
    logger.info("\n--- API Key Test Summary ---")

    if status["groq"] is not None:
        logger.info(
            f"Groq API key: {'✓ Working' if status['groq'] else '✗ Not working'}"
        )

    if status["search"] is not None:
        logger.info(
            f"Search API key: {'✓ Working' if status['search'] else '✗ Not working'}"
        )

    # Return error code if any test failed
    if False in status.values():
        sys.exit(1)


if __name__ == "__main__":
    main()
