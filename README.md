# AutoMend AI - Smart Assistance for vehicles 🔧

An intelligent vehicle repair agent powered by computer vision and RAG technology that instantly identifies car parts from images and provides expert guidance. Combining a pre-loaded automotive knowledge base with real-time internet search capabilities, it delivers accurate troubleshooting steps, repair instructions, and maintenance advice through a natural, multi-turn conversation interface.
Smart Assistance Agent transforms automotive self-repair and diagnostics with adaptive help for any car issue, whether identifying mysterious parts or providing step-by-step repair guides for both novice DIY mechanics and professionals.

## Directory Structure

```
smart-assistance-agent/
├── app.py                 # Main Streamlit application
├── pyproject.toml         # Poetry dependency management
├── .env                   # Environment variables
├── README.md              # Project documentation
├── .streamlit/            # Streamlit configuration
│   └── config.toml        # Streamlit configuration
├── src/                   # Source code
│   ├── __init__.py        # Package initialization
│   ├── config.py          # Configuration management with Pydantic
│   ├── conversation.py    # Conversation memory management
│   ├── image_processor.py # Image recognition utilities
│   ├── knowledge_base.py  # Vector database management
│   ├── rag_engine.py      # RAG implementation
│   └── search_engine.py   # Internet search capabilities
├── tests/                 # Unit and integration tests
│   ├── __init__.py
|   ├── data/                  # Data assets
│       └── sample_images/     # Sample car part images for testing
│   ├── conftest.py        # Test configuration
│   ├── test_config.py
│   ├── test_conversation.py
│   ├── test_image_processor.py
│   ├── test_knowledge_base.py
│   ├── test_rag_engine.py
│   └── test_search_engine.py
└── scripts/               # Utility scripts
    ├── build_knowledge_base.py  # Script to build vector database
    └── test_api_keys.py         # Script to test API keys
```

## Module Descriptions

### Main Application

- **app.py**: Entry point for the Streamlit application. Manages UI and orchestrates the other components.

### Configuration

- **src/config.py**: Pydantic-based configuration system that validates and manages environment variables and application settings.

### Core Components

- **src/conversation.py**: Manages conversation history and context between user interactions.
- **src/image_processor.py**: Handles car part image recognition functionality.
- **src/knowledge_base.py**: Manages the vector database of car repair knowledge.
- **src/rag_engine.py**: Implements retrieval-augmented generation with internet search fallback.
- **src/search_engine.py**: Handles internet search when the knowledge base lacks information.


## Key Features

1. **Robust Configuration Management**:
   - Type-validated configuration using Pydantic
   - Environment variables with proper defaults and validation
   - Separation of configuration from code

2. **Modular Architecture**:
   - Clean separation of concerns
   - Components can be tested independently
   - Easy to maintain and extend

3. **Production-Ready Error Handling**:
   - Comprehensive logging
   - Graceful error handling
   - User-friendly error messages

4. **Multi-Turn Conversation Management**:
   - Persistent conversation history
   - Session management
   - Context pruning for long conversations

5. **Fallback Mechanisms**:
   - Knowledge base with internet search fallback
   - Simulated responses when API keys are missing
   - Sample data when vector database is missing

### Testing

Run tests with Poetry:

```bash
poetry run pytest
```

Or run specific tests:

```bash
poetry run pytest tests/test_knowledge_base.py
```

### Updating the Knowledge Base

Use the provided script:

```bash
poetry run python scripts/build_knowledge_base.py --input_dir path/to/documents
```

### Environment Setup

1. Clone the repository
2. Install dependencies with Poetry:
   ```bash
   poetry install
   ```
3. Copy `.env.example` to `.env` and add your API keys
4. Run the application:
   ```bash
   poetry run streamlit run app.py
   ```

## Security Considerations

1. **API Key Management**:
   - Never commit API keys to version control
   - Use environment variables for sensitive information
   - Use Streamlit secrets in production

2. **User Input Validation**:
   - Validate and sanitize all user inputs
   - Handle file uploads securely

3. **Error Messages**:
   - Don't expose sensitive information in error messages
   - Log detailed errors but show generic messages to users
