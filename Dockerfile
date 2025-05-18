"""
Dockerfile for Car Repair Assistant application.
"""

# Use Python 3.10 slim image as base
FROM python:3.12-slim

# Set work directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    POETRY_VERSION=1.7.0 \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_CREATE=false

# Install system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -

# Add Poetry to PATH
ENV PATH="${PATH}:/root/.local/bin"

# Copy pyproject.toml and poetry.lock (if available)
COPY pyproject.toml ./
COPY poetry.lock* ./

# Install dependencies
RUN poetry install --no-dev --no-root

# Copy the rest of the application
COPY . .

# Create directory for the vector database
RUN mkdir -p ./chroma_db && chmod 777 ./chroma_db

# Expose the port Streamlit will run on
EXPOSE 8501

# Create a non-root user to run the application
RUN useradd -m appuser
USER appuser

# Command to run the application
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]