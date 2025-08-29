# Dockerfile for LLMCache Benchmark
# Supports both local and remote Ollama configurations

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -e .

# Copy source code
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY experiments/ ./experiments/
COPY schema/ ./schema/

# Create results directory
RUN mkdir -p results/raw

# Set environment variables
ENV PYTHONPATH=/app:/app/src
ENV PYTHONUNBUFFERED=1

# Default to localhost Ollama (can be overridden)
ENV OLLAMA_BASE_URL=http://localhost:11434

# Expose port if running a service (not needed for batch jobs)
# EXPOSE 8000

# Default command runs with default config
CMD ["python", "scripts/run.py", "--config", "experiments/experiment.yaml"]

# Usage examples:
# 
# Local development:
#   docker build -t llmcache-bench .
#   docker run -v $(pwd)/results:/app/results llmcache-bench
#
# With remote GPU Ollama:
#   docker run -e OLLAMA_BASE_URL=https://your-ngrok-url.com -v $(pwd)/results:/app/results llmcache-bench
#
# Custom config:
#   docker run -v $(pwd)/my-config.yaml:/app/experiments/my-config.yaml llmcache-bench python scripts/run.py --config experiments/my-config.yaml