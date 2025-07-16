# Use the official python 3.10 image as the base
FROM python:3.10-slim

# Set worikng directory inside the container
WORKDIR /app
ENV PIP_ROOT_USER_ACTION=ignore

# Install system dependencies (for faiss, numpy, etc.)
RUN apt-get update && apt-get install -y\
    build-essential \
    curl\
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy your custom scripts into the image
COPY run_with_gptcache.py ./run_with_gptcache.py

COPY ./*.py ./

# Set default command (can be overriden)
CMD ["python", "run_with_gptcache.py"]
