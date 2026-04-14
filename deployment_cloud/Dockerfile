# Dockerfile — OCT Retinal AI Edge Inference API
# Base: Python 3.11 slim — keeps image under 500MB
# Compatible with Google Cloud Run free tier (4GB RAM)

FROM python:3.11-slim

# Prevent interactive prompts during apt installs
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PORT=8080

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy and install Python dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY main.py .

# Models directory — downloaded at runtime from HuggingFace
RUN mkdir -p /app/models

# Expose port
EXPOSE 8080

# Run with uvicorn — Cloud Run sends traffic to PORT env variable
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
