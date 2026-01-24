# Dockerfile for Hugging Face Spaces
# GPU Memory Calculator - FastAPI Web Application

FROM python:3.12-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=7860

# The two following lines are required for Hugging Face Spaces Dev Mode
# See: https://huggingface.co/docs/hub/spaces-sdks-docker
RUN useradd -m -u 1000 user
USER user

# Set home to the user's home directory
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Set working directory
WORKDIR /app

# Install system dependencies (as root, then switch back to user)
USER root
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        gcc \
        && rm -rf /var/lib/apt/lists/*
USER user

# Copy requirements first for better Docker layer caching
COPY --chown=user requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY --chown=user . .

# Install the package in editable mode
RUN pip install --no-cache-dir -e .

# Expose Hugging Face Spaces default port
EXPOSE 7860

# Run the FastAPI application with uvicorn
CMD ["uvicorn", "web.app:app", "--host", "0.0.0.0", "--port", "7860"]
