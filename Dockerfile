# Use official lightweight Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and files
COPY . .

# Run initial pipeline to ensure model artifacts exist
RUN python run_pipeline.py

# Expose FastAPI port
EXPOSE 8000

# Launch FastAPI web server
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
