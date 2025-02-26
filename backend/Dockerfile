FROM python:3.9-slim

WORKDIR /app

# Install dependencies more efficiently and include optimization tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    libtesseract-dev \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables for better OCR performance
ENV OMP_THREAD_LIMIT=2
ENV PYTHONUNBUFFERED=1

# Copy requirements and install separately (for better caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose API port
EXPOSE 8000

# Run with optimized settings for CPU and memory usage
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2", "--limit-concurrency", "20"]