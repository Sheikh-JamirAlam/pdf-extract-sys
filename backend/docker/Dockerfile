FROM python:3.9-slim

WORKDIR /app

# Optimize apt installations - combine into one layer
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-eng \
    libtesseract-dev \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables for resource control
ENV OMP_THREAD_LIMIT=2
ENV PYTHONUNBUFFERED=1
ENV TESSERACT_NUM_THREADS=2
ENV MALLOC_ARENA_MAX=2
ENV PYTHONMALLOC=malloc

COPY app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ .

EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=5s --start-period=30s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--limit-concurrency", "5"]