FROM python:3.11-slim

# Set workdir
WORKDIR /app

# Install system dependencies yang dibutuhkan numpy/scipy
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements dulu (layer caching optimization)
COPY requirements.txt .

# Install Python deps
RUN pip install --no-cache-dir -r requirements.txt

# Copy semua source code
COPY . .

# Train model saat build image — generate credit_model.pkl
RUN python ai_engine/train_model.py

# Cloud Run inject PORT via env variable, default 8080
ENV PORT=8080

# Expose port
EXPOSE 8080

# Run dengan Gunicorn — production grade
CMD exec gunicorn --bind 0.0.0.0:$PORT --workers 2 --timeout 120 app:app
