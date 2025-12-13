# Multi-stage build for smaller image
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip install --no-cache-dir --target=/app/dependencies \
    Flask==3.1.0 \
    torch==2.1.0+cpu --index-url https://download.pytorch.org/whl/cpu \
    numpy==1.24.3 \
    scikit-learn==1.3.0 \
    gunicorn==21.2.0

# Final stage
FROM python:3.11-slim

WORKDIR /app

# Copy only the dependencies from builder
COPY --from=builder /app/dependencies /usr/local/lib/python3.11/site-packages

# Copy application files
COPY backend ./backend
COPY frontend ./frontend
COPY best_model.pth .
COPY gender_encoder.pkl .
COPY smoke_encoder.pkl .
COPY scaler.pkl .

# Expose port
EXPOSE 5000

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Run with gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--timeout", "120", "--workers", "1", "backend.app:app"]
