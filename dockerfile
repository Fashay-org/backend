FROM python:3.10.14-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy and install requirements
COPY backend/requirements.txt .

# Debug: Show contents of requirements.txt and installed packages
RUN echo "Contents of requirements.txt:" && \
    cat requirements.txt && \
    echo "\nInstalling packages..." && \
    pip install --no-cache-dir -r requirements.txt && \
    echo "\nInstalled packages:" && \
    pip list

# Copy backend files
COPY backend/ /app/backend/
# Copy frontend files
COPY frontend/ /app/frontend/

WORKDIR /app

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]