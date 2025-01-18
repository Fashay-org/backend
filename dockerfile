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

# Install Python packages individually to ensure each one installs correctly
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir fastapi && \
    pip install --no-cache-dir gdown && \
    pip install --no-cache-dir langgraph && \
    pip install --no-cache-dir langsmith && \
    pip install --no-cache-dir numpy && \
    pip install --no-cache-dir openai && \
    pip install --no-cache-dir opencv-python && \
    pip install --no-cache-dir opencv-python-headless && \
    pip install --no-cache-dir Pillow && \
    pip install --no-cache-dir pydantic && \
    pip install --no-cache-dir python-dotenv && \
    pip install --no-cache-dir python-bcrypt && \
    pip install --no-cache-dir scikit-learn && \
    pip install --no-cache-dir starlette && \
    pip install --no-cache-dir supabase && \
    pip install --no-cache-dir tenacity==8.0.0 && \
    pip install --no-cache-dir "pydantic[email]" && \
    pip install --no-cache-dir uvicorn && \
    pip install --no-cache-dir jinja2 && \
    pip install --no-cache-dir "passlib[bcrypt]" && \
    pip install --no-cache-dir pinecone-client && \
    pip install --no-cache-dir python-multipart

COPY . /app/backend/
COPY ../frontend /app/frontend/

WORKDIR /app/backend

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]