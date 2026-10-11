FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV SHINY_PORT=8000
ENV API_BASE_URL=http://127.0.0.1:3000/api
ENV DATA_DIR=/app/data_cache

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY . .

# Ensure data cache directory exists and entrypoint script has executable permissions
RUN mkdir -p /app/data_cache && chmod +x /app/entrypoint.sh

EXPOSE 8000

# Healthcheck checking Shiny HTTP endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=25s --retries=3 \
    CMD curl -f http://localhost:${SHINY_PORT}/ || exit 1

ENTRYPOINT ["/app/entrypoint.sh"]
