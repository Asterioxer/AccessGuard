FROM python:3.12-slim

# Install system dependencies for OpenCV and GIS spatial processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgeos-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install python dependencies
COPY backend/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy backend application and database migrations
COPY backend/app /app/app
COPY scripts /app/scripts
COPY backend/alembic /app/alembic
COPY backend/alembic.ini /app/alembic.ini

# Create unprivileged user for security
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app/data/uploads /app/data/processed && \
    chown -R appuser:appuser /app

USER appuser

ENV PYTHONPATH=/app
ENV PORT=8000

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
