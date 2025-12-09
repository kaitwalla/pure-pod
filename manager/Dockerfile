FROM node:20-alpine AS frontend-builder

WORKDIR /frontend

# Copy package files first for better caching
COPY manager/frontend/package.json manager/frontend/package-lock.json ./
RUN npm ci

# Copy frontend source and build
COPY manager/frontend ./

# Build argument for frontend password (empty = no password required)
ARG VITE_PASSWORD
ENV VITE_PASSWORD=$VITE_PASSWORD

RUN npm run build


FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY manager/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY manager/src ./src
COPY shared ./shared

# Copy built frontend from builder stage
COPY --from=frontend-builder /frontend/dist ./static

# Create audio storage directory
RUN mkdir -p /app/audio

ENV PORT=8000
EXPOSE ${PORT}

CMD uvicorn src.main:app --host 0.0.0.0 --port ${PORT}
