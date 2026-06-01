# Multi-stage production Docker image definition
FROM python:3.10-slim AS builder

WORKDIR /app

# Install system build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install dependencies into a separate wheels / build directory
RUN pip install --no-cache-dir --user -r requirements.txt


# Production Stage
FROM python:3.10-slim AS runner

WORKDIR /app

# Copy installed python dependencies from builder stage
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Copy source directories and files
COPY src/ ./src/
COPY sql/ ./sql/
COPY notebooks/ ./notebooks/
COPY app.py .
COPY README.md .

# Configure environment defaults
ENV DATA_DIR=/data
ENV MODELS_DIR=/models
ENV DATABASE_URL=sqlite:////data/credit_risk.db
ENV ENVIRONMENT=docker
ENV LOG_LEVEL=INFO

# Expose Streamlit application port
EXPOSE 8501

# Run the Streamlit application
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
