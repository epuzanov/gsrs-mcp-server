FROM python:3.13-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY pyproject.toml README.md ./
COPY app/ ./app/
COPY scripts/ ./scripts/

# Install the local package so the documented CLI entrypoint is available in the image.
RUN pip install --no-cache-dir . --no-deps

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/readyz || exit 1

# Run the MCP server (streamable-http transport by default)
CMD ["gsrs-mcp-server"]
