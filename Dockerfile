# Use a lightweight Python image
FROM python:3.10-slim AS builder

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set working directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies into a virtual environment
RUN uv sync --frozen --no-dev --no-install-project

# Final stage
FROM python:3.10-slim

WORKDIR /app

# Copy the virtual environment from the builder
COPY --from=builder /app/.venv /app/.venv
# Copy the application code
COPY app /app/app
# Copy default config (can be overridden by volume mount)
COPY config /app/config

# Ensure logs directory exists
RUN mkdir -p /app/logs && chmod 777 /app/logs

# Set environment variables
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1

# Expose the application port
EXPOSE 1234

# Start the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "1234", "--log-level", "info"]
