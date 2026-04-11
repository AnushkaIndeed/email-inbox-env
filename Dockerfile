FROM python:3.10-slim

WORKDIR /app

# Install uv for fast dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

COPY pyproject.toml .
COPY uv.lock .
COPY requirements.txt .

# Sync dependencies using uv
RUN uv sync --no-dev

COPY . .

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

EXPOSE 7860

# Run the server via the entry point defined in pyproject.toml
CMD ["uv", "run", "server"]