FROM python:3.10-slim

WORKDIR /app

# Install uv for fast dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

COPY pyproject.toml .
COPY uv.lock .
COPY requirements.txt .

# Install dependencies system-wide to ensure all scripts (like inference.py) 
# can find modules even if run via standard python3
RUN uv pip install --system -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

EXPOSE 7860

# Run the server directly using the module path
CMD ["python", "-m", "server.app"]