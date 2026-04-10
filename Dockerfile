FROM python:3.10-slim

WORKDIR /app

# Copy requirements first (better caching)
COPY requirements.txt .

# Upgrade pip + install dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy full project
COPY . .

# Prevent buffering
ENV PYTHONUNBUFFERED=1

# Run inference
CMD ["sh", "-c", "python inference.py & python app.py"]