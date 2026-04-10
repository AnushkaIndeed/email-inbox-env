# Use official Python runtime as base image
FROM python:3.10-slim

# Set working directory in container
WORKDIR /app

# Copy requirements
COPY requirements.txt requirements.txt

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set environment variable
ENV PYTHONUNBUFFERED=1

# Expose port (if needed for serving)
EXPOSE 8000

# Default command
CMD ["python", "inference.py", "spam"]
