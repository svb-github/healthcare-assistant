FROM python:3.11-slim

# Install uv
RUN pip install --no-cache-dir uv

# Set working directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock /app/

# Install dependencies using uv sync (creates a virtual environment automatically)
RUN uv sync

# Copy application code
COPY app.py tools.py /app/

# Cloud Run expects applications to listen on port 8080 by default.
# The app is configured to use the PORT environment variable.
EXPOSE 8080

# Define environment variables
ENV PYTHONUNBUFFERED=1

# Command to run the application using uv's virtual environment
CMD ["uv", "run", "python", "app.py"]
