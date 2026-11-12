
FROM python:3.11-slim

WORKDIR /app

# Copy dependencies
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy API and trained model
COPY api/ ./api/
COPY models/ ./models/

# Expose API port
EXPOSE 8000

# Start FastAPI server
CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
