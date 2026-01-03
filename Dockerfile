FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose only Streamlit port (Render limitation)
EXPOSE 8501

# Create startup script - run FastAPI in background, Streamlit in foreground
RUN echo '#!/bin/bash\n\
uvicorn api:app --host localhost --port 8000 &\n\
sleep 2\n\
streamlit run app.py --server.port=8501 --server.address=0.0.0.0\n\
' > /app/start.sh && chmod +x /app/start.sh

# Health check - check Streamlit health
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

CMD ["/app/start.sh"]
