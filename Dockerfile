FROM python:3.11-slim
LABEL env_id="GoldTrading-XAU/USD-v4" spec="openenv-v1"
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser
EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=10s CMD curl -f http://localhost:7860/ || exit 1
CMD ["python", "app.py"]
