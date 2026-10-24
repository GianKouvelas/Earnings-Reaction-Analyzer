# Same Python minor version used for training, to keep the pickled model compatible
FROM python:3.10-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install dependencies first: this layer is cached until requirements change
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Then copy only the code the API needs
COPY data_pipeline/ data_pipeline/
COPY model/ model/
COPY api/ api/

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]