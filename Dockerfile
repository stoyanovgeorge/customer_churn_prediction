FROM python:3.13.5-slim

WORKDIR /app

COPY fastapi_requirements.txt .
RUN pip install --no-cache-dir -r fastapi_requirements.txt

# Copying the model
COPY app/model.pkl ./app/model.pkl

COPY app/ ./app/

WORKDIR /app

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
