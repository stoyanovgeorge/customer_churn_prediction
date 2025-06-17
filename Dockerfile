FROM python:3.13.5-slim

WORKDIR /app

COPY fastapi_requirements.txt .
RUN pip install --no-cache-dir -r fastapi_requirements.txt

# Copy all app files (including main.py, model.pkl, pipeline.pkl)
COPY ./app .

EXPOSE 8080

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
