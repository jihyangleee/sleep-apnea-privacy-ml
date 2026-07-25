FROM python:3.12-slim

RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY model.py dataset.py secret_sharing.py simulate.py he_client.py schemas.py hospital_app.py ./

ENV HOSPITAL_ID=0
ENV MODEL_PATH=/shared/vertical_model.pt
ENV CSV_PATH=""
ENV PYTHONUNBUFFERED=1

CMD ["sh", "-c", "uvicorn hospital_app:app --host 0.0.0.0 --port 8000"]
