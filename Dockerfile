FROM python:3.10-slim-buster

WORKDIR /app

COPY . /app

RUN pip install -r requirements.txt

RUN aws s3 cp s3://railway-model-pintu/railway_model/ /app/railway_model/ --recursive

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]