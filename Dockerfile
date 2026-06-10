# FROM python:3.10-slim-buster

# WORKDIR /app

# COPY . /app

# RUN pip install -r requirements.txt awscli

# RUN aws s3 cp s3://railway-model-pintu/railway_model/ /app/railway_model/ --recursive

# CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]



FROM python:3.10-slim-buster

WORKDIR /app

COPY . /app

RUN pip install -r requirements.txt awscli

ARG AWS_ACCESS_KEY_ID
ARG AWS_SECRET_ACCESS_KEY
ARG AWS_DEFAULT_REGION=us-east-1

RUN AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID \
    AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY \
    AWS_DEFAULT_REGION=$AWS_DEFAULT_REGION \
    aws s3 cp s3://railway-model-pintu/railway_model/ /app/railway_model/ --recursive

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]