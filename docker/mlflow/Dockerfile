FROM python:3.11-slim

RUN pip install --no-cache-dir \
    mlflow==2.16.2 \
    scikit-learn==1.5.2 \
    xgboost==2.1.1 \
    pandas==2.2.3 \
    numpy==1.26.4 \
    scipy==1.14.1

EXPOSE 5000
WORKDIR /mlflow
