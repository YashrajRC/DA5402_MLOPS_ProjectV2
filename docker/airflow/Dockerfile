FROM apache/airflow:2.9.2-python3.11

USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential && rm -rf /var/lib/apt/lists/*

USER airflow
RUN pip install --no-cache-dir --timeout 120 \
    pandas==2.2.3 \
    scikit-learn==1.5.2 \
    nltk==3.9.1 \
    textstat==0.7.4 \
    vaderSentiment==3.3.2 \
    scipy==1.14.1 \
    numpy==1.26.4

RUN pip install --no-cache-dir --timeout 120 \
    mlflow==2.16.2 \
    xgboost==2.1.1 \
    matplotlib==3.9.2 \
    joblib==1.4.2

RUN python -m nltk.downloader -d /home/airflow/nltk_data stopwords punkt punkt_tab

ENV NLTK_DATA=/home/airflow/nltk_data
