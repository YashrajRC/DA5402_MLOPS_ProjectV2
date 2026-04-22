#!/bin/bash
# Drops a sample CSV into data/incoming/ to trigger the Airflow DAG
set -e
TS=$(date +%Y%m%d_%H%M%S)
cp scripts/sample_incoming.csv "data/incoming/batch_${TS}.csv"
echo "Dropped batch_${TS}.csv into data/incoming/"
echo "Airflow FileSensor should pick it up within 60 seconds."
