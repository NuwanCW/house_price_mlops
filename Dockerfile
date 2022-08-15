# Base image
FROM python:3.8-slim

# Install dependencies
WORKDIR /mlops
COPY setup.py setup.py
COPY requirements.txt requirements.txt
COPY datasource.yml  datasource.yml 
COPY fastapi-dashboard.json  fastapi-dashboard.json 
COPY prometheus.yml  prometheus.yml
COPY config.monitoring  config.monitoring 
COPY time_print.sh time_print.sh


RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m pip install --upgrade pip setuptools wheel \
    && python3 -m pip install -e . --no-cache-dir \
    && python3 -m pip install protobuf==3.20.1 --no-cache-dir \
    && apt-get purge -y --auto-remove build-essential

# Copy
COPY realtime_data_drift realtime_data_drift
RUN python3 ./realtime_data_drift/setup.py install 
COPY guess_price guess_price
COPY app app
COPY data data
COPY config config
COPY stores stores

# Pull assets from S3
RUN dvc init --no-scm
RUN dvc remote add -d storage stores/blob
RUN dvc pull -f

# Export ports
EXPOSE 8000

# Start app
# ENTRYPOINT ["bash","./time_print.sh"]
ENTRYPOINT ["gunicorn", "-c", "/mlops/app/gunicorn.py", "--preload", "-k", "uvicorn.workers.UvicornWorker", "app.api:app"]