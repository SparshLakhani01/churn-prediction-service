# syntax=docker/dockerfile:1
# Churn prediction service. Trains on synthetic data at container
# start (fast: seconds), then serves the FastAPI app. No model
# artifact is baked into the image; the model is regenerated from
# the fixed-seed synthetic generator on every start, so the image
# is self-contained and reproducible.
#
# NOTE: this Dockerfile was written but NOT verified by a build in
# this environment (Docker is not installed here). Build and run:
#   docker build -t churn-service .
#   docker run -p 8000:8000 churn-service
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src/ ./src/
COPY train.py api.py ./
COPY examples/ ./examples/
EXPOSE 8000
CMD ["sh", "-c", "python train.py --quiet && uvicorn api:app --host 0.0.0.0 --port 8000"]
