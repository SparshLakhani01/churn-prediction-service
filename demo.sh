#!/bin/bash
# End-to-end demo: build the venv if needed, train on synthetic data,
# run unit tests, smoke-test the API. All metrics printed are measured
# from this run. No external APIs. Requires curl for the smoke test.
set -e
cd "$(dirname "$0")"

if [ ! -x .venv/bin/python ]; then
  echo "== Creating venv and installing dependencies =="
  python3 -m venv --system-site-packages .venv
  .venv/bin/pip install -q -r requirements.txt
fi

echo "== 1. Train + evaluate (synthetic data) =="
.venv/bin/python train.py

echo ""
echo "== 2. Unit tests =="
.venv/bin/python -m unittest discover -s tests 2>&1 | tail -3

echo ""
echo "== 3. API smoke test =="
.venv/bin/uvicorn api:app --port 8124 --log-level error &
SRV=$!
trap 'kill $SRV 2>/dev/null || true' EXIT
sleep 4
curl -s http://127.0.0.1:8124/health; echo
curl -s -X POST http://127.0.0.1:8124/predict \
  -H "Content-Type: application/json" \
  -d @examples/predict_request.json; echo
echo ""
echo "Demo complete. Metrics: results/metrics.json, run log: runs/"
