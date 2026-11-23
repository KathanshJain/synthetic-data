# Production image: one FastAPI process serves BOTH the API and the built
# React frontend (frontend/web/dist is mounted at / by api/main.py).
FROM python:3.12-slim

WORKDIR /app

# build the frontend first (node stage) so the final image is python-only
FROM node:22-slim AS web
WORKDIR /web
COPY frontend/web/package*.json ./
RUN npm install
COPY frontend/web/ ./
RUN npm run build

FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt requirements-api.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-api.txt
COPY src/ src/
COPY api/ api/
COPY config.yaml criteria.yaml ./
COPY data/raw/ data/raw/
COPY --from=web /web/dist frontend/web/dist/

# the experiment artefacts (results/, data/processed, data/locked,
# data/synthetic) must be produced by running the pipeline; for the demo
# image we expect them to be volume-mounted:
#   docker run -v ./results:/app/results -v ./data:/app/data ...
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
