# Deployment Guide

> **Experimental-document status:** The container stack and several services described here are incomplete and are not a verified clean-machine deployment path. Do not use this guide as evidence that the thesis prototype is deployed or production-ready. The root README contains the supported local verification steps for the evaluated two-tree benchmark and frontend.

This guide preserves the project's target deployment design for future implementation work.

---

## Prerequisites

### Software

| Dependency | Minimum Version | Purpose |
|------------|-----------------|---------|
| Docker | 24.0+ | Container runtime for all services |
| Docker Compose | v2.20+ | Multi-container orchestration |
| Python | 3.11+ | Training scripts, data ingestion, CLI tools |
| Node.js | 20.0+ | Frontend build toolchain |
| Git | 2.40+ | Source control |

Verify your installations:

```bash
docker --version          # Docker version 24.x or higher
docker compose version    # Docker Compose version v2.20.x or higher
python3 --version         # Python 3.11.x or higher
node --version            # v20.x or higher
```

### API Keys

The system integrates with several free public data APIs. Register for API keys before deployment.

| Service | Key Name | Registration URL | Cost |
|---------|----------|------------------|------|
| NOAA Climate Data Online | CDO Token | https://www.ncdc.noaa.gov/cdo-web/token | Free |
| EIA Open Data | API Key | https://www.eia.gov/opendata/register.php | Free |
| Mapbox | Access Token | https://account.mapbox.com/auth/signup/ | Free tier (50k map loads/month) |

NOAA CDO tokens are typically issued within minutes. EIA keys are issued immediately. The Mapbox free tier is sufficient for development and moderate production use.

### Hardware Recommendations

| Environment | CPU | RAM | Storage |
|-------------|-----|-----|---------|
| Development | 4 cores | 8 GB | 20 GB |
| Staging | 8 cores | 16 GB | 50 GB |
| Production | 16+ cores | 32+ GB | 200+ GB (SSD) |

GPU is optional. The LSTM component benefits from CUDA during training but runs efficiently on CPU for inference.

---

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/your-org/outage-prediction-system.git
cd outage-prediction-system
```

### 2. Configure Environment Variables

Copy the example environment file and fill in your API keys:

```bash
cp .env.example .env
```

Edit `.env` with your credentials:

```ini
# Database
POSTGRES_DB=outage_prediction
POSTGRES_PASSWORD=<choose-a-strong-password>

# API Keys
NOAA_CDO_TOKEN=<your-noaa-token>
EIA_API_KEY=<your-eia-key>
MAPBOX_ACCESS_TOKEN=<your-mapbox-token>

# Redis
REDIS_URL=redis://redis:6379/0

# MLflow
MLFLOW_TRACKING_URI=http://mlflow:5000

# API
API_HOST=0.0.0.0
API_PORT=8000
LOG_LEVEL=info

# Frontend
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_MAPBOX_TOKEN=<your-mapbox-token>
```

### 3. Start All Services

```bash
cd docker
docker compose up -d
```

This starts five services:
- **db**: PostgreSQL 16 with TimescaleDB and PostGIS (port 5432)
- **redis**: Redis 7 (port 6379)
- **mlflow**: MLflow tracking server (port 5000)
- **api**: FastAPI backend (port 8000)
- **frontend**: Next.js dashboard (port 3000)

Monitor startup progress:

```bash
docker compose logs -f
```

Wait until all health checks pass (typically 30-60 seconds):

```bash
docker compose ps
```

All services should show `healthy` or `running` status.

### 4. Verify Services

Check the API health endpoint:

```bash
curl http://localhost:8000/api/v1/health
```

Expected response:

```json
{
  "status": "degraded",
  "db_connected": true,
  "redis_connected": true,
  "model_loaded": false,
  "active_models": [],
  "uptime_seconds": 45.2
}
```

The status is `degraded` because no model has been trained yet. This is expected at this stage.

Verify the frontend is accessible at `http://localhost:3000`.

Verify MLflow is accessible at `http://localhost:5000`.

Verify the API interactive docs are accessible at `http://localhost:8000/docs`.

---

## Initial Data Backfill

Before training, ingest historical data from all sources. The ingestion scripts require a Python environment with the project dependencies.

### Set Up the Python Environment

```bash
python3 -m venv venv
source venv/bin/activate        # Linux/macOS
# venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

### Run Ingestion

Run the data ingestion scripts to populate the database with historical weather events, outage observations, and grid load data:

```bash
# NOAA Storm Events (historical weather)
python -m backend.src.ingestion.noaa_storms --region TX --start-year 2019 --end-year 2024

# EAGLE-I Outage Data (historical outages)
python -m backend.src.ingestion.eagle_i --region TX --backfill-days 365

# ERCOT Grid Load
python -m backend.src.ingestion.ercot --region TX --backfill-days 365

# NWS Active Alerts
python -m backend.src.ingestion.nws_alerts --region TX

# Census Socioeconomic Data
python -m backend.src.ingestion.census --region TX

# EIA Utility Territories
python -m backend.src.ingestion.eia --region TX
```

The NOAA backfill is the longest-running operation (5-15 minutes depending on the region and year range). EAGLE-I and ERCOT backfills run in 2-5 minutes. Census and EIA data are relatively small and complete in under a minute.

### Build the Feature Store

After ingestion, materialize the feature store for training:

```bash
python -m backend.src.features.feature_store --region TX --output data/processed/features_tx.csv
```

This computes temporal, spatial, compound, and socioeconomic features for every H3 cell and timestamp combination, producing a CSV suitable for model training.

---

## Model Training

Run the training pipeline to train XGBoost, LightGBM, and LSTM models, build the stacking ensemble, and compute uncertainty estimates:

```bash
python scripts/run_training.py \
  --region TX \
  --data-dir data/processed \
  --model-dir models \
  --version v1.0.0 \
  --n-optuna-trials 50
```

The training pipeline performs the following steps:

1. Loads the feature dataset and performs a temporal train/validation/test split.
2. Trains XGBoost with early stopping on the validation set (default: 500 estimators, max depth 6, learning rate 0.05).
3. Trains LightGBM with early stopping on the validation set (default: 500 estimators, 63 leaves, learning rate 0.05).
4. Trains an LSTM with attention (hidden dim 128, 2 layers, dropout 0.3, sequence length 24, up to 100 epochs with patience 10).
5. Fits the stacking meta-learner (logistic regression) on validation predictions.
6. Computes uncertainty estimates via MC Dropout (50 samples) and ensemble disagreement.
7. Calibrates probabilities with isotonic regression.
8. Reports final metrics with bootstrap 95% confidence intervals.
9. Saves all models to `models/<region>/` and promotes them in the registry.

Training time varies by dataset size and hardware:
- CPU only: 15-45 minutes
- With CUDA GPU: 5-15 minutes

### Verify Training

After training, the models are saved to `models/TX/` and registered in MLflow. Check MLflow at `http://localhost:5000` to review experiment runs, compare metrics, and inspect logged artifacts.

Restart the API server to load the trained models:

```bash
cd docker
docker compose restart api
```

Check health again:

```bash
curl http://localhost:8000/api/v1/health
```

The response should now show `"status": "healthy"` and `"model_loaded": true`.

---

## Verification Checklist

Run through this checklist to confirm the deployment is fully operational:

- [ ] `docker compose ps` shows all services as healthy/running
- [ ] `curl http://localhost:8000/api/v1/health` returns `"status": "healthy"` with `"model_loaded": true`
- [ ] `curl http://localhost:8000/api/v1/health/models` shows three models (xgboost, lightgbm, lstm) with `"has_meta_learner": true`
- [ ] `curl http://localhost:8000/api/v1/admin/config` shows the correct active regions and thresholds
- [ ] A test prediction returns a valid result:
  ```bash
  curl -X POST http://localhost:8000/api/v1/predictions/realtime \
    -H "Content-Type: application/json" \
    -d '{"h3_cell": "872830828ffffff", "region": "TX"}'
  ```
- [ ] Frontend dashboard loads at `http://localhost:3000` and displays the risk map
- [ ] MLflow UI is accessible at `http://localhost:5000` and shows training runs
- [ ] Historical data endpoints return records:
  ```bash
  curl "http://localhost:8000/api/v1/historical/outages?region=48&limit=5"
  curl "http://localhost:8000/api/v1/historical/weather?region=48&limit=5"
  ```

---

## Production Considerations

### SSL/TLS Termination

In production, terminate TLS at a reverse proxy (nginx, Caddy, or a cloud load balancer) in front of the FastAPI server. Do not expose the API or frontend over plain HTTP.

Example nginx configuration snippet:

```nginx
server {
    listen 443 ssl;
    server_name api.outage-prediction.example.com;

    ssl_certificate     /etc/ssl/certs/fullchain.pem;
    ssl_certificate_key /etc/ssl/private/privkey.pem;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    location /api/v1/alerts/stream {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

### Secrets Management

- Never commit `.env` files or API keys to version control.
- Use a secrets manager (AWS Secrets Manager, HashiCorp Vault, or Docker Secrets) for production credentials.
- Rotate the `POSTGRES_PASSWORD` and API keys periodically.
- Set a dedicated `API_KEY` for the outage prediction API itself and distribute keys to authorized consumers only.

### Monitoring

- **Application metrics**: The FastAPI middleware logs every request with method, path, status code, and duration in structured JSON format. Forward logs to a centralized logging system (ELK, Grafana Loki, or Datadog).
- **Infrastructure metrics**: Monitor container resource usage via `docker stats`, Prometheus with cAdvisor, or your cloud provider's container monitoring.
- **Model monitoring**: Track prediction distributions and feature drift over time using MLflow. Set up alerts for significant shifts in mean risk probability or feature value distributions.
- **Database monitoring**: Monitor TimescaleDB chunk sizes, query latency, and connection pool utilization. Enable `pg_stat_statements` for slow query detection.

### Scaling

- **API horizontal scaling**: Run multiple API container replicas behind a load balancer. The API is stateless except for the in-memory ensemble model (loaded at startup per instance). All shared state (predictions, alerts) is persisted in PostgreSQL.
- **Worker scaling**: Run multiple worker instances with a shared Redis broker. Use Celery or ARQ with per-region task routing to distribute batch inference across workers.
- **Database scaling**: TimescaleDB supports read replicas for query-heavy workloads. Configure continuous aggregates for common analytical queries (hourly/daily outage summaries).
- **Redis scaling**: For high pub/sub throughput, switch to Redis Cluster or a managed Redis service. Feature cache can be sharded by H3 cell prefix.

### Backup and Recovery

- **Database**: Configure automated daily backups of PostgreSQL using `pg_dump` or continuous archiving with WAL-G/pgBackRest. Test restore procedures regularly.
- **Model artifacts**: MLflow artifacts are stored on a mounted Docker volume (`mlflow_artifacts`). Back up this volume alongside the database.
- **Configuration**: All configuration files under `config/` are version-controlled and do not require separate backup.

### Updating Models

To deploy a new model version without downtime:

1. Train the new version using `run_training.py` with an incremented `--version` flag.
2. Review metrics in MLflow and compare against the currently active version.
3. Promote the new version via the admin API:
   ```bash
   curl -X POST http://localhost:8000/api/v1/admin/models/promote \
     -H "Content-Type: application/json" \
     -d '{"model_name": "xgboost", "version": "v1.1.0", "region": "TX"}'
   ```
   Repeat for `lightgbm` and `lstm` as needed.
4. Restart the API server to load the new model weights:
   ```bash
   docker compose restart api
   ```
5. Verify via `GET /api/v1/health/models` that the new version is active.
6. Monitor prediction quality over the next 24-48 hours before decommissioning the previous version.
