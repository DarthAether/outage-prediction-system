# Deployment Guide

## Prerequisites

- Docker 24+ and Docker Compose v2
- Node.js 20+ (for frontend development)
- Python 3.11+ (for local development)
- Git

## API Keys (all free)

| Service | Registration URL | Purpose |
|---------|-----------------|---------|
| NOAA CDO | https://www.ncdc.noaa.gov/cdo-web/token | Historical climate data |
| EIA | https://www.eia.gov/opendata/register.php | Grid infrastructure |
| Mapbox | https://account.mapbox.com/auth/signup/ | Map tiles (50k free/month) |

## Quick Start

```bash
# Clone repository
git clone https://github.com/DarthAether/outage-prediction-system.git
cd outage-prediction-system

# Configure environment
cp .env.example .env
# Edit .env with your API keys

# Start all services
cd docker
docker-compose up -d

# Verify services are healthy
curl http://localhost:8000/api/v1/health
```

Services will be available at:
- **API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **Frontend**: http://localhost:3000
- **MLflow**: http://localhost:5000

## Initial Data Backfill

```bash
# Ingest NOAA storm events (place CSVs in data/raw/)
python -m backend.src.ingestion.noaa_storms --region TX --start 2020-01-01 --end 2025-12-31

# Fetch NWS active alerts
python -m backend.src.ingestion.nws_alerts --region TX
```

## Model Training

```bash
# Train all models for Texas
python scripts/run_training.py --region TX --version v1.0.0

# Models saved to models/TX/{xgboost,lightgbm,lstm}/v1.0.0/
```

## Verification Checklist

- [ ] `docker-compose ps` shows all services healthy
- [ ] `curl localhost:8000/api/v1/health` returns `{"status": "ok"}`
- [ ] Frontend loads at `localhost:3000`
- [ ] MLflow UI accessible at `localhost:5000`
- [ ] Database has populated weather_events table

## Production Considerations

- **SSL**: Use a reverse proxy (nginx/traefik) for TLS termination
- **Secrets**: Use Docker secrets or a vault service for API keys
- **Monitoring**: Export Prometheus metrics from FastAPI via `prometheus-fastapi-instrumentator`
- **Backups**: Configure TimescaleDB continuous backup with `pgBackRest`
- **Scaling**: API service is stateless; scale horizontally behind a load balancer
