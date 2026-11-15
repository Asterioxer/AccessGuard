# AccessGuard Deployment & Operations Guide

## 1. Prerequisites

- **Docker**: Version 20.10+ (Recommended: 24+)
- **Docker Compose**: Version 2.0+
- **Local Dev (Optional)**: Python 3.12+, Node.js 20+, npm 10+

---

## 2. Docker Compose Deployment (Recommended)

The easiest and most reliable method to deploy AccessGuard in production or local evaluation is via Docker Compose, which orchestrates the PostgreSQL + PostGIS database, FastAPI backend, and Nginx-served React frontend.

### Step 1: Clone Repository & Configure Environment
```bash
git clone https://github.com/accessguard/accessguard.git
cd accessguard
cp .env.example .env
```

Review and update `.env` with secure production keys:
```ini
SECRET_KEY=generate-a-strong-random-32-byte-secret-key-here
DATABASE_URL=postgresql+psycopg2://accessguard:accessguard_secret_db_pass@db:5432/accessguard_db
CV_ADAPTER=heuristic
ENABLE_PRIVACY_BLURRING=true
STRIP_EXIF_AFTER_EXTRACTION=true
```

### Step 2: Launch All Services
```bash
docker compose up -d --build
```

### Step 3: Verify Container Health
```bash
docker compose ps
```

All three services should display healthy status:
- `accessguard_db`: Running on port `5432`
- `accessguard_backend`: Running on port `8000`
- `accessguard_frontend`: Running on port `3000` (mapped to port 80 internally)

### Step 4: Seed Initial Demonstration Data
```bash
docker compose exec backend python /app/scripts/seed_demo_data.py
```

### Step 5: Access the Platform
- **Web Dashboard**: http://localhost:3000
- **API Swagger Documentation**: http://localhost:8000/docs
- **Health Endpoint**: http://localhost:8000/health

---

## 3. Local Development Deployment (Without Docker)

### Backend Setup:
```bash
cd backend
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
pip install email-validator

# Run database migrations / seed
python ../scripts/seed_demo_data.py

# Launch FastAPI server with auto-reload
uvicorn app.main:app --reload --port 8000
```

### Frontend Setup:
```bash
cd frontend
npm install
npm run dev
```

Visit http://localhost:5173 to access the development server.

---

## 4. Operational Monitoring, Metrics & Migrations

### Database Migrations via Alembic
AccessGuard uses versioned Alembic migrations to manage schema evolutions:
```bash
# Apply pending migrations to latest revision
alembic upgrade head

# Rollback one migration revision
alembic downgrade -1

# Generate new migration following model changes
alembic revision --autogenerate -m "Add new barrier attributes"
```

### Prometheus Observability (`/metrics`)
The backend exposes Prometheus metrics at `GET /metrics` in Prometheus text format (RFC 0001):
- `accessguard_http_requests_total`: Counter partitioned by method, endpoint, and HTTP response status.
- `accessguard_cv_inference_seconds`: Histogram measuring inference and feature extraction durations.
- Python runtime memory, garbage collection, and process statistics.

Sample Prometheus scrape config:
```yaml
scrape_configs:
  - job_name: 'accessguard-backend'
    scrape_interval: 15s
    metrics_path: '/metrics'
    static_configs:
      - targets: ['backend:8000']
```

---

## 5. Database Backup & Disaster Recovery

### PostgreSQL + PostGIS Backup
```bash
# Create compressed logical backup
docker compose exec -t db pg_dump -U accessguard -d accessguard_db -F c -b -v -f /tmp/backup.dump
docker cp accessguard_db:/tmp/backup.dump ./backups/$(date +%Y%m%d_%H%M%S)_accessguard.dump

# Restore from backup
docker cp ./backups/latest_backup.dump accessguard_db:/tmp/restore.dump
docker compose exec -t db pg_restore -U accessguard -d accessguard_db -c -v /tmp/restore.dump
```

### Rollback Strategy
1. **Container Rollback:** If a newly deployed backend container fails health checks, Docker Compose preserves the previous container image. To roll back: `docker compose up -d --no-build <service_name>:<previous_tag>`.
2. **Database Rollback:** Run `alembic downgrade -1` before redeploying the prior backend release.
