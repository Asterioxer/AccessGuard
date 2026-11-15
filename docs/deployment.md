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

## 4. Operational Monitoring & Health Checks

AccessGuard exposes standardized health and operational monitoring endpoints:
- `GET /health`: Returns JSON with database connection state, uptime, version, and active CV adapter.
- `GET /metrics`: Returns platform configuration settings, upload thresholds, and privacy flags.
