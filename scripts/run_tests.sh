#!/usr/bin/env bash
set -e

echo "=========================================="
echo "      AccessGuard Automated Test Suite     "
echo "=========================================="

echo "1. Running Backend Pytest Suite..."
export PYTHONPATH=backend
python -m pytest backend/tests -v

echo "2. Running CV Model Evaluation Framework..."
python backend/app/cv/evaluation/run_eval.py

echo "3. Building Frontend & Verifying Typecheck..."
cd frontend
npm run build

echo "=========================================="
echo "All AccessGuard verification checks passed!"
echo "=========================================="
