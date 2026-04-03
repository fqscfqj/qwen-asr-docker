#!/usr/bin/env bash
set -euo pipefail

python -m asr_service.preflight
exec uvicorn asr_service.app:app --host "${ASR_HOST:-0.0.0.0}" --port "${ASR_PORT:-8000}"
