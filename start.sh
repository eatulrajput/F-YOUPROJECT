#!/usr/bin/env bash
# Start Gunicorn, binding to Render-provided $PORT if set
exec gunicorn --bind 0.0.0.0:${PORT:-8000} app:app
