#!/bin/bash
source .venv/bin/activate
nohup uvicorn "app.main:app" --host 0.0.0.0 --port 1234 --log-level info &
