#!/usr/bin/env bash
set -euo pipefail

# Install uv if not present
python -m pip install --upgrade pip
python -m pip install uv

# Sync dependencies from pyproject.toml
uv sync

echo "Dependencies synced with uv"
