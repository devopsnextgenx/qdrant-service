#!/usr/bin/env bash
set -euo pipefail

# Build the package using uv
uv build

echo "Build completed (check dist/ or build/ depending on backend)"
