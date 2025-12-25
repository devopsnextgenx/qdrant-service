import os
try:
    import yaml
except ImportError as e:
    raise ImportError(
        "Missing required dependency 'pyyaml'. Install it with 'pip install pyyaml' or use the project's workflow: 'pip install uv' then 'uv sync' (see README)."
    ) from e
import logging
from pathlib import Path
from typing import Any, Dict

CONFIG_PATH = Path(os.getenv("CONFIG_PATH", Path(__file__).resolve().parents[1] / "config" / "config.yml"))

DEFAULTS: Dict[str, Any] = {
    "environment": "development",
    "logging": {
        "level": "INFO",
        "file": "logs/qdrant-service.log",
        "rotate": {"enabled": True, "max_bytes": 10 * 1024 * 1024, "backup_count": 5},
    },
    "qdrant": {"url": "http://localhost:6333", "timeout": 30, "collections": {"captions": "captions", "stories": "stories"}},
    "embeddings": {"backend": "ollama", "ollama": {"model": "embeddinggemma:latest", "endpoint": "http://localhost:11434", "timeout": 60}},
    "indexing": {"batch_size": 64, "max_tokens": 512, "chunk_overlap": 32, "min_length": 10},
    "search": {"top_k": 10, "score_threshold": 0.0, "max_query_tokens": 128},
}


def load_config(path: Path = CONFIG_PATH) -> Dict[str, Any]:
    if not path.exists():
        logging.getLogger(__name__).warning("Config file %s not found - using defaults", str(path))
        return DEFAULTS.copy()

    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}

    # merge defaults (simple shallow merge for top-level keys; nested maps replaced if present)
    cfg = DEFAULTS.copy()
    for k, v in data.items():
        if isinstance(v, dict) and k in cfg and isinstance(cfg[k], dict):
            merged = cfg[k].copy()
            merged.update(v)
            cfg[k] = merged
        else:
            cfg[k] = v
    return cfg


CONFIG = load_config()


def init_logging():
    cfg = CONFIG.get("logging", {})
    level_name = cfg.get("level", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    log_file = cfg.get("file", "logs/qdrant-service.log")

    # ensure directory exists
    log_dir = Path(log_file).parent
    log_dir.mkdir(parents=True, exist_ok=True)

    # choose handler
    rotate = cfg.get("rotate", {})
    try:
        if rotate.get("enabled", True):
            from logging.handlers import RotatingFileHandler

            handler = RotatingFileHandler(
                log_file, maxBytes=int(rotate.get("max_bytes", 10 * 1024 * 1024)), backupCount=int(rotate.get("backup_count", 5))
            )
        else:
            handler = logging.FileHandler(log_file)
    except Exception:
        handler = logging.FileHandler(log_file)

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    handler.setFormatter(formatter)

    root = logging.getLogger()
    # Avoid adding duplicate handlers if init_logging is called multiple times
    if not any(isinstance(h, type(handler)) and getattr(h, "baseFilename", None) == str(log_file) for h in root.handlers):
        root.addHandler(handler)
    root.setLevel(level)


# Initialize logging eagerly for modules that import CONFIG
init_logging()
