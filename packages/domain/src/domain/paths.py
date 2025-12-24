"""Common paths used across the application."""

from pathlib import Path

# Health check directory and heartbeat files
HEALTH_DIR = Path("/health")
OPTION_STORE_HEARTBEAT = HEALTH_DIR / "option_store.heartbeat"
EVENT_STORE_HEARTBEAT = HEALTH_DIR / "event_store.heartbeat"
