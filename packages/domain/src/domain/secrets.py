import os
from pathlib import Path

from domain.errors import ConfigurationError


def load_required_secret(name: str, error_message: str | None = None) -> str:
    """Load a secret from Docker secrets or environment variables.

    Args:
        name: Name of the secret (e.g. 'POLYMARKET_API_KEY').
              Looks in Docker secrets first, then os.environ.
        error_message: Optional custom error message if secret is missing.

    Returns:
        The secret value string.

    Raises:
        ConfigurationError: If the secret is not found in either location.
    """
    value = read_docker_secret(name) or os.getenv(name)
    if not value:
        msg = error_message or f"{name} secret is required"
        raise ConfigurationError(msg)
    return value


def load_optional_secret(name: str) -> str | None:
    """Load an optional secret from Docker secrets or environment variables.

    Args:
        name: Name of the secret (e.g. 'OPENAI_API_KEY').
              Looks in Docker secrets first, then os.environ.

    Returns:
        The secret value string, or None if not found.
    """
    return read_docker_secret(name) or os.getenv(name)


def read_docker_secret(name: str) -> str | None:
    """Read a Docker secret from /run/secrets."""
    secret_path = Path("/run/secrets") / name

    if not secret_path.is_file():
        return None

    value = secret_path.read_text(encoding="utf-8").strip()
    return value or None
