"""Tests for Docker secrets reading functionality."""

import os
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from domain.errors import ConfigurationError
from domain.secrets import load_required_secret, read_docker_secret


class TestReadDockerSecret:
    """Tests for read_docker_secret function."""

    def test_read_existing_secret(self, tmp_path: Path) -> None:
        """Test reading an existing Docker secret file."""
        secret_name = "TEST_SECRET"
        secret_value = "my-secret-value"

        # Create a temporary secrets directory
        secrets_dir = tmp_path / "run" / "secrets"
        secrets_dir.mkdir(parents=True)
        secret_file = secrets_dir / secret_name
        secret_file.write_text(secret_value)

        # Mock the secrets path
        with patch("domain.secrets.Path") as mock_path:
            mock_secret_path = Mock()
            mock_secret_path.is_file.return_value = True
            mock_secret_path.read_text.return_value = secret_value
            mock_path.return_value.__truediv__.return_value = mock_secret_path

            result = read_docker_secret(secret_name)

            assert result == secret_value
            mock_secret_path.read_text.assert_called_once_with(encoding="utf-8")

    def test_read_secret_with_whitespace(self, tmp_path: Path) -> None:
        """Test reading a secret with leading/trailing whitespace."""
        secret_name = "TEST_SECRET"
        secret_value = "  my-secret-value  \n"
        expected_value = "my-secret-value"

        with patch("domain.secrets.Path") as mock_path:
            mock_secret_path = Mock()
            mock_secret_path.is_file.return_value = True
            mock_secret_path.read_text.return_value = secret_value
            mock_path.return_value.__truediv__.return_value = mock_secret_path

            result = read_docker_secret(secret_name)

            assert result == expected_value

    def test_read_nonexistent_secret(self) -> None:
        """Test reading a non-existent Docker secret file."""
        secret_name = "NONEXISTENT_SECRET"

        with patch("domain.secrets.Path") as mock_path:
            mock_secret_path = Mock()
            mock_secret_path.is_file.return_value = False
            mock_path.return_value.__truediv__.return_value = mock_secret_path

            result = read_docker_secret(secret_name)

            assert result is None

    def test_read_empty_secret(self) -> None:
        """Test reading an empty Docker secret file."""
        secret_name = "EMPTY_SECRET"

        with patch("domain.secrets.Path") as mock_path:
            mock_secret_path = Mock()
            mock_secret_path.is_file.return_value = True
            mock_secret_path.read_text.return_value = ""
            mock_path.return_value.__truediv__.return_value = mock_secret_path

            result = read_docker_secret(secret_name)

            assert result is None

    def test_read_whitespace_only_secret(self) -> None:
        """Test reading a Docker secret file with only whitespace."""
        secret_name = "WHITESPACE_SECRET"

        with patch("domain.secrets.Path") as mock_path:
            mock_secret_path = Mock()
            mock_secret_path.is_file.return_value = True
            mock_secret_path.read_text.return_value = "   \n\t  "
            mock_path.return_value.__truediv__.return_value = mock_secret_path

            result = read_docker_secret(secret_name)

            assert result is None


class TestLoadRequiredSecret:
    """Tests for load_required_secret function."""

    def test_load_from_docker_secret(self) -> None:
        """Test loading a secret from Docker secrets (first priority)."""
        secret_name = "API_KEY"
        secret_value = "docker-secret-value"

        with (
            patch("domain.secrets.read_docker_secret", return_value=secret_value),
            patch.dict(os.environ, {secret_name: "env-value"}, clear=False),
        ):
            result = load_required_secret(secret_name)

            assert result == secret_value

    def test_load_from_environment_variable(self) -> None:
        """Test loading a secret from environment variable (fallback)."""
        secret_name = "API_KEY"
        env_value = "env-secret-value"

        with (
            patch("domain.secrets.read_docker_secret", return_value=None),
            patch.dict(os.environ, {secret_name: env_value}, clear=False),
        ):
            result = load_required_secret(secret_name)

            assert result == env_value

    def test_load_missing_secret_default_error(self) -> None:
        """Test loading a missing secret raises ConfigurationError with default message."""
        secret_name = "MISSING_SECRET"

        with (
            patch("domain.secrets.read_docker_secret", return_value=None),
            patch.dict(os.environ, {}, clear=True),
        ):
            with pytest.raises(ConfigurationError, match=f"{secret_name} secret is required"):
                load_required_secret(secret_name)

    def test_load_missing_secret_custom_error(self) -> None:
        """Test loading a missing secret raises ConfigurationError with custom message."""
        secret_name = "MISSING_SECRET"
        custom_message = "Custom error: API key is required for authentication"

        with (
            patch("domain.secrets.read_docker_secret", return_value=None),
            patch.dict(os.environ, {}, clear=True),
        ):
            with pytest.raises(ConfigurationError, match=custom_message):
                load_required_secret(secret_name, error_message=custom_message)

    def test_load_empty_docker_secret_fallback_to_env(self) -> None:
        """Test that empty Docker secret falls back to environment variable."""
        secret_name = "API_KEY"
        env_value = "env-value"

        with (
            patch("domain.secrets.read_docker_secret", return_value=None),
            patch.dict(os.environ, {secret_name: env_value}, clear=False),
        ):
            result = load_required_secret(secret_name)

            assert result == env_value

    def test_load_empty_string_from_env_raises_error(self) -> None:
        """Test that empty string from environment variable raises ConfigurationError."""
        secret_name = "EMPTY_SECRET"

        with (
            patch("domain.secrets.read_docker_secret", return_value=None),
            patch.dict(os.environ, {secret_name: ""}, clear=False),
        ):
            with pytest.raises(ConfigurationError, match=f"{secret_name} secret is required"):
                load_required_secret(secret_name)

    def test_docker_secret_priority_over_env(self) -> None:
        """Test that Docker secret takes priority over environment variable."""
        secret_name = "API_KEY"
        docker_value = "docker-value"
        env_value = "env-value"

        with (
            patch("domain.secrets.read_docker_secret", return_value=docker_value),
            patch.dict(os.environ, {secret_name: env_value}, clear=False),
        ):
            result = load_required_secret(secret_name)

            # Docker secret should be used, not env variable
            assert result == docker_value
            assert result != env_value
