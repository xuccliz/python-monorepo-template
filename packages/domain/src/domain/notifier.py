"""Discord notification integration."""

import logging
from typing import Literal

import requests

from .secrets import load_optional_secret

logger = logging.getLogger(__name__)

NotificationLevel = Literal["info", "warning", "error", "success"]


def _format_message(message: str, level: NotificationLevel) -> dict:
    """Format message for Discord webhook."""
    color_map = {
        "info": 0x3498DB,  # Blue
        "success": 0x2ECC71,  # Green
        "warning": 0xF39C12,  # Orange
        "error": 0xE74C3C,  # Red
    }

    emoji_map = {
        "info": "ℹ️",
        "success": "✅",
        "warning": "⚠️",
        "error": "❌",
    }

    return {
        "embeds": [
            {
                "description": f"{emoji_map[level]} {message}",
                "color": color_map[level],
            }
        ]
    }


def send_notification(message: str, level: NotificationLevel = "info") -> None:
    """
    Send a notification to Discord.

    Args:
        message: Message to send
        level: Notification level (info, warning, error, success)
    """
    webhook_url = load_optional_secret("DISCORD_WEBHOOK_URL")
    if not webhook_url:
        logger.debug("Discord webhook URL not configured, skipping notification")
        return

    try:
        payload = _format_message(message, level)
        response = requests.post(webhook_url, json=payload, timeout=5)
        response.raise_for_status()
        logger.debug(f"Discord notification sent: {level}")
    except requests.RequestException as e:
        logger.warning(f"Failed to send Discord notification: {e}")
    except Exception as e:
        logger.error(f"Unexpected error sending Discord notification: {e}")
