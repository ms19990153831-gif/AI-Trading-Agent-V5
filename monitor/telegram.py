"""Optional Telegram alerting for trade events and production health."""

from __future__ import annotations

from config import TELEGRAM_CHAT_ID, TELEGRAM_TOKEN


class TelegramAlert:
    def __init__(self) -> None:
        self.enabled = bool(TELEGRAM_TOKEN and TELEGRAM_CHAT_ID)

    def send(self, text: str) -> bool:
        if not self.enabled:
            return False
        try:
            import requests

            url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
            response = requests.post(
                url,
                json={"chat_id": TELEGRAM_CHAT_ID, "text": text},
                timeout=10,
            )
            return response.ok
        except Exception:
            return False
