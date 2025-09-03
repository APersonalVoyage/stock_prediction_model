from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Load .env if present at project root
load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / ".env", override=False)


@dataclass
class AlpacaSettings:
    api_key: str
    secret_key: str
    base_url: str
    paper: bool


@dataclass
class KiteSettings:
    api_key: str
    access_token: str


def get_alpaca_settings() -> AlpacaSettings:
    api_key = os.getenv("ALPACA_API_KEY", "")
    secret_key = os.getenv("ALPACA_SECRET_KEY", "")
    base_url = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
    paper = os.getenv("ALPACA_PAPER", "true").lower() in {"1", "true", "yes"}
    return AlpacaSettings(api_key=api_key, secret_key=secret_key, base_url=base_url, paper=paper)


def has_alpaca_credentials() -> bool:
    s = get_alpaca_settings()
    return bool(s.api_key and s.secret_key)


def get_kite_settings() -> KiteSettings:
    api_key = os.getenv("KITE_API_KEY", "")
    access_token = os.getenv("KITE_ACCESS_TOKEN", "")
    return KiteSettings(api_key=api_key, access_token=access_token)


def has_kite_credentials() -> bool:
    s = get_kite_settings()
    return bool(s.api_key and s.access_token)
