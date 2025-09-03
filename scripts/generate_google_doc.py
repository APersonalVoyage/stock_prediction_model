from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build

# If modifying these scopes, delete the token.json file.
SCOPES = ["https://www.googleapis.com/auth/documents", "https://www.googleapis.com/auth/drive.file"]

ROOT = Path(__file__).resolve().parents[1]
CREDENTIALS_PATH = ROOT / "client_secret_478092551524-4m932p6ls77rk2up3a4b38k64s60j57l.apps.googleusercontent.com.json"  # Downloaded from Google Cloud Console
TOKEN_PATH = ROOT / "token.json"


def _get_creds() -> Credentials:
    creds: Optional[Credentials] = None
    if TOKEN_PATH.exists():
        creds = Credentials.from_authorized_user_file(str(TOKEN_PATH), SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not CREDENTIALS_PATH.exists():
                raise FileNotFoundError(
                    f"Missing {CREDENTIALS_PATH}. Create OAuth client ID (Desktop app) and download JSON."
                )
            flow = InstalledAppFlow.from_client_secrets_file(str(CREDENTIALS_PATH), SCOPES)
            creds = flow.run_local_server(port=0)
        with TOKEN_PATH.open("w") as token:
            token.write(creds.to_json())
    return creds


def _insert_text(requests, text: str, bold: bool = False, heading: Optional[str] = None):
    if heading:
        requests.append({
            "insertText": {"location": {"index": 1}, "text": heading + "\n"}
        })
        requests.append({
            "updateParagraphStyle": {
                "range": {"startIndex": 1, "endIndex": 1 + len(heading) + 1},
                "paragraphStyle": {"namedStyleType": "HEADING_1"},
                "fields": "namedStyleType",
            }
        })
    requests.append({
        "insertText": {"location": {"index": 1}, "text": text + "\n\n"}
    })


def build_eli5_requests(doc_title: str):
    # We build from bottom up by always inserting at index 1 (top), so order sections reversed
    sections = []

    sections.append((None, "Safety and expectations\n- This is not financial advice.\n- Daily returns are noisy; start small.\n- Paper trade first and add risk controls."))
    sections.append(("Where to change things (scope for modification)", "- features/engineering.py: add indicators, change windows.\n- models/train.py: try new models, tune, walk-forward CV.\n- backtest/backtest.py: sizing, shorting, slippage, multi-asset.\n- data/ingest.py: new data sources or intraday.\n- cli.py & streamlit_app.py: more options, charts."))
    sections.append(("Real-time trading – more detail (next steps)", "- Scheduler: run daily at close.\n- Position sizing: fixed $ or % of cash.\n- Risk: max position, stop-loss, daily loss limit.\n- Logging: save predictions and orders.\n- Paper trading: use broker sandbox first."))
    sections.append(("Real-time trading – ELI5", "Trading live means: get today’s price, make a fresh guess, decide buy or not, place an order.\n- Pick a broker API (Alpaca, IBKR).\n- Store API keys safely.\n- Daily steps: 1) Download latest, 2) Build features, 3) Load model & predict, 4) If prediction > threshold, BUY; else close."))
    sections.append(("How to run things (easy steps)", "- Create venv, install requirements.\n- Train: python -m src.stock_predictor.cli train AAPL --start 2018-01-01 --end 2024-12-31 --model-type rf\n- Backtest: python -m src.stock_predictor.cli run-backtest AAPL --start 2019-01-01 --end 2024-12-31 --threshold 0.0\n- App: streamlit run streamlit_app.py"))
    sections.append(("Backtest – playing pretend", "- Rule: predict > threshold => long; else cash.\n- Add transaction cost.\n- Compare to Buy & Hold.\n- Metrics: final equity, annual return, Sharpe, win rate."))
    sections.append(("Model – the brain", "- RandomForest (default): simple.\n- XGBoost (optional on macOS needs libomp).\n- Time-based split: train/val/test.\n- Saves model to models/ for reuse."))
    sections.append(("Features – making smart numbers", "- SMA/EMA (trend), RSI/MACD (momentum), Bollinger (volatility).\n- Returns over 1/5/10 days.\n- Lags (shift by 1) to avoid future leakage.\n- Target: next-day return."))
    sections.append(("Data – where numbers come from", "- yfinance gets Open/High/Low/Close/Volume.\n- Files cached under data/ so we download once.\n- Handles Yahoo’s MultiIndex columns."))
    sections.append(("The big pieces (folders)", "- data: download & cache prices.\n- features: build indicators.\n- models: train and save.\n- backtest: test simple rules.\n- cli: terminal commands.\n- streamlit_app.py: UI with buttons."))
    sections.append(("What is this?", "A helper robot for stocks. It learns from past prices to guess tomorrow. Then we test a simple trading rule to see if it works."))

    requests = []
    # Create the title first via document creation; we only insert body here
    for title, body in sections:
        if title:
            _insert_text(requests, body, heading=None)
            _insert_text(requests, title, heading=title)  # heading
        else:
            _insert_text(requests, body)
    return requests


def create_google_doc(title: str = "Stock Predictor Framework – ELI5 Guide (Google Docs)") -> str:
    creds = _get_creds()
    docs_service = build("docs", "v1", credentials=creds)

    # Create empty doc
    doc = docs_service.documents().create(body={"title": title}).execute()
    doc_id = doc.get("documentId")

    # Build body requests
    requests = build_eli5_requests(title)

    # Batch update
    docs_service.documents().batchUpdate(documentId=doc_id, body={"requests": requests}).execute()

    print(f"Created Google Doc: https://docs.google.com/document/d/{doc_id}/edit")
    return doc_id


if __name__ == "__main__":
    create_google_doc()
