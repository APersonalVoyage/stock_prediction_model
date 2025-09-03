# Stock Prediction Model

A full-fledged, end-to-end stock prediction project with data ingestion, feature engineering, model training, backtesting, a CLI, and a Streamlit app.

## Quickstart

1. Create a virtual environment and install dependencies:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

2. Run the CLI help:

```bash
python -m src.stock_predictor.cli --help
```

3. Launch the Streamlit app:

```bash
streamlit run streamlit_app.py
```

## Generate ELI5 guides

- Word (.docx):
```bash
python scripts/generate_eli5_doc.py
```

- Google Docs (first time opens a browser to authorize):
```bash
# 1) In Google Cloud Console, create OAuth Client ID (Desktop App)
# 2) Download the JSON and save as google_credentials.json at project root
# 3) Run:
python scripts/generate_google_doc.py
```

This prints the Google Doc URL. Token is cached in `token.json`.

## Generate PDF guide

Create a detailed PDF into `docs/`:

```bash
python scripts/generate_pdf_guide.py
```

## Project Structure

- `src/stock_predictor/`: Core python package (data, features, models, backtesting, CLI)
- `data/`: Raw and processed datasets (gitignored)
- `models/`: Trained models (gitignored)
- `artifacts/`: Outputs, plots, metrics (gitignored)
- `streamlit_app.py`: Interactive UI
- `src/stock_predictor/live/`: Live trading support (Alpaca wrapper, pipeline, settings)
- `src/stock_predictor/scan/`: Market scan (batch predictions and sizing across a universe)

## Disclaimer
This is for educational purposes only. Not financial advice.

## Live trading (Alpaca)

1) Copy `ENV_EXAMPLE.txt` to `.env` and fill in your keys.
2) Paper trading first (default `ALPACA_PAPER=true`).
3) Train a model (RF recommended first). Then:

```bash
python -m src.stock_predictor.cli live-trade AAPL --start 2018-01-01 --end 2025-01-01 --model-type rf --threshold 0.0 --risk-fraction 0.10 --dry-run
```

Remove `--dry-run` to place orders.

## Live trading (India – Zerodha Kite)

1) Copy `ENV_EXAMPLE.txt` to `.env`, fill `KITE_API_KEY`, generate and set `KITE_ACCESS_TOKEN`.
2) Use NSE symbols with exchange prefix, e.g., `NSE:RELIANCE`.
3) Train a model, then dry-run:

```bash
python -m src.stock_predictor.cli live-trade RELIANCE --start 2018-01-01 --end 2025-01-01 --model-type rf --threshold 0.0 --risk-fraction 0.10 --broker kite --data-symbol NSE:RELIANCE --dry-run
```

Remove `--dry-run` to place orders. Note: Quantity rounding is integer-based; adjust for lot sizes if needed.

## Market scan (batch predictions)

1) Prepare a CSV with columns: `yahoo_symbol,broker_symbol,tradingsymbol,exchange`.
   - Example: `universe/sample_nifty.csv`.
2) Train models for each `yahoo_symbol` you want included.
3) Run scan:

```bash
python -m src.stock_predictor.cli market-scan universe/sample_nifty.csv \
  --start 2018-01-01 --end 2025-01-01 --model-type rf \
  --threshold 0.0 --total-equity 500000 --per-position-fraction 0.05 \
  --broker kite --use-broker-price \
  --output-csv artifacts/scan_results.csv
```

This prints a summary and exports suggested qty/allocation per symbol.
