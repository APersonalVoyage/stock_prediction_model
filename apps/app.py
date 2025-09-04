import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import glob
from datetime import date

# Safe imports so it works locally and on Streamlit Cloud without manual PYTHONPATH
try:
    from stock_predictor.config import ARTIFACTS_DIR, MODELS_DIR
    from stock_predictor.scan.market_scan import ScanConfig, scan_market
    from stock_predictor.portfolio.rebalance import rebalance_from_scan
    from stock_predictor.models.train import train_ticker
    from stock_predictor.models.evaluate import evaluate_saved_model
except Exception:
    ROOT = Path(__file__).resolve().parents[1]
    sys.path.append(str(ROOT / "src"))
    from stock_predictor.config import ARTIFACTS_DIR, MODELS_DIR
    from stock_predictor.scan.market_scan import ScanConfig, scan_market
    from stock_predictor.portfolio.rebalance import rebalance_from_scan
    from stock_predictor.models.train import train_ticker
    from stock_predictor.models.evaluate import evaluate_saved_model


st.set_page_config(page_title="Market Toolkit", layout="wide")
st.title("🧰 Market Toolkit: Scan • Portfolio • Models • Dashboard")


def list_scan_csvs() -> pd.DataFrame:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(glob.glob(str(ARTIFACTS_DIR / "scan_*.csv")), key=lambda p: Path(p).stat().st_mtime if Path(p).exists() else 0, reverse=True)
    rows = []
    for p in files:
        try:
            path = Path(p)
            rows.append({
                "file": path.name,
                "path": str(path),
                "modified": pd.to_datetime(path.stat().st_mtime, unit="s"),
            })
        except Exception:
            continue
    return pd.DataFrame(rows)


def normalize_scan_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    required = ["yahoo_symbol", "broker_symbol", "predicted_return", "last_price", "signal_long", "suggested_allocation", "suggested_qty"]
    numeric_cols = ["predicted_return", "last_price", "suggested_allocation", "suggested_qty"]
    for c in required:
        if c not in df.columns:
            df[c] = np.nan
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if "signal_long" in df.columns:
        df["signal_long"] = df["signal_long"].astype(str).str.lower().isin(["true", "1", "yes"])
    for c in ["yahoo_symbol", "broker_symbol"]:
        df[c] = df[c].astype(str)
    return df


tabs = st.tabs(["Scan", "Portfolio", "Models", "Dashboard"])


# Scan Tab
with tabs[0]:
    st.header("Market Scan")
    c1, c2 = st.columns(2)
    with c1:
        st.write("Universe CSV (yahoo_symbol,broker_symbol,tradingsymbol,exchange)")
        # Choose from repo universe dir or upload
        uni_files = sorted(glob.glob(str(Path(__file__).resolve().parents[1] / "universe" / "*.csv")))
        uni_choice = st.selectbox("Select universe file", options=[Path(p).name for p in uni_files], key="scan_universe_file") if uni_files else None
        uploaded_uni = st.file_uploader("...or upload universe CSV", type=["csv"], accept_multiple_files=False)
    with c2:
        st.write("Parameters")
        start = st.date_input("Start", value=date(2018, 1, 1))
        end = st.date_input("End", value=date.today())
        model_type = st.selectbox("Model type", options=["rf", "xgb"], index=0, key="scan_model_type")
        threshold = st.number_input("Signal threshold (predicted_return)", value=0.0, step=0.001, format="%.3f")
        total_equity = st.number_input("Total equity", value=500000.0, step=10000.0, format="%.2f")
        per_frac = st.number_input("Max per-position fraction", value=0.05, step=0.01, format="%.2f")
        top_k = st.number_input("Top K", value=20, step=1)
        broker = st.selectbox("Broker for price", options=["kite", "alpaca"], index=0, key="scan_broker")
        use_broker_price = st.checkbox("Use broker price (today)", value=True)

    run_scan = st.button("Run Scan")
    if run_scan:
        # Resolve universe path
        if uploaded_uni is not None:
            uni_df = pd.read_csv(uploaded_uni)
            uni_path = ARTIFACTS_DIR / "_uploaded_universe.csv"
            ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
            uni_df.to_csv(uni_path, index=False)
        elif uni_choice:
            uni_path = Path(__file__).resolve().parents[1] / "universe" / uni_choice
        else:
            st.error("Please select or upload a universe CSV.")
            st.stop()

        cfg = ScanConfig(
            universe_csv=uni_path,
            start=str(start),
            end=str(end),
            model_type=model_type,
            threshold=float(threshold),
            total_equity=float(total_equity),
            per_position_fraction=float(per_frac),
            top_k=int(top_k),
            broker=broker,
            use_broker_price=bool(use_broker_price),
        )
        df, summary = scan_market(cfg)
        st.subheader("Summary")
        st.dataframe(summary, use_container_width=True)
        st.subheader("Results")
        st.dataframe(df, use_container_width=True)

        # Save and offer download
        out_name = f"scan_{str(end)}.csv"
        out_path = ARTIFACTS_DIR / out_name
        df.to_csv(out_path, index=False)
        st.success(f"Saved: {out_path}")
        st.download_button("Download CSV", data=df.to_csv(index=False).encode("utf-8"), file_name=out_name, mime="text/csv")


# Portfolio Tab
with tabs[1]:
    st.header("Portfolio Rebalance (Kite)")
    scans = list_scan_csvs()
    chosen = st.selectbox("Select scan CSV", options=scans["file"].tolist() if len(scans) else [], key="portfolio_scan_csv") if len(scans) else None
    uploaded_scan = st.file_uploader("...or upload scan CSV", type=["csv"], accept_multiple_files=False, key="scan_upload")
    dry_run = st.checkbox("Dry run (simulate only)", value=True)
    max_per_order_qty = st.number_input("Max per-order qty (0 = no cap)", value=0, step=1)

    do_rebalance = st.button("Run Rebalance")
    if do_rebalance:
        if uploaded_scan is not None:
            df_scan = pd.read_csv(uploaded_scan)
            scan_path = ARTIFACTS_DIR / "_uploaded_scan.csv"
            df_scan.to_csv(scan_path, index=False)
        elif chosen:
            scan_path = Path(scans[scans["file"] == chosen].iloc[0]["path"])
        else:
            st.error("Please select or upload a scan CSV.")
            st.stop()

        summary, plans = rebalance_from_scan(
            scan_csv=scan_path,
            broker="kite",
            dry_run=dry_run,
            max_per_order_qty=int(max_per_order_qty) if max_per_order_qty > 0 else None,
        )
        st.subheader("Summary")
        st.json({"planned": summary.planned, "submitted": summary.submitted, "skipped": summary.skipped})
        st.subheader("Plans (first 50)")
        st.dataframe(pd.DataFrame([p.__dict__ for p in plans[:50]]), use_container_width=True)


# Models Tab
with tabs[2]:
    st.header("Models: Train & Evaluate")
    m1, m2 = st.columns(2)
    with m1:
        ticker = st.text_input("Yahoo symbol", value="RELIANCE.NS")
        start_m = st.date_input("Train start", value=date(2018, 1, 1), key="m_start")
        end_m = st.date_input("Train end", value=date(2024, 12, 31), key="m_end")
        model_type_m = st.selectbox("Model type", ["rf", "xgb"], index=0, key="models_model_type")
        if st.button("Train model"):
            res = train_ticker(ticker=ticker, start=str(start_m), end=str(end_m), model_type=model_type_m, save_model=True)
            st.success(f"Saved: {res.model_path}")
            st.json(res.metrics)
    with m2:
        e_start = st.date_input("Eval start", value=date(2023, 1, 1), key="e_start")
        e_end = st.date_input("Eval end", value=date(2024, 12, 31), key="e_end")
        if st.button("Evaluate model"):
            # Prefer RF file; fallback to XGB
            rf = MODELS_DIR / f"{ticker.upper()}_rf.joblib"
            xgb = MODELS_DIR / f"{ticker.upper()}_xgb.joblib"
            mpath = rf if rf.exists() else xgb
            import os
            if not mpath.exists():
                st.error("Model file not found. Train first.")
            else:
                metrics = evaluate_saved_model(model_path=mpath, ticker=ticker, start=str(e_start), end=str(e_end))
                st.json(metrics)


# Dashboard Tab (analysis)
with tabs[3]:
    st.header("Dashboard: View Scan Files")
    scans = list_scan_csvs()
    if len(scans) == 0:
        st.info("No scans found. Run a scan in the Scan tab.")
    else:
        chosen = st.selectbox("Select scan CSV", options=scans["file"].tolist(), key="dashboard_scan_csv")
        path = scans[scans["file"] == chosen].iloc[0]["path"]
        df = pd.read_csv(path)
        df = normalize_scan_df(df)

        st.subheader("Summary")
        total_candidates = int((df.get("signal_long") == True).sum())
        selected_qty = int((df.get("suggested_qty", pd.Series([0]*len(df))) > 0).sum())
        st.json({"candidates": total_candidates, "selected": selected_qty, "capital_used": float(df.get("suggested_allocation", pd.Series([0]*len(df))).sum())})

        st.subheader("Top by predicted_return")
        show = df[df["signal_long"] == True].sort_values("predicted_return", ascending=False).head(50)
        st.dataframe(show, use_container_width=True)

        st.subheader("Download")
        st.download_button("Download CSV", data=df.to_csv(index=False).encode("utf-8"), file_name=Path(path).name, mime="text/csv")


