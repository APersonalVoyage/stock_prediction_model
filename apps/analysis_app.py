import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import glob
import sys

# Ensure package import works when running directly with Streamlit
try:
    from stock_predictor.config import ARTIFACTS_DIR, MODELS_DIR
except Exception:
    ROOT = Path(__file__).resolve().parents[1]
    sys.path.append(str(ROOT / "src"))
    from stock_predictor.config import ARTIFACTS_DIR, MODELS_DIR

st.set_page_config(page_title="Market Scan Dashboard", layout="wide")

st.title("📊 Market Scan & Model Dashboard (Analysis Only)")

@st.cache_data(show_spinner=False)
def list_scan_csvs() -> pd.DataFrame:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    pattern = str(ARTIFACTS_DIR / "scan_*.csv")
    files = glob.glob(pattern)
    files = sorted(files, key=lambda p: Path(p).stat().st_mtime if Path(p).exists() else 0, reverse=True)
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

REQUIRED_COLS = [
    "yahoo_symbol", "broker_symbol", "predicted_return", "last_price",
    "signal_long", "suggested_allocation", "suggested_qty"
]
NUMERIC_COLS = ["predicted_return", "last_price", "suggested_allocation", "suggested_qty"]
BOOL_COLS = ["signal_long"]


def normalize_scan_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Ensure required columns exist
    for c in REQUIRED_COLS:
        if c not in df.columns:
            df[c] = np.nan
    # Coerce types
    for c in NUMERIC_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in BOOL_COLS:
        if c in df.columns:
            df[c] = df[c].astype(str).str.lower().isin(["true", "1", "yes"])
    # Symbol columns to string
    for c in ["yahoo_symbol", "broker_symbol"]:
        df[c] = df[c].astype(str)
    return df

@st.cache_data(show_spinner=False)
def read_scan_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return normalize_scan_df(df)

with st.sidebar:
    st.header("Data Sources")
    scan_df = list_scan_csvs()
    scan_file = None
    if len(scan_df) > 0:
        scan_file = st.selectbox("Select a scan CSV", options=scan_df["file"].tolist(), index=0)
    uploaded = st.file_uploader("...or upload a scan CSV", type=["csv"], accept_multiple_files=False)

    st.divider()
    st.caption("This app is analysis-only. It does not place any trades.")

# Resolve chosen scan
df_scan = None
scan_path = None
if uploaded is not None:
    try:
        df_scan = pd.read_csv(uploaded)
        df_scan = normalize_scan_df(df_scan)
        scan_path = uploaded.name
    except Exception as e:
        st.error(f"Failed to read uploaded CSV: {e}")
elif scan_file:
    try:
        scan_row = scan_df[scan_df["file"] == scan_file].iloc[0]
        scan_path = scan_row["path"]
        df_scan = read_scan_csv(scan_path)
    except Exception as e:
        st.error(f"Failed to open CSV: {e}")

if df_scan is None or len(df_scan) == 0:
    st.info("No scan selected or scan is empty. Add CSVs to artifacts/ or upload one.")
    st.stop()

st.subheader("Selected Scan")
st.write(f"File: {Path(scan_path).name}")

# Filters
c1, c2, c3, c4 = st.columns(4)
with c1:
    only_signal = st.checkbox("Show signal_long only", value=True)
with c2:
    min_pred = st.number_input("Min predicted_return", value=0.0, step=0.001, format="%.3f")
with c3:
    min_qty = st.number_input("Min suggested_qty", value=0, step=1)
with c4:
    sort_by = st.selectbox("Sort by", options=["predicted_return", "suggested_allocation", "suggested_qty"], index=0)

view = df_scan.copy()
if only_signal and "signal_long" in view.columns:
    view = view[view["signal_long"] == True]
if "predicted_return" in view.columns:
    view = view[view["predicted_return"].fillna(-np.inf) >= float(min_pred)]
if "suggested_qty" in view.columns:
    view = view[view["suggested_qty"].fillna(0) >= int(min_qty)]
if sort_by in view.columns:
    try:
        view = view.sort_values(sort_by, ascending=False)
    except Exception:
        pass

left, right = st.columns(2)
with left:
    total_candidates = int((df_scan.get("signal_long") == True).sum()) if "signal_long" in df_scan.columns else 0
    st.metric("Total candidates (signal_long)", total_candidates)
with right:
    selected_qty = int((df_scan.get("suggested_qty", pd.Series([0]*len(df_scan))) > 0).sum())
    st.metric("Selected (qty>0)", selected_qty)

# Summary charts
st.subheader("Summary")
colA, colB = st.columns(2)
with colA:
    if "predicted_return" in view.columns and len(view) > 0:
        chart_df = view[["yahoo_symbol", "predicted_return"]].set_index("yahoo_symbol").head(50)
        st.bar_chart(chart_df, height=300)
with colB:
    if "suggested_allocation" in view.columns and len(view) > 0:
        chart_df = view[["yahoo_symbol", "suggested_allocation"]].set_index("yahoo_symbol").head(50)
        st.bar_chart(chart_df, height=300)

st.subheader("Detailed Table")
st.dataframe(view, use_container_width=True)

# Download filtered view
csv_bytes = view.to_csv(index=False).encode("utf-8")
st.download_button("Download filtered CSV", data=csv_bytes, file_name=f"filtered_{Path(scan_path).name}", mime="text/csv")

st.divider()
st.subheader("Model Files Present")
# Show which models exist for symbols in the scan
symbols = sorted(set(df_scan.get("yahoo_symbol", [])))
model_rows = []
for sym in symbols:
    path = MODELS_DIR / f"{sym.upper()}_rf.joblib"
    model_rows.append({"yahoo_symbol": sym, "rf_model": path.exists(), "path": str(path)})
model_df = pd.DataFrame(model_rows)
st.dataframe(model_df, use_container_width=True)
