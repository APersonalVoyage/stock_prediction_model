from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from .config import MODELS_DIR
from .data.ingest import download_prices
from .features.engineering import build_training_frame, split_features_target
from .models.train import TrainResult, load_model, train_ticker
from .models.evaluate import evaluate_saved_model
from .backtest.backtest import backtest
from .live.pipeline import LiveConfig, run_live
from .live.settings import has_alpaca_credentials
from .scan.market_scan import ScanConfig, scan_market
from .portfolio.rebalance import rebalance_from_scan

app = typer.Typer(add_completion=False, help="Stock predictor CLI")
console = Console()


@app.command()
def download(
    ticker: str = typer.Argument(..., help="Ticker symbol, e.g., AAPL"),
    start: str = typer.Option("2015-01-01", help="Start date YYYY-MM-DD"),
    end: str = typer.Option("2025-01-01", help="End date YYYY-MM-DD"),
    interval: str = typer.Option("1d", help="Data interval: 1d, 1h, etc."),
    no_cache: bool = typer.Option(False, help="Disable cache and force download"),
):
    df = download_prices(ticker=ticker, start=start, end=end, interval=interval, use_cache=not no_cache)
    console.print(f"Downloaded {len(df)} rows for [bold]{ticker}[/bold].")


@app.command()
def train(
    ticker: str = typer.Argument(..., help="Ticker symbol"),
    start: str = typer.Option("2015-01-01", help="Start date"),
    end: str = typer.Option("2025-01-01", help="End date"),
    model_type: str = typer.Option("rf", help="Model type: rf or xgb"),
    save_model: bool = typer.Option(True, help="Persist model to disk"),
):
    result: TrainResult = train_ticker(ticker=ticker, start=start, end=end, model_type=model_type, save_model=save_model)

    table = Table(title="Training Metrics")
    table.add_column("Metric")
    table.add_column("Value")
    for k, v in result.metrics.items():
        table.add_row(k, f"{v:.6f}")

    console.print(table)
    console.print(f"Model saved to: [green]{result.model_path}[/green]")


@app.command()
def evaluate(
    ticker: str = typer.Argument(..., help="Ticker symbol"),
    start: str = typer.Option("2015-01-01", help="Start date"),
    end: str = typer.Option("2025-01-01", help="End date"),
    model_path: Optional[Path] = typer.Option(None, help="Path to saved model"),
):
    if model_path is None:
        # default to RF model if present, else XGB
        rf = MODELS_DIR / f"{ticker.upper()}_rf.joblib"
        xgb = MODELS_DIR / f"{ticker.upper()}_xgb.joblib"
        model_path = rf if rf.exists() else xgb
    metrics = evaluate_saved_model(model_path=model_path, ticker=ticker, start=start, end=end)

    table = Table(title="Evaluation Metrics")
    table.add_column("Metric")
    table.add_column("Value")
    for k, v in metrics.items():
        table.add_row(k, f"{v:.6f}")
    console.print(table)


@app.command()
def market_scan(
    universe_csv: Path = typer.Argument(..., help="CSV with columns: yahoo_symbol,broker_symbol,tradingsymbol,exchange"),
    start: str = typer.Option("2018-01-01", help="Start date for features"),
    end: str = typer.Option("2025-01-01", help="End date for features"),
    model_type: str = typer.Option("rf", help="Model type used for saved models"),
    threshold: float = typer.Option(0.0, help="Long signal threshold"),
    total_equity: float = typer.Option(100000.0, help="Total equity for sizing"),
    per_position_fraction: float = typer.Option(0.05, help="Max fraction per position (e.g., 0.05 = 5%)"),
    top_k: int = typer.Option(20, help="Take top-K by predicted_return above threshold"),
    broker: str = typer.Option("kite", help="Broker for price lookup: kite or alpaca"),
    use_broker_price: bool = typer.Option(True, help="Use broker last price for today's tradable signal; fallback to Yahoo if unavailable"),
    output_csv: Optional[Path] = typer.Option(None, help="If provided, export detailed results to this CSV"),
):
    cfg = ScanConfig(
        universe_csv=universe_csv,
        start=start,
        end=end,
        model_type=model_type,
        threshold=threshold,
        total_equity=total_equity,
        per_position_fraction=per_position_fraction,
        top_k=top_k,
        broker=broker,
        use_broker_price=use_broker_price,
    )
    df, summary = scan_market(cfg)

    table = Table(title="Market Scan Summary")
    table.add_column("Metric")
    table.add_column("Value")
    for k, v in summary.iloc[0].items():
        table.add_row(str(k), f"{v}")
    console.print(table)

    if output_csv:
        df.to_csv(output_csv, index=False)
        console.print(f"Exported results to: [green]{output_csv}[/green]")


@app.command()
def rebalance(
    scan_csv: Path = typer.Argument(..., help="CSV produced by market-scan with suggested_qty/allocation"),
    broker: str = typer.Option("kite", help="Broker: kite (supported)"),
    dry_run: bool = typer.Option(True, help="If true, simulate orders only"),
    max_per_order_qty: int = typer.Option(0, help="If >0, cap each order's absolute qty"),
):
    summary, plans = rebalance_from_scan(
        scan_csv=scan_csv,
        broker=broker,
        dry_run=dry_run,
        max_per_order_qty=max_per_order_qty if max_per_order_qty > 0 else None,
    )

    table = Table(title="Rebalance Summary")
    table.add_column("Metric")
    table.add_column("Value")
    for k, v in {
        "planned": summary.planned,
        "submitted": summary.submitted,
        "skipped": summary.skipped,
    }.items():
        table.add_row(str(k), str(v))
    console.print(table)

    # Show first 20 plans for visibility
    plan_table = Table(title="Plans (first 20)")
    for col in ["broker_symbol", "action", "delta", "message"]:
        plan_table.add_column(col)
    for p in plans[:20]:
        plan_table.add_row(str(p.broker_symbol), p.action, str(p.delta), p.message)
    console.print(plan_table)

@app.command()
def live_trade(
    ticker: str = typer.Argument(..., help="Ticker symbol"),
    start: str = typer.Option("2018-01-01", help="Start date to build features"),
    end: str = typer.Option("2025-01-01", help="End date to build features"),
    model_type: str = typer.Option("rf", help="Model type: rf or xgb"),
    threshold: float = typer.Option(0.0, help="Predicted next-day return threshold to go long"),
    risk_fraction: float = typer.Option(0.10, help="Fraction of account equity to allocate when long"),
    dry_run: bool = typer.Option(True, help="If true, do not place orders"),
    simulate_equity: float = typer.Option(100000.0, help="Equity to use in dry-run without broker creds"),
    broker: str = typer.Option("alpaca", help="Broker: alpaca or kite"),
    data_symbol: Optional[str] = typer.Option(None, help="Override data/exec symbol (e.g., NSE:RELIANCE for Kite)"),
):
    if dry_run and not has_alpaca_credentials():
        console.print("[yellow]Dry-run: proceeding without Alpaca credentials.[/yellow]")
    elif broker == "alpaca" and not has_alpaca_credentials():
        console.print("[red]Missing Alpaca credentials. Set ALPACA_API_KEY and ALPACA_SECRET_KEY.[/red]")
        raise typer.Exit(1)

    cfg = LiveConfig(
        ticker=ticker,
        start=start,
        end=end,
        model_type=model_type,
        threshold=threshold,
        risk_fraction=risk_fraction,
        dry_run=dry_run,
        simulate_equity=simulate_equity,
        broker=broker,
        data_symbol=data_symbol,
    )
    decision = run_live(cfg)

    table = Table(title="Live Decision")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("predicted_return", f"{decision.predicted_return:.6f}")
    table.add_row("signal_long", str(decision.signal_long))
    table.add_row("qty", f"{decision.qty:.0f}")
    table.add_row("action", decision.action)
    table.add_row("last_price", f"{decision.price:.2f}")
    console.print(table)

@app.command()
def run_backtest(
    ticker: str = typer.Argument(..., help="Ticker symbol"),
    start: str = typer.Option("2018-01-01", help="Start date"),
    end: str = typer.Option("2025-01-01", help="End date"),
    threshold: float = typer.Option(0.0, help="Long entry threshold on predicted next-day return"),
    transaction_cost: float = typer.Option(0.0005, help="Per-trade transaction cost"),
    model_path: Optional[Path] = typer.Option(None, help="Path to saved model"),
):
    if model_path is None:
        rf = MODELS_DIR / f"{ticker.upper()}_rf.joblib"
        xgb = MODELS_DIR / f"{ticker.upper()}_xgb.joblib"
        model_path = rf if rf.exists() else xgb

    result = backtest(
        model_path=model_path,
        ticker=ticker,
        start=start,
        end=end,
        threshold=threshold,
        transaction_cost=transaction_cost,
    )

    table = Table(title="Backtest Summary")
    table.add_column("Metric")
    table.add_column("Value")
    for k, v in result.summary.items():
        table.add_row(k, f"{v:.6f}")
    console.print(table)


if __name__ == "__main__":
    app()
