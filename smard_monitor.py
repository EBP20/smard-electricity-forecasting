import joblib
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone

# Config

MODEL_PATH      = Path("smard_xgb_model.pkl")
LOG_PATH        = Path("monitoring_log.json")

# Alert thresholds
MAE_THRESHOLD       = 20.0   # €/MWh — alert if MAE exceeds this
MAE_STD_THRESHOLD   =  0.40  # alert if MAE/STD exceeds this (40%)
REGIME_THRESHOLD    = 30.0   # €/MWh — alert if recent vs historical mean differs

# Rolling window for performance tracking
PERFORMANCE_WINDOW  = 7      # days — evaluate last 7 days of predictions

# Logging helpers

def load_log() -> dict:
    """Load monitoring log from disk."""
    if LOG_PATH.exists():
        with open(LOG_PATH) as f:
            return json.load(f)
    return {"predictions": [], "performance": [], "alerts": []}


def save_log(log: dict):
    """Save monitoring log to disk."""
    with open(LOG_PATH, "w") as f:
        json.dump(log, f, indent=2, default=str)


def add_alert(log: dict, level: str, message: str):
    """Add an alert to the log."""
    alert = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "level":     level,   # "WARNING" or "CRITICAL"
        "message":   message,
    }
    log["alerts"].append(alert)
    print(f"\n  {'⚠' if level == 'WARNING' else '🚨'}  [{level}] {message}")


# Step 1 — Fetch latest actuals from SMARD and compare

def run_daily_monitoring(log: dict, df_recent: pd.DataFrame,
                         y_pred_yesterday: np.ndarray,
                         y_actual_today: np.ndarray,
                         price_std: float):
    """
    Compare yesterday's predictions with today's actual prices.
    Fires alerts if performance degrades.
    """
    print("=" * 60)
    print(f"  Daily Monitoring — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)

    mae  = np.mean(np.abs(y_actual_today - y_pred_yesterday))
    rmse = np.sqrt(np.mean((y_actual_today - y_pred_yesterday) ** 2))
    mae_std_ratio = mae / price_std if price_std > 0 else 0

    # Regime shift check — recent 30 days vs historical mean
    recent_mean   = df_recent["price_DE_LU"].tail(30 * 24).mean()
    historical_mean = df_recent["price_DE_LU"].mean()
    regime_diff   = abs(recent_mean - historical_mean)

    print(f"\n  Performance (last 24h predictions):")
    print(f"  MAE            : {mae:.2f} €/MWh  (threshold: {MAE_THRESHOLD}€)")
    print(f"  RMSE           : {rmse:.2f} €/MWh")
    print(f"  MAE / STD      : {mae_std_ratio:.2%}  (threshold: {MAE_STD_THRESHOLD:.0%})")
    print(f"\n  Regime check:")
    print(f"  Recent 30d mean   : {recent_mean:.1f} €/MWh")
    print(f"  Historical mean   : {historical_mean:.1f} €/MWh")
    print(f"  Difference        : {regime_diff:.1f} €/MWh  (threshold: {REGIME_THRESHOLD}€)")

    # Log performance
    log["performance"].append({
        "date":          datetime.now().date().isoformat(),
        "mae":           round(mae, 3),
        "rmse":          round(rmse, 3),
        "mae_std_ratio": round(mae_std_ratio, 4),
        "regime_diff":   round(regime_diff, 2),
    })

    # Fire alerts 
    alerts_fired = False

    if mae > MAE_THRESHOLD:
        add_alert(log, "WARNING",
                  f"MAE {mae:.1f}€ exceeds threshold {MAE_THRESHOLD}€ — "
                  f"model performance degrading")
        alerts_fired = True

    if mae_std_ratio > MAE_STD_THRESHOLD:
        add_alert(log, "WARNING",
                  f"MAE/STD {mae_std_ratio:.1%} exceeds {MAE_STD_THRESHOLD:.0%} — "
                  f"model explaining less variance than expected")
        alerts_fired = True

    if regime_diff > REGIME_THRESHOLD:
        add_alert(log, "CRITICAL",
                  f"Regime shift detected! Recent mean {recent_mean:.1f}€ vs "
                  f"historical {historical_mean:.1f}€ (diff: {regime_diff:.1f}€) — "
                  f"consider retraining immediately")
        alerts_fired = True

    if not alerts_fired:
        print("\n  ✓ All metrics within thresholds — model healthy")

    return mae, regime_diff


# Step 2 — Weekly retrain trigger

def should_retrain(log: dict, mae: float, regime_diff: float) -> bool:
    """
    Decide whether to retrain the model.
    Retrain if:
      - It's Monday (weekly schedule), OR
      - MAE is critically high, OR
      - Regime shift detected
    """
    today = datetime.now()
    is_monday = today.weekday() == 0

    reasons = []
    if is_monday:
        reasons.append("weekly schedule (Monday)")
    if mae > MAE_THRESHOLD * 1.5:
        reasons.append(f"MAE {mae:.1f}€ critically high")
    if regime_diff > REGIME_THRESHOLD:
        reasons.append(f"regime shift ({regime_diff:.1f}€ difference)")

    if reasons:
        print(f"\n  → Retrain triggered: {', '.join(reasons)}")
        return True

    print(f"\n  → No retrain needed today")
    return False


def run_retrain():
    """
    Retrain the model with the latest data.
    Imports pipeline and train functions and reruns them.
    """
    print("\n" + "=" * 60)
    print("  Starting retrain...")
    print("=" * 60)

    # In production: call smard_pipeline + smard_train programmatically
    # For now: print instructions
    print("""
  To retrain:
  1. Run smard_pipeline.py  → fetches latest 2 years of data
  2. Run smard_train.py     → retrains XGBoost, saves new model

  Or automate with:
      import subprocess
      subprocess.run(["python", "smard_pipeline.py"])
      subprocess.run(["python", "smard_train.py"])
    """)

    # Log retrain event
    log = load_log()
    log["performance"].append({
        "date":    datetime.now().date().isoformat(),
        "event":   "retrain",
        "message": "Model retrained with latest data",
    })
    save_log(log)


# Step 3 — Performance trend (last N days)

def print_performance_trend(log: dict):
    """Print MAE trend over last PERFORMANCE_WINDOW days."""
    perf = [p for p in log["performance"] if "mae" in p]
    if not perf:
        print("\n  No performance history yet.")
        return

    recent = perf[-PERFORMANCE_WINDOW:]
    print(f"\n  Performance trend (last {len(recent)} days):")
    print(f"  {'Date':<12} {'MAE':>8}  {'MAE/STD':>8}  Status")
    print(f"  {'-'*45}")

    for p in recent:
        mae     = p.get("mae", 0)
        ratio   = p.get("mae_std_ratio", 0)
        status  = "✓" if mae <= MAE_THRESHOLD else "⚠"
        print(f"  {p['date']:<12} {mae:>7.2f}€  {ratio:>7.1%}  {status}")

    maes = [p["mae"] for p in recent]
    trend = "↑ increasing" if maes[-1] > maes[0] else "↓ decreasing"
    print(f"\n  Trend: MAE is {trend} over last {len(recent)} days")


# Step 4 — Recent alerts summary

def print_recent_alerts(log: dict, n: int = 5):
    """Print the most recent alerts."""
    alerts = log.get("alerts", [])
    if not alerts:
        print("\n  No alerts in history.")
        return

    print(f"\n  Recent alerts (last {min(n, len(alerts))}):")
    for alert in alerts[-n:]:
        icon = "⚠" if alert["level"] == "WARNING" else "🚨"
        ts   = alert["timestamp"][:10]
        print(f"  {icon} [{ts}] {alert['message']}")


# Main

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrain", action="store_true",
                        help="Force model retrain")
    parser.add_argument("--status",  action="store_true",
                        help="Show monitoring status only")
    args = parser.parse_args()

    log = load_log()

    if args.retrain:
        run_retrain()

    elif args.status:
        print_performance_trend(log)
        print_recent_alerts(log)

    else:
        print("""
  smard_monitor.py — Usage:

  python smard_monitor.py --status    # show performance trend + alerts
  python smard_monitor.py --retrain   # force retrain now

  In production — add to cron:
  # Daily at 13:00 CET (after SMARD publishes actuals):
  0 13 * * * python /path/to/smard_monitor.py

  # Weekly retrain check (Monday 06:00):
  0 6 * * 1 python /path/to/smard_monitor.py --retrain
        """)

        # Show current status
        print_performance_trend(log)
        print_recent_alerts(log)

    save_log(log)
