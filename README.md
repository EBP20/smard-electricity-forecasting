# SMARD Germany — Day-Ahead Electricity Price Forecasting

 [View notebook on nbviewer](https://nbviewer.org/github/EBP20/smard-electricity-forecasting/blob/main/power_trading.ipynb)

Predicts all 24 hourly Day-Ahead prices for the German market (EPEX Spot DE/LU) using XGBoost. Run it before the 12:00 CET auction — by then you have real generation data up to ~hour 10–11 plus TSO forecasts for tomorrow, which together give a much richer signal than forecasts alone.

---

## Results

| Method | MAE | RMSE |
|---|---|---|
| XGBoost (test set) | 25.5 €/MWh | 36.7 €/MWh |
| Walk-forward CV (mean, 5 folds) | 24.8 ± 2.2 €/MWh | 36.8 €/MWh |
| Rolling backtest (daily retrain) | 22.6 €/MWh | 33.4 €/MWh |
| Naive benchmark (lag-24h) | 33.5 €/MWh | — |

**~33% better than naive.** Directional accuracy: 71.2%.

Trained on 3 years of hourly data (June 2023 → May 2026), 67 features, 25,872 rows. Target price range: -500 to +936 €/MWh — the model handles negative prices and extreme spikes without special treatment.

Walk-forward CV is stable across all 5 folds (worst: 28.8, best: 22.1 €/MWh), which means it's not just fitting one particular market regime.

All results using free public data only — SMARD, Open-Meteo, TSO forecasts. Commercial data (gas prices, CO2) would likely improve this further.

---

## Top features (SHAP)

1. `renewable_share_now` — current renewable share is the single strongest signal
2. `price_lag_48h` — two days ago same hour
3. `tomorrow_dow` — day of week matters a lot for demand profile
4. `price_NL_lag24`, `price_AT_lag24`, `price_CZ_lag24` — European market coupling
5. `wind_solar_x_weekend` — high renewables on a weekend → price can go deeply negative
6. `forecast_wind_solar_delta_6h` — direction of forecast revision matters more than absolute level

---

## How it works

The model runs every morning around 11:30 CET. At that point:
- **Actual data** (generation, load, prices) is available up to ~hour 10–11 today
- **TSO forecasts** for tomorrow's wind, solar, and total generation are already published
- **Target**: all 24 hourly DA prices for the next day (shift -24h on price_DE_LU)

Feature engineering is built around what's actually available at prediction time — no data leakage. Neighbour prices use lag-24h because SMARD publishes them with a delay. Rolling stats use shift(1) to exclude the current hour.

---

## Features

**Price history** — lags at 24h, 48h, 168h, rolling means and volatility, spread vs European average

**Today's actual** — load mean and trend, wind total, solar peak, renewable share right now

**Tomorrow's TSO forecasts** — wind offshore/onshore, solar, total generation, residual conventional proxy

**Forecast revisions** — 6h and 12h deltas on wind+solar forecasts (direction of update is a signal in itself)

**Weather** — temperature, wind speed, solar radiation, historical + forecast, bias-corrected

**Calendar** — day of week, month, German holidays, bridge days, cyclical encoding

**Interactions** — wind × weekend, cold × night

---

## Files

```
smard-electricity-forecasting/
├── power_trading.ipynb   — full pipeline: data → features → model → evaluation → live forecast
├── smard_cache.py        — SQLite cache layer, only downloads what's new from SMARD
├── smard_monitor.py      — daily monitoring: MAE tracking, regime shift detection, retrain triggers
└── README.md
```

**smard_cache.py** — checks the latest available SMARD bucket timestamp before downloading anything. On re-runs, only fetches new data. Saves to a local SQLite DB so the full 3-year history doesn't need to be re-downloaded every day.

**smard_monitor.py** — run daily after 13:00 CET when SMARD publishes actual prices. Compares yesterday's predictions against actuals, tracks MAE over a 7-day rolling window, fires alerts if performance degrades, and triggers retraining if a regime shift is detected (recent 30-day mean vs historical mean diverges by more than 30 €/MWh).

---

## Data sources

- **SMARD** (Bundesnetzagentur) — generation mix, consumption, DA prices DE/LU and 9 neighbours, TSO forecasts
- **Open-Meteo** — historical weather archive + 2-day forecast (temperature, wind, solar radiation)

All free, no API key needed.

---

## Setup

```bash
pip install xgboost scikit-learn shap optuna lightgbm pandas numpy matplotlib requests holidays joblib
```

Run `smard_cache.py` utilities are imported by the notebook — no separate run needed. Just open the notebook and run top to bottom.

**Timing**: run before 12:00 CET. After the EPEX auction closes (~12:30), tomorrow's prices are published on SMARD and live prediction rows are no longer available.

---

## What's next

- Quantile regression for proper probabilistic price bands instead of ±MAE
- Regime detection (low price / normal / high volatility) with separate models per regime
- Spike modelling for extreme prices
- Add gas and CO2 price data as features
- Intraday + Day-Ahead combination

---

## License

MIT
