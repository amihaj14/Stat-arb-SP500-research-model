import os
from concurrent.futures import ThreadPoolExecutor

import yfinance as yf
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import coint
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import Strategy

CACHE_PATH = r"C:\Users\crnar\Quant_Projects\pairs-trading-SP-500-model\research\intraday\intraday_price_cache.csv"
MAX_CANDIDATE_PAIRS = 1000
CORRELATION_THRESHOLD = 0.92
INTERVAL = "15m"
LOOKBACK_DAYS = 59  # yfinance hard caps 15m data at 60 days; stay under it

tickers_df = pd.read_csv(r"C:\Users\crnar\Quant_Projects\pairs-trading-SP-500-model\research\constituents.csv")
tickers = tickers_df["Symbol"].dropna().astype(str).str.strip().tolist()

ticker_map = {"BRK.B": "BRK-B", "BF.B": "BF-B"}
tickers_yf = [ticker_map.get(t, t) for t in tickers]


def download_and_cache():
    print(f"Downloading {INTERVAL} data for {len(tickers_yf)} symbols "
          f"(last {LOOKBACK_DAYS} days)...")
    df = yf.download(
        tickers_yf,
        period=f"{LOOKBACK_DAYS}d",
        interval=INTERVAL,
        auto_adjust=True,
        progress=False,
        threads=True,
        group_by="column",
    )

    if isinstance(df.columns, pd.MultiIndex):
        prices_local = df["Close"]
    else:
        prices_local = df

    prices_local = prices_local.dropna(axis=1, how="all").astype(float)
    prices_local.to_csv(CACHE_PATH)
    return prices_local


def _cache_is_stale(prices: pd.DataFrame) -> bool:
    if prices.empty:
        return True
    last_bar_age = pd.Timestamp.now(tz=prices.index.tz) - prices.index.max()
    return last_bar_age > pd.Timedelta(hours=6)


if os.path.exists(CACHE_PATH):
    print(f"Loading cached intraday data from {CACHE_PATH}...")
    prices = pd.read_csv(CACHE_PATH, index_col=0, parse_dates=True)
    if _cache_is_stale(prices):
        print("Cache is stale, re-downloading...")
        prices = download_and_cache()
else:
    prices = download_and_cache()

n = len(prices)
split_idx = int(n * 0.7)
train_prices = prices.iloc[:split_idx]
test_prices = prices.iloc[split_idx:]

print(f"Train: {train_prices.shape[0]} bars | Test: {test_prices.shape[0]} bars")

correlation_matrix = train_prices.corr().abs()
triangular_mask = np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
correlated_pairs = correlation_matrix.where(triangular_mask).stack()
correlated_pairs = correlated_pairs[correlated_pairs > CORRELATION_THRESHOLD].sort_values(ascending=False)
correlated_pairs = correlated_pairs.head(MAX_CANDIDATE_PAIRS)

stockCombi_df = pd.DataFrame(
    [(s1, s2, float(corr)) for (s1, s2), corr in correlated_pairs.items()],
    columns=["Stock1", "Stock2", "Correlation"],
)
print(f"Evaluating {len(stockCombi_df)} correlated pairs...")


def _evaluate_pair(args):
    s1, s2, training_pair = args
    if training_pair.shape[0] < 200:
        return s1, s2, np.nan, np.nan
    x = training_pair[s1]
    y = training_pair[s2]
    _, p_val, _ = coint(x, y)
    _, beta, _ = Strategy.lin_reg(y, x)
    return s1, s2, p_val, beta


pair_inputs = []
for row in stockCombi_df.itertuples(index=False):
    s1, s2, _ = row
    training_pair = train_prices[[s1, s2]].dropna()
    pair_inputs.append((s1, s2, training_pair))

worker_count = min(8, max(1, os.cpu_count() or 1))
with ThreadPoolExecutor(max_workers=worker_count) as executor:
    results = list(executor.map(_evaluate_pair, pair_inputs))

stockCombi_df[["Coint PVal", "BetaValue"]] = pd.DataFrame(
    results, columns=["Stock1", "Stock2", "Coint PVal", "BetaValue"]
)[["Coint PVal", "BetaValue"]]
stockCombi_df = stockCombi_df[stockCombi_df["Coint PVal"] < 0.05]

print(stockCombi_df.head(20))

trainPrices = train_prices
testPrices = test_prices