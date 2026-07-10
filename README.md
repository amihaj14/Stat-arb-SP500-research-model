# Statistical Arbitrage - S&P 500 Research Model

## Overview

This repository implements a pairs trading research model over the S&P 500 universe.
It downloads historical adjusted prices, identifies highly correlated and cointegrated stock pairs,
then backtests a mean-reversion strategy using residual z-score entry and exit signals.

## Key Files

- `DataLoader.py` - loads price data from Yahoo Finance, caches it in `price_cache.csv`,
  splits into train/test, and selects candidate pairs using correlation and cointegration.
- `Strategy.py` - fits a linear hedge ratio, computes residual z-scores, and generates long/short signals.
- `Backtest.py` - computes strategy returns, cumulative performance, Sharpe, Sortino, drawdown, and plots.
- `main.py` - orchestrates pair evaluation, sorts results by Sharpe, and plots top-pair performance.
- `constituents.csv` - S&P 500 ticker universe used in the research.

## Dependencies

The project uses the following Python packages:

- `pandas`
- `numpy`
- `statsmodels`
- `yfinance`
- `matplotlib`

## Running the Model

Run the main script to execute the research pipeline and generate plots for the top candidate pairs:

```powershell
python main.py
```

The first run may take longer if it needs to download data for all tickers.

## Notes on Data Caching

- `DataLoader.py` writes price history to `price_cache.csv` after the first successful download.
- If the cached data does not cover the requested `START_DATE` or if tickers are missing,
  the script will automatically refresh the cache.
- This keeps repeated runs faster once the cache is valid.

## Customizing the Date Ranges

There are two key date settings in `DataLoader.py`:

- `START_DATE` - the beginning of the price history used for training and testing.
- `split_date` - the boundary between training data and test data.

For example, to test on 2023–present, set:

```python
START_DATE = "2018-01-01"
split_date = "2023-01-01"
```

If you only want recent history, adjust `START_DATE` accordingly.

## Strategy Notes

- The current signals are generated using z-score thresholds.
- Stronger entry/exit thresholds were added to reduce noise and improve Sharpe ratios.
- The model evaluates top pairs separately and plots their cumulative return, drawdown, and entry/exit points.

## Output

- The console prints the top pairs by Sharpe ratio.
- The script generates separate plot pages for each top pair.
- `price_cache.csv` stores downloaded price data to speed future runs.

## Troubleshooting

- If Yahoo Finance fails for some tickers, the download may skip delisted symbols.
- If the data cache becomes stale after changing `START_DATE`, delete `price_cache.csv` and rerun.

```powershell
Remove-Item .\price_cache.csv
python main.py
```

## Recommended Next Steps

- Use higher entry/exit z-score thresholds to reduce weak trades.
- Filter candidate pairs by sector or minimum spread volatility.
- Add trade-level profit and loss tracking if precise win-rate metrics are desired.
