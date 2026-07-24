import numpy as np
import pandas as pd

BARS_PER_YEAR = 252 * 26  # ~26 fifteen-minute bars per 6.5hr trading day

SLIPPAGE_BPS = 2.0    # per-leg slippage assumption
SPREAD_COST_BPS = 3.0  # bid-ask cost assumption, per leg, per round trip


def backtest_intraday(prices, signals, beta, residuals):
    s1, s2 = prices.columns[0], prices.columns[1]

    spreadReturn = prices[s2].pct_change() - beta * prices[s1].pct_change()
    position = signals.shift(1).fillna(0)
    stratReturn = position * spreadReturn

    trade_changes = (position != position.shift(1)).cumsum()
    entries = (position != 0) & (position.shift(1) == 0)
    exits = (position == 0) & (position.shift(1) != 0)
    cost_bps = (SLIPPAGE_BPS + SPREAD_COST_BPS) / 10000
    stratReturn = stratReturn.copy()
    stratReturn[entries | exits] -= cost_bps

    cumulative = (1 + stratReturn).cumprod()
    sharpe = stratReturn.mean() / stratReturn.std() * np.sqrt(BARS_PER_YEAR) if stratReturn.std() != 0 else 0
    rollingMax = cumulative.cummax()
    drawdown = (cumulative - rollingMax) / rollingMax
    maxDd = drawdown.min()
    totReturn = cumulative.iloc[-1] - 1

    bars = len(stratReturn)
    years = bars / BARS_PER_YEAR
    cagr = (1 + totReturn) ** (1 / years) - 1 if years > 0 else 0

    downside = stratReturn[stratReturn < 0]
    sortino = stratReturn.mean() / downside.std() * np.sqrt(BARS_PER_YEAR) if downside.std() != 0 else 0

    active_mask = position != 0
    trade_returns = stratReturn[active_mask].groupby(trade_changes[active_mask]).sum()
    winRate = (trade_returns > 0).mean() if len(trade_returns) else 0
    avgTrade = trade_returns.mean() if len(trade_returns) else 0
    trade_count = len(trade_returns)

    results = {
        "Total Return": totReturn,
        "CAGR": cagr,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "Max Drawdown": maxDd,
        "Win Rate": winRate,
        "Average Trade Return": avgTrade,
        "Trade Count": trade_count,
    }
    return stratReturn, cumulative, results