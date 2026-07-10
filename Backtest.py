import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk

def backtest(prices, signals, beta, residuals, show_plots=False):
    s1, s2 = prices.columns[0], prices.columns[1]

    spreadReturn = prices[s1].pct_change() - beta * prices[s2].pct_change()
    position = signals.shift(1).fillna(0)
    stratReturn = position * spreadReturn

    cumulative = (1 + stratReturn).cumprod()
    sharpe = stratReturn.mean()/stratReturn.std() * np.sqrt(252) if stratReturn.std() != 0 else 0
    rollingMax = cumulative.cummax()
    drawdown = (cumulative - rollingMax) / rollingMax
    maxDd = drawdown.min()
    totReturn = cumulative.iloc[-1] - 1
    days = len(stratReturn)
    years = days / 252
    cagr = (1 + totReturn) ** (1/years) - 1
    downside = stratReturn[stratReturn < 0]
    sortino = stratReturn.mean() / downside.std() * np.sqrt(252) if downside.std() != 0 else 0
    winRate = (stratReturn > 0).mean()
    avgTrade = stratReturn.mean()

    if show_plots:
        cumulative_ret(cumulative)
        drawdown_plot(drawdown)
        trades_plot(residuals, signals)

        df = pd.DataFrame({
            "Calculation": [
                "Total Return", "CAGR", "Sharpe Ratio", "Sortino Ratio",
                "Max Drawdown", "Win Rate", "Average Trade Return"
            ],
            "Results": [totReturn, cagr, sharpe, sortino, maxDd, winRate, avgTrade],
        })

        root = tk.Tk()
        root.title("Performance Metrics")
        tree = ttk.Treeview(root, columns=list(df.columns), show="headings", height=len(df))
        for col in df.columns:
            tree.heading(col, text=col)
            tree.column(col, width=150, anchor="center")
        for _, row in df.iterrows():
            tree.insert("", "end", values=list(row))
        tree.pack(expand=True, fill="both")
        root.mainloop()

    # always return metrics
    results = {
        "Total Return": totReturn,
        "CAGR": cagr,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "Max Drawdown": maxDd,
        "Win Rate": winRate,
        "Average Trade Return": avgTrade,
    }
    return stratReturn, cumulative, results


#Plotting
    #Cumulative return
def cumulative_ret(cumulative):
    plt.figure(figsize=(12,6))
    plt.title("Pair Trading Backtest")
    plt.plot(cumulative, label="Cumulative Strategy Return")
    plt.legend()
    plt.show()

#Drawdown
def drawdown_plot(drawdown):
    plt.figure(figsize=(12,6))
    plt.title("Drawdowns")
    plt.plot(drawdown, label="Drawdown", color="red")
    plt.legend()
    plt.show()

def trades_plot(spread, signals):
    plt.figure(figsize=(12,6))
    plt.plot(spread, label="Spread")
    plt.plot(spread[signals == 1], 'r^', markersize=8, label="Short signal")
    plt.plot(spread[signals == -1], 'gv', markersize=8, label="Long signal")
    plt.title("Spread with trading signals")
    plt.legend()
    plt.show()
