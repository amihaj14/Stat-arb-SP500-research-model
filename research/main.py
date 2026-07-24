import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import research.DataLoader as DataLoader
import research.Backtest as Backtest
import research.Strategy as Strategy
import pandas as pd

ENTRY_THRESHOLD = 2.5
EXIT_THRESHOLD = 0.5
WINDOW_LENGTH = 60
TOP_N_PAIRS = 32

ROOT_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT_DIR / "live" / "config" / "pairs_config.csv"

pairs = list(DataLoader.stockCombi_df.itertuples(index=False))
print(f"Running backtest on {len(pairs)} candidate pairs using up to {min(8, os.cpu_count() or 1)} threads...")


def evaluate_pair(row):
    s1, s2 = row.Stock1, row.Stock2

    training_pair = DataLoader.trainPrices[[s1, s2]].dropna()
    if training_pair.shape[0] < 100:
        return None

    alpha, beta, _ = Strategy.lin_reg(training_pair[s1], training_pair[s2])

    test_pair = DataLoader.testPrices[[s1, s2]].dropna()
    if test_pair.shape[0] < 100:
        return None

    residuals = test_pair[s2] - (alpha + beta * test_pair[s1])
    zscore = Strategy.z_score(residuals, WINDOW_LENGTH)
    signals = Strategy.generate_signals(zscore, entry_threshold=ENTRY_THRESHOLD, exit_threshold=EXIT_THRESHOLD)

    _, _, metrics = Backtest.backtest(test_pair, signals, beta, residuals, show_plots=False)
    metrics["Pair"] = f"{s1}-{s2}"
    metrics["Stock1"] = s1
    metrics["Stock2"] = s2
    metrics["Beta"] = beta
    metrics["Alpha"] = alpha
    metrics["Entry Threshold"] = ENTRY_THRESHOLD
    metrics["Exit Threshold"] = EXIT_THRESHOLD
    metrics["Window Length"] = WINDOW_LENGTH
    return metrics


def export_pairs_config(results_df, top_n=TOP_N_PAIRS, output_path=CONFIG_PATH):
    config_df = results_df.head(top_n).copy()
    export_columns = [
        "Pair",
        "Stock1",
        "Stock2",
        "Alpha",
        "Beta",
        "Entry Threshold",
        "Exit Threshold",
        "Window Length",
        "Sharpe",
        "Sortino",
        "CAGR",
        "Total Return",
        "Max Drawdown",
        "Win Rate",
        "Average Trade Return",
        "Trade Count",
    ]
    export_columns = [col for col in export_columns if col in config_df.columns]
    config_df = config_df[export_columns]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    config_df.to_csv(output_path, index=False)
    print(f"Exported top {len(config_df)} pairs to {output_path}")


def plot_summary(results_df):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    top10 = results_df.head(10)
    axes[0, 0].barh(top10["Pair"][::-1], top10["Sharpe"][::-1], color="tab:blue")
    axes[0, 0].set_title("Top 10 Pairs by Sharpe Ratio")
    axes[0, 0].set_xlabel("Sharpe Ratio")

    axes[0, 1].hist(results_df["Sharpe"].dropna(), bins=30, color="tab:green", edgecolor="black")
    axes[0, 1].set_title("Sharpe Ratio Distribution")
    axes[0, 1].set_xlabel("Sharpe")
    axes[0, 1].set_ylabel("Count")

    axes[1, 0].scatter(results_df["Sharpe"], results_df["CAGR"], alpha=0.6)
    axes[1, 0].set_title("Sharpe vs CAGR")
    axes[1, 0].set_xlabel("Sharpe")
    axes[1, 0].set_ylabel("CAGR")

    axes[1, 1].scatter(results_df["Sharpe"], results_df["Max Drawdown"], alpha=0.6, color="tab:red")
    axes[1, 1].set_title("Sharpe vs Max Drawdown")
    axes[1, 1].set_xlabel("Sharpe")
    axes[1, 1].set_ylabel("Max Drawdown")

    fig.suptitle("Backtest Summary for Candidate Pairs", fontsize=16, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


def plot_top_pairs(best_pairs):
    # Plot each pair on its own figure (3 panels: cumulative, drawdown, residuals+markers)
    for row in best_pairs.itertuples(index=False):
        pair = row.Pair
        s1, s2 = pair.split("-")
        pair_test = DataLoader.testPrices[[s1, s2]].dropna()

        training_pair = DataLoader.trainPrices[[s1, s2]].dropna()
        alpha, beta, _ = Strategy.lin_reg(training_pair[s1], training_pair[s2])
        residuals = pair_test[s2] - (alpha + beta * pair_test[s1])
        zscore = Strategy.z_score(residuals, window=60)
        signals = Strategy.generate_signals(zscore, entry_threshold=2.5, exit_threshold=0.5)
        _, cumulative, _ = Backtest.backtest(
            pair_test,
            signals,
            beta,
            residuals,
            show_plots=False,
        )

        drawdown = cumulative / cumulative.cummax() - 1

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        ax_cum, ax_dd, ax_spread = axes[0], axes[1], axes[2]

        ax_cum.plot(cumulative, color="tab:blue")
        ax_cum.set_title(f"{pair} Cumulative Return")
        ax_cum.set_ylabel("Cumulative Value")

        ax_dd.plot(drawdown, color="tab:red")
        ax_dd.set_title(f"{pair} Drawdown")
        ax_dd.set_ylabel("Drawdown")

        # Plot residuals and mark trade entries/exits
        ax_spread.plot(residuals, label="Residuals", color="tab:purple")
        Backtest.plot_trade_markers(residuals, signals, ax=ax_spread)
        ax_spread.set_title(f"{pair} Residuals + Signals")

        plt.tight_layout()
        plt.show()

    plt.tight_layout()
    plt.show()


worker_count = min(8, os.cpu_count() or 1)
with ThreadPoolExecutor(max_workers=worker_count) as executor:
    results = list(executor.map(evaluate_pair, pairs))

results = [r for r in results if r is not None]
results_df = pd.DataFrame(results)
if results_df.empty:
    print("No valid pairs found")
    raise SystemExit

results_df.sort_values("Sharpe", ascending=False, inplace=True)

print("\nTop 10 pairs by Sharpe:")
print(results_df.head(10).reset_index(drop=True).to_string(index=False))

export_pairs_config(results_df, top_n=TOP_N_PAIRS)

best_pairs = results_df.head(3).copy()
print("\nBest 3 pairs by Sharpe:")
print(best_pairs[["Pair", "Sharpe", "CAGR", "Total Return", "Max Drawdown"]].to_string(index=False))

plot_summary(results_df)
plot_top_pairs(best_pairs)
