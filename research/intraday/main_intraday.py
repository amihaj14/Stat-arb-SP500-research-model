import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import pandas as pd
import research.intraday.DataLoader_intraday as DataLoader
import research.intraday.Backtest_intraday as Backtest
import research.intraday.Strategy_intraday as StrategyCfg
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import Strategy

pairs = list(DataLoader.stockCombi_df.itertuples(index=False))
print(f"Running intraday backtest on {len(pairs)} candidate pairs...")

TOP_N_PAIRS = 10
CONFIG_PATH = r"C:\Users\crnar\Quant_Projects\pairs-trading-SP-500-model\live\config\pairs_config_intraday.csv"

def evaluate_pair(row):
    s1, s2 = row.Stock1, row.Stock2

    training_pair = DataLoader.trainPrices[[s1, s2]].dropna()
    if training_pair.shape[0] < 200:
        return None

    alpha, beta, _ = Strategy.lin_reg(training_pair[s1], training_pair[s2])

    test_pair = DataLoader.testPrices[[s1, s2]].dropna()
    if test_pair.shape[0] < 200:
        return None

    residuals = test_pair[s2] - (alpha + beta * test_pair[s1])
    zscore = Strategy.z_score(residuals, StrategyCfg.WINDOW_LENGTH)
    signals = Strategy.generate_signals(
        zscore,
        entry_threshold=StrategyCfg.ENTRY_THRESHOLD,
        exit_threshold=StrategyCfg.EXIT_THRESHOLD,
    )

    _, _, metrics = Backtest.backtest_intraday(test_pair, signals, beta, residuals)
    metrics["Pair"] = f"{s1}-{s2}"
    metrics["Stock1"] = s1
    metrics["Stock2"] = s2
    metrics["Alpha"] = alpha
    metrics["Beta"] = beta
    metrics["Entry Threshold"] = StrategyCfg.ENTRY_THRESHOLD
    metrics["Exit Threshold"] = StrategyCfg.EXIT_THRESHOLD
    metrics["Window Length"] = StrategyCfg.WINDOW_LENGTH
    return metrics

def export_pairs_config(results_df, top_n=TOP_N_PAIRS, output_path=CONFIG_PATH):
    output_path = Path(output_path)
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


worker_count = min(8, os.cpu_count() or 1)
with ThreadPoolExecutor(max_workers=worker_count) as executor:
    results = list(executor.map(evaluate_pair, pairs))

results = [r for r in results if r is not None]
results_df = pd.DataFrame(results)
if results_df.empty:
    print("No valid pairs found")
    raise SystemExit

results_df.sort_values("Sharpe", ascending=False, inplace=True)

print("\nTop 10 intraday pairs by Sharpe (after cost drag):")
print(results_df.head(10).reset_index(drop=True).to_string(index=False))

results_df.to_csv(r"C:\Users\crnar\Quant_Projects\pairs-trading-SP-500-model\research\intraday\intraday_backtest_results.csv", index=False)
print("\nSaved full results to intraday_backtest_results.csv")

export_pairs_config(results_df, top_n=TOP_N_PAIRS)