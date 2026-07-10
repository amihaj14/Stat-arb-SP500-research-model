import matplotlib.pyplot as plt
import DataLoader
import Backtest
import Strategy
import pandas as pd

results=[]
# VECTORIZE EVERYTHING TO MAXIMIZE EFFICIENY AND SPEED

for row in DataLoader.stockCombi_df.itertuples(index=False):
    s1, s2 = row.Stock1, row.Stock2
    beta = row.BetaValue

    testPair = DataLoader.testPrices[[s1,s2]].dropna()
    if testPair.shape[0] < 100:
        continue

    x = testPair[s1]
    y = testPair[s2]    
    
    _, _, residuals = Strategy.lin_reg(y, x)

    zscore = Strategy.z_score(residuals, 60)
    signals = Strategy.generate_signals(zscore)

    stratReturn, cumulative, metrics = Backtest.backtest(testPair, signals, beta, residuals)
    metrics["Pair"] = f"{s1}-{s2}"
    results.append(metrics)

results_df = pd.DataFrame(results)
if results_df.empty:
    print("No valid pairs found")
    exit()

print(results_df.sort_values("Sharpe", ascending=False).head(10))

top_pairs = results_df.sort_values("Sharpe", ascending=False).head(3)["Pair"]
for pair in top_pairs:
    s1, s2 = pair.split("-")
    pair_test = DataLoader.testPrices[[s1, s2]].dropna()
    y_test = pair_test[s1]
    x_test = pair_test[s2]
    beta = DataLoader.stockCombi_df.query("Stock1 == @s1 and Stock2 == @s2")["BetaValue"].iloc[0]
    _, _, residuals = Strategy.lin_reg(y_test, x_test)
    zscore = Strategy.z_score(residuals, window=60)
    signals = Strategy.generate_signals(zscore)

    Backtest.backtest(pair_test, signals, beta, residuals, show_plots=True)