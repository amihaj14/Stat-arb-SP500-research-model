import yfinance as yf
import pandas as pd
from itertools import combinations
import numpy as np
from statsmodels.tsa.stattools import coint
import Strategy

tickersDF = pd.read_csv("constituents.csv")
tickers = tickersDF["Symbol"].dropna().astype(str).tolist()
tickers = [t.strip() for t in tickers]

#Mapping know tickers with issues in yf
tickerMap = {
    "BRK.B": "BRK-B",
    "BF.B": "BF-B"
}

tickersYF = [tickerMap.get(t, t) for t in tickers]

df = yf.download(tickersYF, start='2015-01-01')['Close']
df = df.dropna(axis=1, how="all")

splitDate = "2020-01-01"
trainPrices = df.loc[:splitDate]
testPrices = df.loc[splitDate:]

#Combinations formula to choose 2 of 503 assets => 503!/2!(503-2)! = 503*502/2 (501! cancel out)
stockCombi_df = pd.DataFrame(combinations(trainPrices.columns,2))
stockCombi_df.columns = ['Stock1','Stock2']

##Try to calculate correlation using returns##
stockCombi_df['Correlation'] = stockCombi_df.apply(lambda row: np.corrcoef(trainPrices[row['Stock1']],trainPrices[row['Stock2']])[0,1],axis=1)
stockCombi_df = stockCombi_df[stockCombi_df.Correlation > 0.90]

print(stockCombi_df)

#The Engle-Granger two-step cointegration test
#if the score value is less than critVal then the pair is cointegrated
#pval < 0.05 => statistically significant cointegration, pval >0.10 => no meaningful coinegration. Lower pval the better

pVals = []
betas = []

for row in stockCombi_df.itertuples(index=False):
    s1, s2 = row.Stock1, row.Stock2

    trainingPair = trainPrices[[s1, s2]].dropna()

    if trainingPair.shape[0] < 100:
        pVals.append(np.nan)
        betas.append(np.nan)
        continue

    x = trainingPair[s1]
    y = trainingPair[s2]

    score, pVal, critVal = coint(x,y)
    pVals.append(pVal)

    _, beta, _ = Strategy.lin_reg(y, x)
    betas.append(beta)

stockCombi_df["Coint PVal"] = pVals
stockCombi_df["BetaValue"] = betas

stockCombi_df = stockCombi_df[stockCombi_df["Coint PVal"] < 0.05]
print(stockCombi_df)