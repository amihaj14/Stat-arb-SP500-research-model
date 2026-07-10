import statsmodels.api as sm
import pandas as pd
from statsmodels.tsa.stattools import coint
#import DataLoader

"""Uses linear regression to obtain residuals, alpha, and beta values"""
def lin_reg(Y, X):
        X_const = sm.add_constant(X)
        model = sm.OLS(Y, X_const).fit()
        alpha, beta = model.params
        residuals = Y - (alpha + beta * X)
        return alpha, beta, residuals

"""Uses spread residuals found from linear regression to compute z-score"""
def z_score(residuals, window=None):
    if window:
        mean = residuals.rolling(window=window).mean()
        std = residuals.rolling(window=window).std()
    else:
        mean = residuals.mean()
        std = residuals.std()
    zscore = (residuals - mean)/std
    return zscore 
    
"""Calculates short and long signals using z-score"""
def generate_signals(zscore):
    signals = pd.Series(index=zscore.index, dtype='int')
    signals[zscore > 0.5] = 1   #Short A, Long B
    signals[zscore < -0.5] = -1 #Short B, Long A
    signals[(zscore >= -0.5) & (zscore <= 0.5)] = 0   #Neutral, close the position
    return signals    
