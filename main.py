import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# --------------------
# Settings
# --------------------
tickers = ["AAPL", "AMZN", "GOOGL", "MSFT"]
start_date = "2020-01-01"
trading_days = 252
risk_free_rate = 0.02   # 2% annual, change if you want
n_portfolios = 20000

# --------------------
# Download prices (Close)
# --------------------
raw = yf.download(tickers, start=start_date, progress=True)

if isinstance(raw.columns, pd.MultiIndex):
    prices = raw["Close"]
else:
    prices = raw["Close"]

prices = prices.dropna()
returns = prices.pct_change().dropna()

# Annualized stats
mu = returns.mean() * trading_days
cov = returns.cov() * trading_days

# --------------------
# Random portfolio simulation
# --------------------
rng = np.random.default_rng(42)

weights = rng.random((n_portfolios, len(tickers)))
weights = weights / weights.sum(axis=1, keepdims=True)

port_returns = weights @ mu.values
port_vol = np.sqrt(np.einsum("ij,jk,ik->i", weights, cov.values, weights))
sharpe = (port_returns - risk_free_rate) / port_vol

# Best portfolios
i_max_sharpe = np.argmax(sharpe)
i_min_var = np.argmin(port_vol)

w_max_sharpe = weights[i_max_sharpe]
w_min_var = weights[i_min_var]

# --------------------
# Print results
# --------------------
def fmt_weights(w):
    return {t: float(f"{x:.4f}") for t, x in zip(tickers, w)}

print("\n=== Annualized Expected Returns (mu) ===")
print(mu)

print("\n=== Max Sharpe Portfolio ===")
print("Return:", float(port_returns[i_max_sharpe]))
print("Vol:", float(port_vol[i_max_sharpe]))
print("Sharpe:", float(sharpe[i_max_sharpe]))
print("Weights:", fmt_weights(w_max_sharpe))

print("\n=== Min Variance Portfolio ===")
print("Return:", float(port_returns[i_min_var]))
print("Vol:", float(port_vol[i_min_var]))
print("Sharpe:", float(sharpe[i_min_var]))
print("Weights:", fmt_weights(w_min_var))

# --------------------
# Plot efficient frontier cloud
# --------------------
plt.figure()
plt.scatter(port_vol, port_returns, s=5)
plt.scatter(port_vol[i_max_sharpe], port_returns[i_max_sharpe], s=80, marker="*", label="Max Sharpe")
plt.scatter(port_vol[i_min_var], port_returns[i_min_var], s=80, marker="X", label="Min Variance")
plt.xlabel("Annualized Volatility")
plt.ylabel("Annualized Return")
plt.title("Random Portfolio Frontier (Close Prices)")
plt.legend()
plt.show()

