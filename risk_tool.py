import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.stats import norm, chi2
from matplotlib.colors import LinearSegmentedColormap

WEIGHT_COLOR = "#2a78d6"   # blue
RISK_COLOR = "#eb6834"     # orange
TEXT_MUTED = "#52514e"     # dark gray for the zero line


# =====================================================================
# 0. INPUT CHECKS
# =====================================================================

def check_inputs(tickers, weights):
    """Stop early with a clear message if the portfolio is invalid."""
    if len(tickers) != len(weights):
        raise ValueError("Need one weight per ticker.")
    if abs(sum(weights) - 1) > 1e-6:
        raise ValueError(f"Weights sum to {sum(weights):.3f}, not 1.")
    if any(w < 0 for w in weights):
        raise ValueError("Weights must be non-negative (long-only).")


# =====================================================================
# 1. DATA
# =====================================================================

def load_prices(tickers, start="2012-01-01"):
    """Download daily closing prices from Yahoo Finance.

    Returns: DataFrame, rows = dates, columns = tickers.
    """
    prices = yf.download(tickers, start=start, auto_adjust=True)["Close"]
    return prices


def check_history(prices, window=500, min_test_days=750):
    """
    Print each ticker's first available date and warn if the usable
    history (dates where ALL tickers have prices) is too short for the
    rolling window plus a meaningful test period.
    """
    first_dates = {}
    for t in prices.columns:
        first = prices[t].first_valid_index()
        first_dates[t] = first

    usable = len(prices.dropna())
    needed = window + min_test_days

    if usable < needed:
        latest = max(first_dates, key=first_dates.get) # compares the key dates
        print(f"Not OK: only {usable:,} usable days, need {needed:,}). "
              f"{latest} has the shortest history, starting from {first_dates[latest].date()}.")
    else:
        print(f"OK: contians {usable:,} days (need {needed:,}).")

    return usable

def compute_returns(prices):
    """Daily returns for each asset, with missing rows dropped."""
    return prices.pct_change().dropna()
    


def portfolio_returns(returns, w):
    """
    One daily return series for the whole portfolio.

    returns: DataFrame of daily asset returns (rows = dates, columns = tickers).
    w:       Series of weights, indexed by ticker, summing to 1.

    Returns: Series of daily portfolio returns.
    """
    return returns[w.index] @ w


# =====================================================================
# 2. RISK CONCENTRATION ANALYSIS
# =====================================================================

def risk_shares(returns, w):
    """
    Each asset's share of total portfolio variance.

    returns: DataFrame of daily asset returns (rows = dates, columns = tickers).
    w:       Series of weights, indexed by ticker, summing to 1.

    Returns: DataFrame with columns ["weight", "risk_share"]; risk_share sums to 1.
    """
    cov = returns[w.index].cov()          # Σ, in the same ticker order as w
    port_var = w @ cov @ w                # w'Σw, a single number
    contrib = w * (cov @ w)               # w_i × (Σw)_i, one per asset
    return pd.DataFrame({"weight": w, "risk_share": contrib / port_var})


def plot_risk_shares(shares):
    """Side-by-side bar chart: weight vs. risk share for each asset."""
    labels = {"weight": "Weight (share of capital)",
              "risk_share": "Risk share (share of variance)"}
    ax = shares[["weight", "risk_share"]].rename(columns=labels).plot(
        kind="bar",
        color=[WEIGHT_COLOR, RISK_COLOR],
        width=0.75,
        edgecolor="white",
        linewidth=2,
        figsize=(8, 4.5),
    )
    ax.axhline(0, color=TEXT_MUTED, linewidth=1)

    # Label each risk-share bar with its value, so small and negative shares stay visible
    risk_bars = ax.containers[1]
    ax.bar_label(risk_bars,
                 labels=[f"{v:.1%}" for v in shares["risk_share"]],
                 padding=3, fontsize=9, color=TEXT_MUTED)

    # Leave room below zero if any share is negative
    low = min(0, shares["risk_share"].min())
    ax.set_ylim(low - 0.05, shares[["weight", "risk_share"]].max().max() + 0.08)

    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.set_title("Share of capital vs. share of risk", loc="left", fontsize=13)
    ax.set_xlabel("")
    ax.set_ylabel("Share of portfolio")
    ax.tick_params(axis="x", rotation=0)
    ax.legend(frameon=False)
    ax.grid(axis="y", color="#e5e4e0", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    plt.tight_layout()
    plt.show()


def plot_correlations(returns):
    """Heatmap of the correlation matrix (which assets move together)."""
    corr = returns.corr()
    cmap = LinearSegmentedColormap.from_list(
        "blue_gray_red", ["#2a78d6", "#f0efec", "#e34948"]
    )

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(corr, cmap=cmap, vmin=-1, vmax=1)

    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns)
    ax.set_yticklabels(corr.columns)

    for i in range(len(corr)):
        for j in range(len(corr)):
            value = corr.iloc[i, j]
            ax.text(j, i, f"{value:.2f}", ha="center", va="center",
                    color="white" if abs(value) > 0.6 else "#0b0b0b", fontsize=9)

    fig.colorbar(im, ax=ax, label="Correlation", shrink=0.8)
    ax.set_title("Correlation of daily returns", loc="left", fontsize=13)
    plt.tight_layout()
    plt.show()


def pca_concentration(returns):
    """
    PCA via eigen-decomposition of the correlation matrix.

    returns: DataFrame of daily asset returns (rows = dates, columns = tickers).

    Returns:
      explained: Series    -  share of total variation per component (largest first)
      loadings:  DataFrame - loadings of every component (rows = tickers, columns = PC1..PCn)
    """
    corr = returns.corr()
    eigvals, eigvecs = np.linalg.eigh(corr)

    # eigh sorts smallest first, flip so PC1 comes first
    eigvals = eigvals[::-1]
    eigvecs = eigvecs[:, ::-1]

    names = [f"PC{k+1}" for k in range(len(eigvals))]
    explained = pd.Series(eigvals / eigvals.sum(), index=names)

    # An eigenvector's sign is arbitrary... flip each one so its largest loading is positive
    for k in range(eigvecs.shape[1]):
        if eigvecs[np.argmax(np.abs(eigvecs[:, k])), k] < 0:
            eigvecs[:, k] = -eigvecs[:, k]

    loadings = pd.DataFrame(eigvecs, index=corr.columns, columns=names)

    return explained, loadings


# =====================================================================
# 3. LOSS
# =====================================================================

def var_historical(port, level=0.95):
    """Historical VaR: the (1 - level) quantile of past returns."""
    # TODO
    raise NotImplementedError


def var_parametric(port, level=0.95):
    """Normal VaR: mean + z * sd, where z = norm.ppf(1 - level)."""
    # TODO
    raise NotImplementedError


def expected_shortfall(port, level=0.95):
    """Average return on the days worse than historical VaR."""
    # TODO: var = var_historical(port, level); return port[port <= var].mean()
    raise NotImplementedError


def worst_day(port):
    """Largest one-day loss in the data and the date it happened."""
    # TODO: return port.min(), port.idxmin()
    raise NotImplementedError


def loss_table(port, portfolio_value=1_000_000):
    """
    Summary table of loss measures at 95% and 99%, as returns and dollars.
    Rows: Historical VaR, Parametric VaR, Expected Shortfall.
    """
    # TODO: build a DataFrame from the functions above
    # TODO: add dollar columns = return * portfolio_value
    raise NotImplementedError


def qq_plot(port):
    """QQ plot of portfolio returns vs. a normal distribution."""
    # TODO: import statsmodels.api as sm; sm.qqplot(port, line="s"); plt.show()
    raise NotImplementedError


# =====================================================================
# 4. BACKTEST... can we trust the VaR?
# =====================================================================

def rolling_var(port, method="historical", level=0.95, window=500):
    """
    VaR for each day using ONLY the previous `window` days.

    method: "historical" or "parametric"
    The .shift(1) at the end is essential - without it, each day's
    VaR would use that same day's return (look-ahead bias).
    """
    alpha = 1 - level
    # TODO historical: port.rolling(window).quantile(alpha)
    # TODO parametric: rolling mean + norm.ppf(alpha) * rolling std
    # TODO: return result.shift(1)
    raise NotImplementedError


def exceedances(port, var_series):
    """
    True on days the return was worse than that day's VaR.
    Only count days where a VaR exists (drop the NaN warm-up period).
    """
    # TODO: valid = var_series.notna()
    # TODO: return (port < var_series)[valid]
    raise NotImplementedError


def kupiec_test(exceed, level=0.95):
    """
    Likelihood ratio test: is the exceedance rate equal to 1 - level?

    H0: p = 1 - level.   Alternative: p = observed rate x / T.
    LR = -2 * (loglik_H0 - loglik_alt) ~ chi-squared(1).

    Returns: dict with x, T, expected, observed_rate, LR, p_value.
    Watch out: if x == 0, log(p_hat) breaks - handle that case.
    """
    # TODO: x = exceed.sum(); T = len(exceed); p = 1 - level; p_hat = x / T
    # TODO: log-likelihoods, LR, p_value = 1 - chi2.cdf(LR, df=1)
    raise NotImplementedError


def plot_backtest(port, var_series, exceed, title="VaR backtest"):
    """Returns over time, VaR line on top, exceedances marked in red."""
    # TODO: plot port (thin grey), var_series (line), exceedance days (red dots)
    raise NotImplementedError


# =====================================================================
# 5. Reporting
# =====================================================================

def risk_report(tickers, weights, start="2012-01-01",
                portfolio_value=1_000_000, window=500):
    """
    Full risk report for any long-only portfolio.
    Returns a dict of results so the notebook can reuse them.
    """
    check_inputs(tickers, weights)
    w = pd.Series(weights, index=tickers)

    # 1. Data
    prices = load_prices(tickers, start=start)
    check_history(prices, window=window)
    returns = compute_returns(prices)
    port = portfolio_returns(returns, w)

    # 2. Concentration
    shares = risk_shares(returns, w)
    explained, loadings = pca_concentration(returns[w.index])
    print("\n== Where the risk comes from ==")
    print(shares)
    print(f"First principal component explains {explained.iloc[0]:.1%} of total variation")
    plot_risk_shares(shares)
    plot_correlations(returns[w.index])

    # 3. Loss
    table = loss_table(port, portfolio_value)
    worst, worst_date = worst_day(port)
    print("\n== How much could we lose ==")
    print(table)
    print(f"Worst day: {worst:.2%} on {worst_date.date()}")
    qq_plot(port)

    # 4. Backtest (95%, both methods)
    backtests = {}
    print("\n== Can we trust the VaR ==")
    for method in ["historical", "parametric"]:
        var_series = rolling_var(port, method=method, level=0.95, window=window)
        exceed = exceedances(port, var_series)
        result = kupiec_test(exceed, level=0.95)
        backtests[method] = result
        print(f"{method}: {result}")
        plot_backtest(port, var_series, exceed, title=f"{method.title()} VaR backtest")

    return {
        "returns": returns,
        "portfolio": port,
        "risk_shares": shares,
        "pca_explained": explained,
        "pca_loadings": loadings,
        "loss_table": table,
        "worst_day": (worst, worst_date),
        "backtests": backtests,
    }


if __name__ == "__main__":
    # Example: the six-asset portfolio
    tickers = ["SPY", "EFA", "IEF", "GLD", "XLE", "XME"]
    weights = [0.30, 0.15, 0.25, 0.10, 0.10, 0.10]
    risk_report(tickers, weights)