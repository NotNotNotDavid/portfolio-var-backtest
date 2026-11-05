import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.stats import norm, chi2, probplot
from scipy.special import xlogy

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

"""Returns are computed as percentages, with formula (Price_t - Price_t-1)/Price_t-1"""
def compute_returns(prices):
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

    This is the beginning of the lin alg process

    returns: DataFrame of daily asset returns (rows = dates, columns = tickers).
    w:       Series of weights, indexed by ticker, summing to 1.

    Returns: DataFrame with columns ["weight", "risk_share"]; risk_share sums to 1.
    """
    cov = returns[w.index].cov()          # Σ, in the same ticker order as w
    port_var = w @ cov @ w                # w'Σw, a single number
    contrib = w * (cov @ w)               # w_i × (Σw)_i, one per asset
    return pd.DataFrame({"weight": w, "risk_share": contrib / port_var})

def plot_return_histograms(returns, limit=0.05, bins=80):
    """One histogram of daily returns per asset, all on the same axes."""
    edges = np.linspace(-limit, limit, bins + 1)
    n = len(returns.columns)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.5), sharex=True, sharey=True)

    for ax, ticker in zip(axes, returns.columns):
        r = returns[ticker]
        ax.hist(r, bins=edges, color=WEIGHT_COLOR, alpha=0.85)
        ax.set_title(f"{ticker}\nsd = {r.std():.2%} a day", fontsize=11)
        for side in ["top", "right"]:
            ax.spines[side].set_visible(False)

    axes[0].set_ylabel("Number of days")
    fig.suptitle("Daily returns by asset", x=0.01, ha="left", fontsize=13)
    plt.tight_layout()
    plt.show()

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
    """Heatmap of the correlation matrix."""
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
    PCA via eigen-decomposition of the correlation matrix. As outlined in lectures

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
    """
    Historical VaR: the (1 - level) quantile of past returns.
    e.g. level=0.95 -> the 5th percentile. Returned as a (negative) return.
    """
    return port.quantile(1 - level)


def var_parametric(port, level=0.95):
    """
    Normal (parametric) VaR: mean + z * sd, where z = norm.ppf(1 - level).
    e.g. level=0.95 -> z = -1.645. Returned as a (negative) return.
    """
    z = norm.ppf(1 - level)
    return port.mean() + z * port.std()


def expected_shortfall(port, level=0.95): # also known as CVaR
    """
    Expected Shortfall is the average return on the days
    that were worse than historical VaR. "When a bad day happens,
    how bad is it on average?" gives us more insight onto the losses
    """
    var = var_historical(port, level)
    return port[port <= var].mean()

def worst_day(port):
    """Largest one-day loss in the data and the date it happened."""
    return port.min(), port.idxmin()


def loss_table(port, portfolio_value=1_000_000):
    """
    Summary of loss measures at 95% and 99%.

    Returns a DataFrame with rows = measures and columns:
      "95%", "99%"                 -> daily return (negative = loss)
      "95% loss ($)", "99% loss ($)" -> the same loss in dollars (positive)
    """
    rows = {}
    for name, func in [("Historical VaR", var_historical),
                       ("Parametric VaR", var_parametric),
                       ("Expected Shortfall", expected_shortfall)]:
        rows[name] = {f"{int(lvl * 100)}%": func(port, lvl) for lvl in (0.95, 0.99)}

    table = pd.DataFrame(rows).T
    for col in ["95%", "99%"]:
        table[f"{col} loss ($)"] = (-table[col] * portfolio_value).round(0)
    return table


def qq_plot(port):
    """
    QQ plot of standardized portfolio returns vs. a normal distribution.
    Points on the line = normal. Points bending away at the ends = fat tails.
    """
    z = (port - port.mean()) / port.std()
    (theoretical, sample), _ = probplot(z, dist="norm")

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(theoretical, sample, s=6, color=WEIGHT_COLOR, alpha=0.5, edgecolor="none")
    lim = max(abs(sample).max(), abs(theoretical).max()) * 1.05
    ax.plot([-lim, lim], [-lim, lim], color=TEXT_MUTED, linewidth=1, label="Normal distribution")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("Normal quantiles (standard deviations)")
    ax.set_ylabel("Portfolio return quantiles (standard deviations)")
    ax.set_title("QQ plot: portfolio returns vs. normal", loc="left", fontsize=13)
    ax.legend(frameon=False, loc="upper left")
    ax.grid(color="#e5e4e0", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    plt.tight_layout()
    plt.show()


# =====================================================================
# 4. BACKTEST - can we trust the VaR?
# =====================================================================

def rolling_var(port, method="historical", level=0.95, window=500):
    """
    VaR for each day using ONLY the previous `window` days.

    method: "historical" or "parametric"
    The .shift(1) at the end is essential - without it, each day's
    VaR would use that same day's return (look-ahead bias).
    """
    alpha = 1 - level
    rolling = port.rolling(window) # a rolling object
    if method == "historical":
        result = rolling.quantile(alpha)
    elif method == "parametric":
        result = rolling.mean() + norm.ppf(alpha) * rolling.std()
    else:
        raise ValueError('method must be "historical" or "parametric"')
    return result.shift(1) 
    # shifts results because of window being pushed by a day,
    # so a day 4 would use day 1-3 instead of day 3 using day 1-3


def exceeds(port, var_series):
    """
    True on days the return was worse than that day's VaR.
    Only counts days where a VaR exists (drops the warm-up period).
    """
    valid = var_series.notna() # removes the NaN's from previous 500
    return (port < var_series)[valid]


def kupiec_test(exceed, level=0.95):
    """
    Kupiec (1995) likelihood ratio test: is the exceedance rate equal to 1 - level?

    Each day is a Bernoulli trial (breach or not).
    H0: p = 1 - level.   Alternative: p = observed rate x / T (the MLE).
    LR = -2 * (loglik_H0 - loglik_alt) ~ chi-squared(1) under H0.

    xlogy(a, b) computes a*log(b) but returns 0 when a == 0, which handles
    the x == 0 and x == T edge cases without log(0) errors.
    """
     # TODO: x = exceed.sum(); T = len(exceed); p = 1 - level; p_hat = x / T
    # TODO: log-likelihoods, LR, p_value = 1 - chi2.cdf(LR, df=1)
    raise NotImplementedError


def exceedances_by_year(exceed, level=0.95):
    """Breaches per calendar year vs. the number expected - shows whether they cluster."""
    by_year = exceed.groupby(exceed.index.year).agg(["sum", "count"])
    by_year.columns = ["breaches", "days"]
    by_year["expected"] = (by_year["days"] * (1 - level)).round(1)
    by_year["rate"] = (by_year["breaches"] / by_year["days"]).round(3)
    by_year.index.name = "year"
    return by_year


def plot_backtest(port, var_series, exceed, title="VaR backtest"):
    """Daily returns (grey), the VaR line (blue), and breach days (red dots)."""
    valid = var_series.notna()
    r = port[valid]
    v = var_series[valid]
    breach_days = exceed[exceed].index

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(r.index, r, color="#b4b2a9", linewidth=0.6, label="Daily return")
    ax.plot(v.index, v, color=WEIGHT_COLOR, linewidth=1.5, label="VaR")
    ax.scatter(breach_days, r[breach_days], color="#e34948", s=12, zorder=3,
               label=f"Breach ({len(breach_days)} of {len(exceed)} days)")
    ax.axhline(0, color=TEXT_MUTED, linewidth=0.8)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.1%}"))
    ax.set_title(title, loc="left", fontsize=13)
    ax.set_ylabel("Daily portfolio return")
    ax.legend(frameon=False, loc="lower left")
    ax.grid(axis="y", color="#e5e4e0", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    plt.tight_layout()
    plt.show()


# =====================================================================
# 5. THE REPORT - one call does everything
# =====================================================================

def risk_report(tickers, weights, start="2012-01-01",
                portfolio_value=1_000_000, window=500, level=0.95):
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
    print(shares.round(3))
    print(f"First principal component explains {explained.iloc[0]:.1%} of total variation")
    plot_risk_shares(shares)
    plot_correlations(returns[w.index])

    # 3. Loss
    table = loss_table(port, portfolio_value)
    worst, worst_date = worst_day(port)
    print("\n== How much could we lose ==")
    print(table.round(4))
    print(f"Worst day: {worst:.2%} on {worst_date.date()}")
    qq_plot(port)

    # 4. Backtest (both methods)
    backtests = {}
    print(f"\n== Can we trust the VaR ({level:.0%}, {window}-day window) ==")
    for method in ["historical", "parametric"]:
        var_series = rolling_var(port, method=method, level=level, window=window)
        exceed = exceed(port, var_series)
        result = kupiec_test(exceed, level=level)
        backtests[method] = {"var": var_series, "exceed": exceed, "kupiec": result}
        print(f"{method}: {result}")
        plot_backtest(port, var_series, exceed,
                      title=f"{method.title()} VaR backtest ({level:.0%}, {window}-day window)")

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
    tickers = ["SPY", "EFA", "IEF", "GLD"]
    weights = [0.50, 0.25, 0.15, 0.10]
    risk_report(tickers, weights)