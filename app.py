import os
import json
import tempfile
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from portfolio_report import generate_portfolio_report
from portfolio_report_pptx import generate_pptx_report
from portfolio_factsheet_pdf import generate_factsheet
from portfolio_report_excel import generate_excel_report

TRADING_DAYS = 252

# ---------------- Page / Style ----------------
st.set_page_config(page_title="Portfolio Optimizer", layout="wide")

CSS = """
<style>
.block-container { padding-top: 1.1rem; padding-bottom: 2rem; max-width: 1350px; }
.small-muted { color: rgba(0,0,0,0.55); font-size: 0.9rem; }
.kpi { font-size: 0.95rem; font-weight: 700; margin: 0; }
.kpi-sub { color: rgba(0,0,0,0.65); font-size: 0.8rem; margin: 0; }
.kpi-amount { font-size: 0.9rem; font-weight: 700; margin: 0; white-space: nowrap; color: #31333F; }
.kpi-amount-label { color: rgba(0,0,0,0.65); font-size: 0.75rem; margin: 0; }
hr { border-color: rgba(0,0,0,0.12); }
/* Fit metrics in 8 columns */
[data-testid="stMetric"] label { font-size: 0.8rem !important; color: rgba(0,0,0,0.65) !important; }
[data-testid="stMetric"] [data-testid="stMetricValue"] { font-size: 1.15rem !important; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# Header
st.write(
        '<span class="small-muted">Cumulative returns · Frontier + CML · Cov/Cor matrices · Betas</span>',
        unsafe_allow_html=True,
)

# ---------------- Data ----------------
@st.cache_data(show_spinner=False)
def download_close_prices(tickers, start, end) -> pd.DataFrame:
    """
    Download close prices from Yahoo Finance.

    Key change: removed bfill() to avoid look-ahead bias.
    We forward-fill small gaps, then align series to a common start date by dropping
    remaining NaNs row-wise (ensures all assets have real history from that point).
    """
    raw = yf.download(tickers, start=start, end=end, progress=False)
    if raw is None or len(raw) == 0:
        return pd.DataFrame()

    # Handle yfinance column shapes
    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" not in raw.columns.get_level_values(0):
            return pd.DataFrame()
        prices = raw["Close"].copy()
    else:
        if "Close" not in raw.columns:
            return pd.DataFrame()
        prices = raw[["Close"]].copy()
        # single ticker -> rename
        if len(tickers) == 1:
            prices = prices.rename(columns={"Close": tickers[0]})

    prices = prices.dropna(how="all")

    # Forward-fill only (no bfill) to avoid look-ahead bias
    prices = prices.ffill()

    # Drop columns that are still entirely missing
    prices = prices.dropna(axis=1, how="all")

    # Align to common history: drop any rows with missing data
    # (ensures you don't fabricate early history for late-start tickers)
    prices = prices.dropna(axis=0, how="any")

    return prices


# Yahoo Finance FX ticker mapping: target currency → ticker
# Yahoo quotes XXXUSD=X as "how many USD per 1 XXX"
# We invert to get "how many XXX per 1 USD" (i.e. the rate to multiply USD prices by)
_FX_TICKERS = {
    "EUR": "EURUSD=X",   # USD per 1 EUR → inverted to EUR per 1 USD
    "GBP": "GBPUSD=X",   # USD per 1 GBP → inverted to GBP per 1 USD
    "CAD": "USDCAD=X",   # CAD per 1 USD → used directly (no inversion needed)
    "AUD": "AUDUSD=X",   # USD per 1 AUD → inverted to AUD per 1 USD
    "JPY": "USDJPY=X",   # JPY per 1 USD → used directly (no inversion needed)
}

# These tickers already give "target CCY per 1 USD", no inversion needed
_FX_DIRECT = {"CAD", "JPY"}


@st.cache_data(show_spinner=False)
def download_fx_rates(currency: str, start: str, end: str) -> pd.Series:
    """
    Download daily FX rate to convert USD prices into *currency*.
    Returns a Series of (1 / XXXUSD) so that:  price_CCY = price_USD * fx_rate
    For EUR: fx_rate = 1 / EURUSD  (i.e. how many EUR per 1 USD).
    If currency == 'USD' returns a series of 1.0.
    """
    if currency == "USD":
        return pd.Series(dtype=float)  # sentinel: no conversion needed

    # Yahoo quotes EURUSD=X as "how many USD per 1 EUR", so invert.
    ticker = _FX_TICKERS.get(currency)
    if ticker is None:
        return pd.Series(dtype=float)

    raw = yf.download(ticker, start=start, end=end, progress=False)
    if raw is None or len(raw) == 0:
        return pd.Series(dtype=float)

    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"].iloc[:, 0] if "Close" in raw.columns.get_level_values(0) else pd.Series(dtype=float)
    else:
        close = raw["Close"] if "Close" in raw.columns else pd.Series(dtype=float)

    close = close.dropna()
    if close.empty:
        return pd.Series(dtype=float)

    # For XXXUSD=X tickers (EUR, GBP, AUD): gives USD-per-XXX → invert to get XXX-per-USD
    # For USDXXX=X tickers (CAD, JPY): already gives XXX-per-USD → use directly
    if currency in _FX_DIRECT:
        fx = close.copy()
    else:
        fx = 1.0 / close
    fx = fx.ffill().bfill()
    fx.name = currency
    return fx


def convert_prices_to_currency(prices: pd.DataFrame, fx_rates: pd.Series) -> pd.DataFrame:
    """
    Multiply USD-denominated price columns by daily FX rate to get prices in
    the target currency.  If fx_rates is empty, return prices unchanged (USD).
    """
    if fx_rates.empty:
        return prices
    common = prices.index.intersection(fx_rates.index)
    if len(common) < 10:
        return prices  # not enough FX data, keep USD
    # Align and multiply
    p = prices.loc[common].copy()
    fx = fx_rates.loc[common]
    return p.multiply(fx, axis=0)


def annual_stats(prices):
    rets = prices.pct_change().dropna()
    mu = rets.mean() * TRADING_DAYS
    cov = rets.cov() * TRADING_DAYS
    return rets, mu, cov


# ---------------- Portfolio math ----------------
def random_portfolios(mu, cov, n, max_w, seed=42):
    rng = np.random.default_rng(seed)
    k = len(mu)
    def _enforce_max_weight(Warr, max_w):
        Warr = Warr.copy()
        eps = 1e-12
        for i in range(Warr.shape[0]):
            w = Warr[i]
            # Iteratively clip and redistribute until all weights <= max_w
            it = 0
            while True:
                over = w > max_w
                if not over.any() or it > 20:
                    break
                w[over] = max_w
                under = ~over
                total_under = w[under].sum()
                if total_under <= eps:
                    # fallback: equally distribute
                    w = np.ones_like(w) / len(w)
                    break
                remaining = 1.0 - w.sum()
                if remaining <= eps:
                    break
                prop = w[under] / w[under].sum()
                w[under] = w[under] + prop * remaining
                it += 1
            Warr[i] = w
        return Warr

    if max_w >= 1.0:
        W = rng.random((n, k))
        W = W / W.sum(axis=1, keepdims=True)
    else:
        # Rejection sampling in batches to collect n valid portfolios
        collected = []
        needed = n
        attempts = 0
        batch = max(n, 10000)
        while needed > 0 and attempts < 50:
            Wb = rng.random((batch, k))
            Wb = Wb / Wb.sum(axis=1, keepdims=True)
            mask = (Wb <= max_w).all(axis=1)
            good = Wb[mask]
            if good.shape[0] > 0:
                take = min(needed, good.shape[0])
                collected.append(good[:take])
                needed -= take
            attempts += 1

        if collected:
            W = np.vstack(collected)
        else:
            # Fallback to unconstrained sampling then enforce max weights
            W = rng.random((n, k))
            W = W / W.sum(axis=1, keepdims=True)

        if W.shape[0] < n:
            extra = rng.random((n - W.shape[0], k))
            extra = extra / extra.sum(axis=1, keepdims=True)
            W = np.vstack([W, extra])

        W = W[:n]
        # Ensure rows meet the max_w constraint
        W = _enforce_max_weight(W, max_w)

    pr = W @ mu.values
    pv = np.sqrt(np.einsum("ij,jk,ik->i", W, cov.values, W))
    return W, pr, pv


def portfolio_metrics_risky(w, mu, cov):
    r = float(w @ mu.values)
    v = float(np.sqrt(w @ cov.values @ w))
    return r, v


def cml_metrics(alpha, w_tan, mu, cov, rf):
    r_t, v_t = portfolio_metrics_risky(w_tan, mu, cov)
    r_p = (1 - alpha) * rf + alpha * r_t
    v_p = abs(alpha) * v_t
    s_p = (r_p - rf) / v_p if v_p > 0 else np.nan
    return float(r_p), float(v_p), float(s_p), float(r_t), float(v_t)


def rf_daily_from_annual(rf_annual: float) -> float:
    return (1.0 + float(rf_annual)) ** (1.0 / TRADING_DAYS) - 1.0


def portfolio_daily_returns_from_weights(
    weights_df: pd.DataFrame,
    asset_returns: pd.DataFrame,
    rf_annual: float,
) -> pd.Series:
    """
    Build portfolio daily return series from weights_df.
    IMPORTANT FIX: includes risk-free leg when present (RISK-FREE row).
    """
    rf_d = rf_daily_from_annual(rf_annual)

    # Start with RF contribution (supports leverage if RF weight is negative)
    w_rf = 0.0
    if "RISK-FREE" in set(weights_df["Asset"]):
        w_rf = float(weights_df.loc[weights_df["Asset"] == "RISK-FREE", "Weight"].iloc[0])

    port = pd.Series(w_rf * rf_d, index=asset_returns.index, dtype=float)

    # Add risky assets
    for _, row in weights_df.iterrows():
        asset = row["Asset"]
        if asset == "RISK-FREE":
            continue
        w = float(row["Weight"])
        if asset in asset_returns.columns:
            port = port + w * asset_returns[asset]

    return port


# ---------------- Optional cvxpy ----------------
def _has_cvxpy() -> bool:
    try:
        import cvxpy  # noqa
        return True
    except Exception:
        return False


def solve_target_vol_max_return(mu, cov, target_vol, long_only=True, max_w=1.0):
    import cvxpy as cp
    n = len(mu)
    w = cp.Variable(n)
    ret = mu.values @ w
    risk = cp.quad_form(w, cov.values)

    cons = [cp.sum(w) == 1, risk <= float(target_vol ** 2)]
    if long_only:
        cons.append(w >= 0)
    if max_w < 1.0:
        cons.append(w <= max_w)

    prob = cp.Problem(cp.Maximize(ret), cons)
    prob.solve(solver=cp.SCS, verbose=False)

    if w.value is None:
        raise RuntimeError("Target-vol optimization failed (likely infeasible).")
    wv = np.array(w.value).reshape(-1)
    wv[np.abs(wv) < 1e-8] = 0.0
    return wv / wv.sum()


def solve_min_variance(mu, cov, long_only=True, max_w=1.0):
    """Minimize variance subject to fully-invested and constraints."""
    import cvxpy as cp
    n = len(mu)
    w = cp.Variable(n)
    risk = cp.quad_form(w, cov.values)

    cons = [cp.sum(w) == 1]
    if long_only:
        cons.append(w >= 0)
    if max_w < 1.0:
        cons.append(w <= max_w)

    prob = cp.Problem(cp.Minimize(risk), cons)
    prob.solve(solver=cp.SCS, verbose=False)

    if w.value is None:
        raise RuntimeError("Min-variance optimization failed.")
    wv = np.array(w.value).reshape(-1)
    wv[np.abs(wv) < 1e-8] = 0.0
    return wv / wv.sum()


def solve_max_sharpe(mu, cov, rf, long_only=True, max_w=1.0):
    """
    More defensible Sharpe "direction" via convex program:
    maximize excess return subject to variance <= 1.

    We do NOT impose sum(w)=1 here; we normalize after to get weights that sum to 1.
    This avoids the incorrect (risk==1 AND sum(w)==1) constraint combo.
    """
    import cvxpy as cp
    n = len(mu)
    w = cp.Variable(n)

    excess = (mu.values - rf) @ w
    risk = cp.quad_form(w, cov.values)

    cons = [risk <= 1.0]
    if long_only:
        cons.append(w >= 0)
    if max_w < 1.0:
        cons.append(w <= max_w)

    prob = cp.Problem(cp.Maximize(excess), cons)
    prob.solve(solver=cp.SCS, verbose=False)

    if w.value is None:
        raise RuntimeError("Max-Sharpe optimization failed.")
    wv = np.array(w.value).reshape(-1)
    wv[np.abs(wv) < 1e-8] = 0.0

    s = float(wv.sum())
    if abs(s) < 1e-12:
        raise RuntimeError("Max-Sharpe solution is ~0; cannot normalize.")
    return wv / s


# ---------------- Risk diagnostics ----------------
def compute_betas(asset_returns: pd.DataFrame, benchmark_returns: pd.Series) -> pd.Series:
    mkt_var = float(benchmark_returns.var())
    if mkt_var <= 1e-12:
        raise RuntimeError("Benchmark variance is ~0; cannot compute betas.")
    betas = asset_returns.apply(lambda x: float(x.cov(benchmark_returns)) / mkt_var)
    return betas.astype(float)


def compute_capm_metrics(
    asset_returns: pd.DataFrame,
    benchmark_returns: pd.Series,
    rf_annual: float,
) -> pd.DataFrame:
    rf_daily = rf_daily_from_annual(rf_annual)
    mkt_excess = benchmark_returns - rf_daily

    rows = []
    for col in asset_returns.columns:
        asset_excess = asset_returns[col] - rf_daily
        common_idx = asset_excess.index.intersection(mkt_excess.index)
        if len(common_idx) < 60:
            continue
        a = asset_excess.loc[common_idx]
        m = mkt_excess.loc[common_idx]

        mkt_var = float(m.var())
        if mkt_var <= 1e-12:
            continue

        beta = float(a.cov(m)) / mkt_var
        alpha = float(a.mean() - beta * m.mean())
        corr = float(a.corr(m))
        r2 = float(corr ** 2)

        rows.append(
            {
                "Asset": col,
                "Corr (excess vs mkt)": corr,
                "Alpha (daily)": alpha,
                "Beta": beta,
                "R^2": r2,
            }
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("R^2", ascending=False).reset_index(drop=True)
    return df


# =========================
# UI (Inputs)
# =========================
st.title("📈 Portfolio Optimizer")
st.write(
    '<span class="small-muted">Cumulative returns · Frontier + CML · Cov/Cor matrices · Betas</span>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown('<div class="sidebar-section"><div class="section-label">Assets</div></div>', unsafe_allow_html=True)
    tickers_text = st.text_input("Tickers (comma-separated)", "AAPL,MSFT,GOOGL,AMZN,NVDA,META,TSLA")
    tickers = [t.strip().upper() for t in tickers_text.split(",") if t.strip()]

    st.markdown('<div class="sidebar-section"><div class="section-label">Period</div></div>', unsafe_allow_html=True)
    start = st.date_input("Start", value=pd.to_datetime("2019-08-01"))
    end = st.date_input("End", value=pd.to_datetime("today"))

    st.markdown('<div class="sidebar-section"><div class="section-label">Capital Allocation</div></div>', unsafe_allow_html=True)
    currency_choice = st.selectbox("Currency", ["EUR", "USD", "GBP", "CAD", "AUD", "JPY"], index=0)
    investment_amount = st.number_input(f"Investment amount ({currency_choice})", min_value=0.0, value=10000.0, step=100.0, format="%.2f")
    investment_horizon = st.slider("Investment horizon (years)", 1, 30, 5, 1)

    st.markdown('<div class="sidebar-section"><div class="section-label">Benchmark (Beta)</div></div>', unsafe_allow_html=True)
    benchmark = st.text_input("Market benchmark ticker", "SPY").strip().upper()
    st.caption(f"Benchmark returns will be converted to {currency_choice}.")

    st.markdown('<div class="sidebar-section"><div class="section-label">Risk-Free & Objective</div></div>', unsafe_allow_html=True)
    rf_pct = st.slider("Risk-Free rate (annual %) ", 0.0, 20.0, 2.0, 0.1)
    rf = float(rf_pct) / 100.0
    include_rf = st.checkbox("Include Risk-Free asset (CML)", value=True)
    leverage_cap = st.slider("Max risky exposure alpha (CML)", 0.0, 3.0, 2.0, 0.05)

    st.markdown('<div class="sidebar-section"><div class="section-label">Constraints</div></div>', unsafe_allow_html=True)
    long_only = st.checkbox("Long-only (no shorting)", value=True)
    max_w = st.slider("Max weight per asset", 0.10, 1.00, 1.00, 0.05)

    st.markdown('<div class="sidebar-section"><div class="section-label">Engine</div></div>', unsafe_allow_html=True)
    use_cvxpy = st.checkbox("Use true optimizer (cvxpy when available)", value=True)
    n_sims = st.slider("Simulation count (frontier cloud)", 2000, 80000, 25000, 2500)

    st.markdown('<div class="sidebar-section"><div class="section-label">Optimization Method</div></div>', unsafe_allow_html=True)
    portfolio_strategy = st.selectbox(
        "Method",
        ["Custom Allocation", "Max Sharpe", "Minimum Variance", "Target Volatility"],
    )
    target_vol_input = None
    if portfolio_strategy == "Target Volatility":
        # slider inputs shown as percent in UI; keep decimal used elsewhere
        tv_pct = st.slider("Target Volatility (annual %)", 1, 40, 15)
        target_vol_input = float(tv_pct) / 100.0

    # Pre-set target weights for the Magnificent 7
    _DEFAULT_WEIGHTS = {
        "AAPL": 14.3, "MSFT": 14.3, "GOOGL": 14.3, "AMZN": 14.3,
        "NVDA": 14.3, "META": 14.3, "TSLA": 14.2,
    }

    custom_weights_input = {}
    custom_rf_weight = 0.0
    if portfolio_strategy == "Custom Allocation":
        st.markdown("**Set weight for each asset (%)**")
        n_assets = len(tickers)
        if include_rf:
            default_equal = round(100.0 / max(n_assets + 1, 1), 1)
            custom_rf_weight = st.number_input(
                "RISK-FREE %", min_value=0.0, max_value=100.0,
                value=default_equal,
                step=0.5, key="cw_RISK_FREE"
            )
        else:
            default_equal = round(100.0 / max(n_assets, 1), 1)
        for tk in tickers:
            custom_weights_input[tk] = st.number_input(
                f"{tk} %", min_value=0.0, max_value=100.0,
                value=_DEFAULT_WEIGHTS.get(tk, default_equal),
                step=0.5, key=f"cw_{tk}"
            )
        total_pct = sum(custom_weights_input.values()) + custom_rf_weight
        if abs(total_pct - 100.0) > 0.5:
            st.warning(f"⚠️ Weights sum to {total_pct:.1f}% — they should add up to 100%.")
    
    # ---------------- Figma section ----------------
    # (no additional sidebar integrations)

# Guard
if len(tickers) < 2:
    st.warning("Add at least **2 tickers**.")
    st.stop()

# =========================
# Compute results
# =========================
try:
    prices_usd = download_close_prices(tickers, str(start), str(end))
    if prices_usd.empty or prices_usd.shape[0] < 60:
        st.error("Not enough price data. Try different tickers or a wider date range.")
        st.stop()

    # ── FX conversion: convert all prices to the selected currency ──
    fx_rates = download_fx_rates(currency_choice, str(start), str(end))
    if currency_choice != "USD" and fx_rates.empty:
        st.warning(
            f"⚠️ Could not download {currency_choice}/USD exchange rates. "
            f"Falling back to USD-denominated returns."
        )
    prices = convert_prices_to_currency(prices_usd, fx_rates)
    if prices.empty or prices.shape[0] < 60:
        st.error("Not enough price data after FX conversion. Try a wider date range.")
        st.stop()

    rets, mu, cov = annual_stats(prices)
    cum_returns = (1 + rets).cumprod() * 100

    cov_matrix = cov.copy()
    corr_matrix = rets.corr()

    # Betas vs benchmark (also converted to the selected currency)
    betas = None
    capm_df = pd.DataFrame()
    bench_prices_usd = download_close_prices([benchmark], str(start), str(end))
    bench_prices = convert_prices_to_currency(bench_prices_usd, fx_rates)
    if not bench_prices.empty:
        bench_rets = bench_prices.pct_change().dropna().iloc[:, 0]
        common_idx = rets.index.intersection(bench_rets.index)
        if len(common_idx) >= 60:
            betas = compute_betas(rets.loc[common_idx], bench_rets.loc[common_idx])
            capm_df = compute_capm_metrics(rets.loc[common_idx], bench_rets.loc[common_idx], rf)
        else:
            st.warning(f"Not enough overlapping dates to compute betas vs {benchmark}.")
    else:
        st.warning(f"Could not download benchmark {benchmark}. Betas not available.")

    # Frontier cloud
    W_cloud, pr_cloud, pv_cloud = random_portfolios(mu, cov, n_sims, max_w=max_w)
    # Avoid divide-by-zero / infs in Sharpe calculation
    sharpe_cloud = np.where(pv_cloud > 1e-12, (pr_cloud - rf) / pv_cloud, np.nan)

    cvxpy_ok = _has_cvxpy()
    if use_cvxpy and not cvxpy_ok:
        st.info("cvxpy not installed → target-vol risky-only uses simulation. (pip install cvxpy)")

    # Tangency portfolio (risky-only)
    if use_cvxpy and cvxpy_ok:
        try:
            w_tan = solve_max_sharpe(mu, cov, rf, long_only=long_only, max_w=max_w)
        except Exception:
            w_tan = W_cloud[int(np.nanargmax(sharpe_cloud))]
    else:
        w_tan = W_cloud[int(np.nanargmax(sharpe_cloud))]

    r_tan, v_tan = portfolio_metrics_risky(w_tan, mu, cov)

    # Determine portfolio strategy
    active_strategy = portfolio_strategy
    active_target_vol = target_vol_input
    
    selected_label = f"{active_strategy}"

    # Build portfolio based on selected strategy
    if active_strategy == "Target Volatility" and active_target_vol is not None:
        target_vol = float(active_target_vol)
        
        if include_rf:
            alpha = 0.0 if v_tan <= 1e-12 else float(target_vol / v_tan)
            alpha = max(0.0, min(alpha, leverage_cap))
            selected_label = f"Target Vol {target_vol:.0%} (CML, α={alpha:.2f})"
            w_rf = 1.0 - alpha
            w_risky = alpha * w_tan
            port_r, port_v, port_s, _, _ = cml_metrics(alpha, w_tan, mu, cov, rf)
            weights_df = pd.DataFrame(
                {"Asset": ["RISK-FREE"] + list(prices.columns),
                 "Weight": [w_rf] + list(w_risky)}
            )
        else:
            if use_cvxpy and cvxpy_ok:
                try:
                    w_risky = solve_target_vol_max_return(mu, cov, target_vol, long_only=long_only, max_w=max_w)
                    selected_label = f"Target Vol {target_vol:.0%} (risky-only, cvxpy)"
                except Exception:
                    idx = int(np.nanargmin(np.abs(pv_cloud - target_vol)))
                    w_risky = W_cloud[idx]
                    selected_label = f"Target Vol {target_vol:.0%} (risky-only, simulation)"
            else:
                idx = int(np.nanargmin(np.abs(pv_cloud - target_vol)))
                w_risky = W_cloud[idx]
                selected_label = f"Target Vol {target_vol:.0%} (risky-only, simulation)"
            
            port_r, port_v = portfolio_metrics_risky(w_risky, mu, cov)
            port_s = (port_r - rf) / port_v if port_v > 0 else np.nan
            weights_df = pd.DataFrame({"Asset": list(prices.columns), "Weight": list(w_risky)})

    elif active_strategy == "Minimum Variance":
        # Min variance portfolio
        w_min = W_cloud[int(np.nanargmin(pv_cloud))]
        if use_cvxpy and cvxpy_ok:
            try:
                w_min = solve_min_variance(mu, cov, long_only=long_only, max_w=max_w)
            except Exception:
                pass
        
        if include_rf:
            alpha = 1.0
            w_rf = 1.0 - alpha
            w_risky = alpha * w_min
            selected_label = "Min Variance (CML, α=1)"
            port_r, port_v, port_s, _, _ = cml_metrics(alpha, w_min, mu, cov, rf)
            weights_df = pd.DataFrame(
                {"Asset": ["RISK-FREE"] + list(prices.columns),
                 "Weight": [w_rf] + list(w_risky)}
            )
        else:
            port_r, port_v = portfolio_metrics_risky(w_min, mu, cov)
            port_s = (port_r - rf) / port_v if port_v > 0 else np.nan
            selected_label = "Min Variance (risky-only)"
            weights_df = pd.DataFrame({"Asset": list(prices.columns), "Weight": list(w_min)})

    elif active_strategy == "Custom Allocation" and custom_weights_input:
        total_pct = sum(custom_weights_input.values()) + custom_rf_weight
        w_raw = np.array([custom_weights_input.get(t, 0.0) for t in prices.columns], dtype=float)
        w_rf_raw = float(custom_rf_weight)

        if total_pct > 0:
            scale = 1.0 / total_pct
            w_custom = w_raw * scale
            w_rf = w_rf_raw * scale
        else:
            denom = len(prices.columns) + (1 if include_rf else 0)
            w_custom = np.ones(len(prices.columns)) / max(denom, 1)
            w_rf = 1.0 / denom if include_rf else 0.0

        weights_rows = [{"Asset": "RISK-FREE", "Weight": w_rf}] if include_rf else []
        weights_rows += [{"Asset": tk, "Weight": float(w)} for tk, w in zip(prices.columns, w_custom)]
        weights_df = pd.DataFrame(weights_rows)

        # Compute moments (expected) from weights
        if include_rf:
            alpha = 1.0 - w_rf
            w_risky_norm = w_custom / w_custom.sum() if w_custom.sum() > 1e-12 else w_custom
            r_risky, v_risky = portfolio_metrics_risky(w_risky_norm, mu, cov)
            port_r = w_rf * rf + alpha * r_risky
            port_v = abs(alpha) * v_risky
            port_s = (port_r - rf) / port_v if port_v > 0 else np.nan
            selected_label = f"Custom Allocation (RF {w_rf:.0%})"
        else:
            w_custom = w_custom / w_custom.sum() if w_custom.sum() > 1e-12 else w_custom
            port_r, port_v = portfolio_metrics_risky(w_custom, mu, cov)
            port_s = (port_r - rf) / port_v if port_v > 0 else np.nan
            selected_label = "Custom Allocation"

    else:  # Max Sharpe (default)
        if include_rf:
            alpha = 1.0
            w_rf = 1.0 - alpha
            w_risky = alpha * w_tan
            selected_label = "Max Sharpe (CML, α=1)"
            port_r, port_v, port_s, _, _ = cml_metrics(alpha, w_tan, mu, cov, rf)
            weights_df = pd.DataFrame(
                {"Asset": ["RISK-FREE"] + list(prices.columns),
                 "Weight": [w_rf] + list(w_risky)}
            )
        else:
            w_risky = w_tan
            selected_label = "Max Sharpe (risky-only)"
            port_r, port_v = portfolio_metrics_risky(w_risky, mu, cov)
            port_s = (port_r - rf) / port_v if port_v > 0 else np.nan
            weights_df = pd.DataFrame({"Asset": list(prices.columns), "Weight": list(w_risky)})


    # Clean weights table
    weights_df["Weight"] = weights_df["Weight"].astype(float)
    weights_df = weights_df.sort_values("Weight", ascending=False).reset_index(drop=True)

    # ====== IMPORTANT FIX: portfolio daily returns INCLUDE RISK-FREE leg ======
    port_daily_rets = portfolio_daily_returns_from_weights(weights_df, rets, rf)

    total_period_return = float((1 + port_daily_rets).prod() - 1)
    n_years = len(rets) / TRADING_DAYS
    total_return_cash = float(investment_amount) * total_period_return

    # ── Additional risk metrics ──
    # Max drawdown
    port_cum = (1 + port_daily_rets).cumprod()
    running_max = port_cum.cummax()
    drawdown_series = (port_cum - running_max) / running_max
    max_drawdown = float(drawdown_series.min()) if len(drawdown_series) > 0 else 0.0

    # Sortino Ratio
    rf_daily = rf_daily_from_annual(rf)
    excess_daily = port_daily_rets - rf_daily
    downside = excess_daily[excess_daily < 0]
    downside_std = float(np.sqrt((downside ** 2).mean()) * np.sqrt(TRADING_DAYS)) if len(downside) > 0 else np.nan
    sortino_ratio = float((port_r - rf) / downside_std) if downside_std > 1e-12 else np.nan

    # Tracking Error & Information Ratio (vs benchmark)
    tracking_error = np.nan
    information_ratio = np.nan
    port_alpha = np.nan
    port_beta = np.nan
    port_r_squared = np.nan
    if not bench_prices.empty:
        bench_rets_daily = bench_prices.pct_change().dropna().iloc[:, 0]
        common_idx_te = port_daily_rets.index.intersection(bench_rets_daily.index)
        if len(common_idx_te) >= 60:
            active_rets = port_daily_rets.loc[common_idx_te] - bench_rets_daily.loc[common_idx_te]
            tracking_error = float(active_rets.std() * np.sqrt(TRADING_DAYS))
            information_ratio = float(active_rets.mean() * TRADING_DAYS / tracking_error) if tracking_error > 1e-12 else np.nan
            # Portfolio-level alpha/beta on excess returns
            bx = (bench_rets_daily.loc[common_idx_te] - rf_daily).values
            by = (port_daily_rets.loc[common_idx_te] - rf_daily).values
            bx_mean = float(np.mean(bx))
            by_mean = float(np.mean(by))
            denom_b = float(np.sum((bx - bx_mean) ** 2))
            port_beta = float(np.sum((bx - bx_mean) * (by - by_mean)) / denom_b) if denom_b > 1e-12 else 0.0
            port_alpha_daily = float(by_mean - port_beta * bx_mean)
            port_alpha = float((1 + port_alpha_daily) ** TRADING_DAYS - 1)
            _corr = float(np.corrcoef(bx, by)[0, 1]) if len(bx) > 1 else np.nan
            port_r_squared = float(_corr ** 2) if np.isfinite(_corr) else np.nan

    # Save context
    st.session_state.app_context = {
        "selected_label": selected_label,
        "rf": rf,
        "port_r": port_r,
        "port_v": port_v,
        "port_s": port_s,
        "total_period_return": total_period_return,
        "total_return_cash": total_return_cash,
        "n_years": n_years,
        "weights_df": weights_df,
        "investment_amount": float(investment_amount),
        "expected_return_dollars": float(investment_amount) * float(port_r),
        "currency_choice": currency_choice,
        "betas": betas,
        "capm_df": capm_df,
        "benchmark": benchmark,
        "cov_matrix": cov_matrix,
        "corr_matrix": corr_matrix,
        "max_drawdown": max_drawdown,
        "sortino_ratio": sortino_ratio,
        "tracking_error": tracking_error,
        "information_ratio": information_ratio,
        "port_alpha": port_alpha,
        "port_beta": port_beta,
        "port_r_squared": port_r_squared,
        "port_daily_rets": port_daily_rets,
        "rets": rets,
        "mu": mu,
        "cov": cov,
        "w_tan": w_tan,
        "pv_cloud": pv_cloud,
        "pr_cloud": pr_cloud,
        "sharpe_cloud": sharpe_cloud,
        "cum_returns": cum_returns,
        "bench_prices": bench_prices,
    }

except Exception as e:
    st.error("The app crashed. Here is the exact error:")
    st.exception(e)
    st.stop()

# =========================
# KPIs (styled cards)
# =========================
max_sharpe_cloud = float(np.nanmax(sharpe_cloud)) if len(sharpe_cloud) > 0 else np.nan
st.markdown("### Key Metrics")
st.info(f"📐 All risk and performance metrics are measured in **{currency_choice}**. "
        f"Asset and benchmark returns have been converted to {currency_choice}.")
k1, k2, k3, k4, k5, k6, k7, k8 = st.columns(8)
with k1:
    st.metric("Annual Return", f"{port_r:.2%}")
with k2:
    st.metric("Volatility", f"{port_v:.2%}")
with k3:
    st.metric("Your Sharpe", f"{port_s:.2f}")
with k4:
    st.metric("Max Sharpe", f"{max_sharpe_cloud:.2f}")
with k5:
    expected_return_value = float(investment_amount) * float(port_r)
    st.markdown(
        f"<div class=\"kpi-amount\">{expected_return_value:,.2f} {currency_choice}</div>"
        f"<div class=\"kpi-amount-label\">Expected Annual ({currency_choice})</div>",
        unsafe_allow_html=True,
    )
with k6:
    st.metric(f"Total Return ({n_years:.1f}y)", f"{total_period_return:.2%}")
with k7:
    st.markdown(
        f"<div class=\"kpi-amount\">{total_return_cash:+,.2f} {currency_choice}</div>"
        f"<div class=\"kpi-amount-label\">Total P&L ({currency_choice})</div>",
        unsafe_allow_html=True,
    )
with k8:
    st.info(f"📋 {selected_label}")

# Second row of KPIs
k9, k10, k11, k12, k13, k14 = st.columns(6)
with k9:
    st.metric("Sortino", f"{sortino_ratio:.2f}" if np.isfinite(sortino_ratio) else "N/A")
with k10:
    st.metric("Max Drawdown", f"{max_drawdown:.2%}")
with k11:
    st.metric("Tracking Error", f"{tracking_error:.2%}" if np.isfinite(tracking_error) else "N/A")
with k12:
    st.metric("Info Ratio", f"{information_ratio:.2f}" if np.isfinite(information_ratio) else "N/A")
with k13:
    st.metric(f"Alpha vs {benchmark}", f"{port_alpha:.2%}" if np.isfinite(port_alpha) else "N/A")
with k14:
    st.metric(f"Beta vs {benchmark}", f"{port_beta:.2f}" if np.isfinite(port_beta) else "N/A")

# Third row – Expected Future Value (lognormal bands)
def future_value_lognormal(
    pv: float, mu_ann: float, sigma_ann: float, years: int
) -> dict:
    """
    Expected value uses arithmetic compounding: E[V]=V0*(1+mu)^T.
    Bands use lognormal: ln(V_T/V0) ~ N( (ln(1+mu)-0.5*sigma^2)T, sigma*sqrt(T) ).
    """
    T = float(years)
    pv = float(pv)
    mu_ann = float(mu_ann)
    sigma_ann = float(max(0.0, sigma_ann))
    mu_for_log = max(mu_ann, -0.9999)

    expected = pv * (1.0 + mu_ann) ** years
    mu_log = (np.log1p(mu_for_log) - 0.5 * sigma_ann ** 2) * T
    sig_log = sigma_ann * np.sqrt(T)

    median = pv * np.exp(mu_log)
    upper_1s = pv * np.exp(mu_log + 1.0 * sig_log)
    lower_1s = pv * np.exp(mu_log - 1.0 * sig_log)

    return {
        "expected": float(expected),
        "median": float(median),
        "upper_1s": float(upper_1s),
        "lower_1s": float(lower_1s),
    }

fv = future_value_lognormal(investment_amount, port_r, port_v, investment_horizon)
expected_future_value = fv["expected"]
expected_pnl_horizon = expected_future_value - float(investment_amount)
expected_cagr = float(port_r)

st.markdown(f"### Expected Future Value ({investment_horizon}Y Horizon)")
f1, f2, f3, f4, f5 = st.columns(5)
with f1:
    st.markdown(
        f"<div class=\"kpi-amount\">{expected_future_value:,.2f} {currency_choice}</div>"
        f"<div class=\"kpi-amount-label\">Expected Value ({investment_horizon}Y)</div>",
        unsafe_allow_html=True,
    )
with f2:
    st.markdown(
        f"<div class=\"kpi-amount\">{expected_pnl_horizon:+,.2f} {currency_choice}</div>"
        f"<div class=\"kpi-amount-label\">Expected P&L ({investment_horizon}Y)</div>",
        unsafe_allow_html=True,
    )
with f3:
    st.metric(f"CAGR ({investment_horizon}Y)", f"{expected_cagr:.2%}")
with f4:
    st.markdown(
        f"<div class=\"kpi-amount\">{fv['upper_1s']:,.2f} {currency_choice}</div>"
        f"<div class=\"kpi-amount-label\">Optimistic (+1σ, lognormal)</div>",
        unsafe_allow_html=True,
    )
with f5:
    st.markdown(
        f"<div class=\"kpi-amount\">{fv['lower_1s']:,.2f} {currency_choice}</div>"
        f"<div class=\"kpi-amount-label\">Pessimistic (−1σ, lognormal)</div>",
        unsafe_allow_html=True,
    )

st.markdown("---")

# =========================
# PDF Report Download
# =========================
def _build_report_data() -> dict:
    """
    IMPORTANT FIX: report portfolio series uses the same correct portfolio daily returns
    including the risk-free leg when enabled.
    """
    # Use already-correct daily series from session state
    port_daily = st.session_state.app_context["port_daily_rets"]
    port_cum = (1 + port_daily).cumprod()

    perf = {}
    total_days = len(port_cum)
    if total_days > 21:
        perf["1M"] = float(port_cum.iloc[-1] / port_cum.iloc[-22] - 1)
    first_of_year = port_cum.loc[port_cum.index >= pd.Timestamp(port_cum.index[-1].year, 1, 1)]
    if len(first_of_year) > 1:
        perf["YTD"] = float(first_of_year.iloc[-1] / first_of_year.iloc[0] - 1)
    if total_days > 252:
        perf["1Y"] = float(port_cum.iloc[-1] / port_cum.iloc[-253] - 1)
    # 3Y and 5Y annualized
    if total_days > 252 * 3:
        cum_3y = float(port_cum.iloc[-1] / port_cum.iloc[-252 * 3] - 1)
        perf["3Y Ann."] = float((1 + cum_3y) ** (1 / 3) - 1)
    if total_days > 252 * 5:
        cum_5y = float(port_cum.iloc[-1] / port_cum.iloc[-252 * 5] - 1)
        perf["5Y Ann."] = float((1 + cum_5y) ** (1 / 5) - 1)
    perf["Since Inception"] = float(port_cum.iloc[-1] / port_cum.iloc[0] - 1)

    # Calendar year returns
    cal_year_returns = {}
    port_cum_idx = port_cum.copy()
    for yr in sorted(port_cum_idx.index.year.unique()):
        yr_data = port_cum_idx.loc[port_cum_idx.index.year == yr]
        if len(yr_data) > 1:
            cal_year_returns[str(yr)] = float(yr_data.iloc[-1] / yr_data.iloc[0] - 1)

    # Scale cumulative to base 100 for chart
    port_cum_chart = port_cum * 100

    # Max drawdown from cumulative series
    running_max = port_cum.cummax()
    drawdown = (port_cum - running_max) / running_max
    max_dd_report = float(drawdown.min()) if len(drawdown) > 0 else 0.0

    # Future value (lognormal bands) for report
    fv2 = future_value_lognormal(investment_amount, port_r, port_v, investment_horizon)

    weights_df_local = st.session_state.app_context["weights_df"].copy()
    h_df = weights_df_local.copy()
    h_df.columns = ["Ticker", "Weight"]
    h_df["Market Value"] = h_df["Weight"] * float(investment_amount)
    h_df["P/L"] = h_df["Market Value"] * float(port_r)

    bench_cum = None
    try:
        bench_prices_local = st.session_state.app_context["bench_prices"]
        if bench_prices_local is not None and not bench_prices_local.empty:
            b_rets = bench_prices_local.pct_change().dropna().iloc[:, 0]
            common = port_daily.index.intersection(b_rets.index)
            if len(common) > 10:
                bench_cum = (1 + b_rets.loc[common]).cumprod() * 100
    except Exception:
        pass

    # CML curve
    _cml_v, _cml_r = None, None
    if include_rf:
        _alphas = np.linspace(0.0, leverage_cap, 80)
        _cml_v, _cml_r = [], []
        for _a in _alphas:
            _rr, _vv, _, _, _ = cml_metrics(float(_a), st.session_state.app_context["w_tan"], mu, cov, rf)
            _cml_r.append(float(_rr))
            _cml_v.append(float(_vv))

    # Auto-generated commentary
    _commentary = [
        f"Strategy: {selected_label}. "
        f"Expected annual return {port_r:.2%} with {port_v:.2%} volatility (Sharpe {port_s:.2f}).",
        f"All risk and performance metrics are measured in {currency_choice}. "
        f"Asset and benchmark returns have been converted to {currency_choice}.",
        f"Assets analyzed over {start} – {end}. "
        f"Risk-free rate assumed at {rf:.2%} annually. Benchmark: {benchmark} (returns in {currency_choice}).",
    ]

    return {
        "client_name": "Portfolio Optimizer Client",
        "report_date": pd.Timestamp.today().strftime("%Y-%m-%d"),
        "portfolio_value": float(investment_amount),
        "currency": currency_choice,
        "performance": perf,
        "risk": {
            "Volatility": float(port_v),
            "Sharpe Ratio": float(port_s),
            "Sortino Ratio": float(sortino_ratio) if np.isfinite(sortino_ratio) else None,
            "Max Drawdown": float(max_dd_report),
            "Tracking Error": float(tracking_error) if np.isfinite(tracking_error) else None,
            "Information Ratio": float(information_ratio) if np.isfinite(information_ratio) else None,
            "Alpha": float(port_alpha) if np.isfinite(port_alpha) else None,
            "Beta": float(port_beta) if np.isfinite(port_beta) else None,
            "R-Squared": float(port_r_squared) if np.isfinite(port_r_squared) else None,
        },
        "calendar_year_returns": cal_year_returns,
        "inception_date": str(start),
        "num_holdings": int((weights_df_local["Weight"].abs() > 0.001).sum()),
        "weights": {row["Asset"]: float(row["Weight"]) for _, row in weights_df_local.iterrows()},
        "holdings_df": h_df,
        "portfolio_series": port_cum_chart,
        "benchmark_series": bench_cum,
        "portfolio_label": selected_label,
        "benchmark_label": benchmark,
        "frontier": {
            "pv_cloud": list(st.session_state.app_context["pv_cloud"]),
            "pr_cloud": list(st.session_state.app_context["pr_cloud"]),
            "sharpe_cloud": list(st.session_state.app_context["sharpe_cloud"]),
            "port_v": float(port_v),
            "port_r": float(port_r),
            "cml_v": _cml_v,
            "cml_r": _cml_r,
            "selected_label": selected_label,
        },
        "cum_returns": st.session_state.app_context["cum_returns"],
        "corr_matrix": corr_matrix,
        "cov_matrix": cov_matrix,
        "betas": betas,
        "capm_df": capm_df if capm_df is not None and not capm_df.empty else None,
        "investment_horizon": investment_horizon,
        "expected_future_value": fv2["expected"],
        "expected_pnl_horizon": fv2["expected"] - float(investment_amount),
        "fv_upper": fv2["upper_1s"],
        "fv_lower": fv2["lower_1s"],
        "commentary": _commentary,
        "measurement_currency_note": (
            f"All risk and performance metrics are measured in {currency_choice}. "
            f"Asset and benchmark returns have been converted to {currency_choice}."
        ),
    }

dl1, dl2, dl3, dl4 = st.columns(4)
with dl1:
    if st.button("📄 Generate PDF Report"):
        with st.spinner("Generating PDF…"):
            rd = _build_report_data()
            tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
            generate_portfolio_report(rd, tmp.name)
            with open(tmp.name, "rb") as f:
                pdf_bytes = f.read()
            os.unlink(tmp.name)
        st.download_button(
            label="⬇️ Save PDF",
            data=pdf_bytes,
            file_name=f"portfolio_report_{pd.Timestamp.today().strftime('%Y%m%d')}.pdf",
            mime="application/pdf",
        )
with dl2:
    if st.button("📊 Generate PowerPoint Report"):
        with st.spinner("Generating PPTX…"):
            rd = _build_report_data()
            tmp = tempfile.NamedTemporaryFile(suffix=".pptx", delete=False)
            generate_pptx_report(rd, tmp.name)
            with open(tmp.name, "rb") as f:
                pptx_bytes = f.read()
            os.unlink(tmp.name)
        st.download_button(
            label="⬇️ Save PowerPoint",
            data=pptx_bytes,
            file_name=f"portfolio_report_{pd.Timestamp.today().strftime('%Y%m%d')}.pptx",
            mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
        )
with dl3:
    if st.button("📋 Generate Fund Factsheet"):
        with st.spinner("Generating Factsheet…"):
            rd = _build_report_data()
            tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
            generate_factsheet(rd, tmp.name)
            with open(tmp.name, "rb") as f:
                fs_bytes = f.read()
            os.unlink(tmp.name)
        st.download_button(
            label="⬇️ Save Factsheet",
            data=fs_bytes,
            file_name=f"fund_factsheet_{pd.Timestamp.today().strftime('%Y%m%d')}.pdf",
            mime="application/pdf",
        )
with dl4:
    if st.button("📊 Generate Excel Report"):
        with st.spinner("Generating Excel…"):
            rd = _build_report_data()
            tmp = tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False)
            generate_excel_report(rd, tmp.name)
            with open(tmp.name, "rb") as f:
                xlsx_bytes = f.read()
            os.unlink(tmp.name)
        st.download_button(
            label="⬇️ Save Excel",
            data=xlsx_bytes,
            file_name=f"portfolio_report_{pd.Timestamp.today().strftime('%Y%m%d')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

# =========================
# Two-tab layout
# =========================
tab1, tab2 = st.tabs(["Portfolio Overview", "Frontier & Returns"])

with tab1:
    st.subheader("Weights")
    st.dataframe(weights_df, width='stretch', height=240)

    fig_w = go.Figure()
    fig_w.add_trace(go.Bar(x=weights_df["Asset"], y=weights_df["Weight"]))
    fig_w.update_layout(height=290, margin=dict(l=10, r=10, t=40, b=10), yaxis_title="Weight", title="Allocation")
    st.plotly_chart(fig_w, use_container_width=True)

    # --- Normal Distribution of Portfolio Returns ---
    st.subheader("Return Distribution (Normal Approximation)")
    from scipy.stats import norm
    _nd_x = np.linspace(port_r - 4 * port_v, port_r + 4 * port_v, 400)
    _nd_y = norm.pdf(_nd_x, loc=port_r, scale=port_v) if port_v > 1e-12 else np.zeros_like(_nd_x)

    fig_nd = go.Figure()

    # Shaded areas for σ bands (±3σ, ±2σ, ±1σ — layered wide to narrow)
    for n_sig, color, label in [
        (3, "rgba(239,85,59,0.10)", "±3σ (99.7%)"),
        (2, "rgba(255,161,90,0.18)", "±2σ (95.4%)"),
        (1, "rgba(99,110,250,0.22)", "±1σ (68.3%)"),
    ]:
        lo, hi = port_r - n_sig * port_v, port_r + n_sig * port_v
        mask = (_nd_x >= lo) & (_nd_x <= hi)
        fig_nd.add_trace(go.Scatter(
            x=_nd_x[mask], y=_nd_y[mask],
            fill="tozeroy", mode="none",
            fillcolor=color, name=label,
            hoverinfo="skip",
        ))

    # PDF curve
    fig_nd.add_trace(go.Scatter(
        x=_nd_x, y=_nd_y, mode="lines",
        line=dict(color="#636EFA", width=2.5), name="PDF",
        hovertemplate="Return: %{x:.2%}<br>Density: %{y:.4f}<extra></extra>",
    ))

    # Mean line
    fig_nd.add_vline(x=port_r, line_dash="dash", line_color="#636EFA",
                     annotation_text=f"μ = {port_r:.2%}", annotation_position="top right")

    # ±1σ, ±2σ lines
    for n_sig, dash, col in [(1, "dot", "#00CC96"), (2, "dashdot", "#FFA15A")]:
        fig_nd.add_vline(x=port_r - n_sig * port_v, line_dash=dash, line_color=col,
                         annotation_text=f"-{n_sig}σ ({port_r - n_sig * port_v:.2%})",
                         annotation_position="bottom left")
        fig_nd.add_vline(x=port_r + n_sig * port_v, line_dash=dash, line_color=col,
                         annotation_text=f"+{n_sig}σ ({port_r + n_sig * port_v:.2%})",
                         annotation_position="bottom right")

    fig_nd.update_layout(
        height=360, margin=dict(l=10, r=10, t=40, b=10),
        xaxis_title="Annualized Return",
        yaxis_title="Probability Density",
        xaxis_tickformat=".1%",
        title=f"Normal Distribution — μ={port_r:.2%}, σ={port_v:.2%}",
        showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_nd, use_container_width=True)

    if investment_amount > 0:
        st.subheader("Capital Allocation")
        expected_return_dollars = float(investment_amount) * float(port_r)
        st.metric(f"Expected Annual Return ({currency_choice})", f"{expected_return_dollars:,.2f} {currency_choice}")

        alloc_df = weights_df.copy()
        alloc_df[f"Allocation ({currency_choice})"] = alloc_df["Weight"] * float(investment_amount)
        st.dataframe(alloc_df, width='stretch', height=240)

    st.subheader("Betas & Matrices")
    if betas is not None:
        beta_df = pd.DataFrame({"Asset": betas.index, f"Beta vs {benchmark}": betas.values})
        beta_df = beta_df.sort_values(f"Beta vs {benchmark}", ascending=False).reset_index(drop=True)
        st.write("**Betas**")
        st.dataframe(beta_df, width='stretch', height=200)
    else:
        st.info("Betas not available (benchmark download failed or not enough overlap).")

    show_corr = st.checkbox("Show correlation (instead of covariance)", value=False)
    mat = corr_matrix if show_corr else cov_matrix
    st.dataframe(mat, width='stretch', height=250)

with tab2:
    st.subheader("Cumulative Returns (Base = 100)")
    fig_ret = go.Figure()
    for t in cum_returns.columns:
        fig_ret.add_trace(go.Scatter(x=cum_returns.index, y=cum_returns[t], mode="lines", name=t))
    fig_ret.update_layout(height=360, margin=dict(l=10, r=10, t=40, b=10), yaxis_title="Growth of 100")
    st.plotly_chart(fig_ret, use_container_width=True)

    st.subheader("Frontier (risky-only) + CML (if enabled)")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=pv_cloud, y=pr_cloud, mode="markers", marker=dict(size=4), name="RISKY portfolios"))
    
    # Add max Sharpe marker from cloud
    i_max_sharpe = int(np.nanargmax(sharpe_cloud))
    fig.add_trace(go.Scatter(x=[pv_cloud[i_max_sharpe]], y=[pr_cloud[i_max_sharpe]], 
                            mode="markers", marker=dict(size=12, symbol="star", color="gold"), 
                            name="Max Sharpe (cloud)"))
    
    fig.add_trace(go.Scatter(x=[port_v], y=[port_r], mode="markers", marker=dict(size=14, symbol="diamond"), name="Selected"))

    if include_rf:
        alphas = np.linspace(0.0, leverage_cap, 80)
        cml_r, cml_v = [], []
        for a in alphas:
            rr, vv, _, _, _ = cml_metrics(float(a), w_tan, mu, cov, rf)
            cml_r.append(rr)
            cml_v.append(vv)
        fig.add_trace(go.Scatter(x=cml_v, y=cml_r, mode="lines", name="CML (RF + Tangency)"))

    fig.update_layout(
        height=420,
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis_title="Annualized Volatility",
        yaxis_title="Annualized Return",
    )
    st.plotly_chart(fig, use_container_width=True)

