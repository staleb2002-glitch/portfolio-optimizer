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
.kpi-sub { color: rgba(0,0,0,0.55); font-size: 0.8rem; margin: 0; }
.kpi-amount { font-size: 0.9rem; font-weight: 700; margin: 0; white-space: nowrap; }
.kpi-amount-label { color: rgba(0,0,0,0.55); font-size: 0.75rem; margin: 0; }
hr { border-color: rgba(0,0,0,0.1); }
/* Fit metrics in 8 columns */
[data-testid="stMetric"] label { font-size: 0.8rem !important; }
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
def download_close_prices(tickers, start, end):
    raw = yf.download(tickers, start=start, end=end, progress=False)
    if raw is None or len(raw) == 0:
        return pd.DataFrame()

    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" not in raw.columns.get_level_values(0):
            return pd.DataFrame()
        prices = raw["Close"]
    else:
        if "Close" not in raw.columns:
            return pd.DataFrame()
        prices = raw[["Close"]].rename(columns={"Close": tickers[0]})

    prices = prices.dropna(how="all")
    prices = prices.ffill().bfill()  # fill gaps: ffill for holidays, bfill for late-start tickers
    prices = prices.dropna(axis=1, how="any")  # drop tickers still missing data
    return prices


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


def solve_max_sharpe(mu, cov, rf, long_only=True, max_w=1.0):
    """
    Convex proxy: maximize excess return subject to risk == 1.
    Returns tangency weights over risky assets.
    """
    import cvxpy as cp
    n = len(mu)
    w = cp.Variable(n)
    excess = (mu.values - rf) @ w
    risk = cp.quad_form(w, cov.values)

    cons = [cp.sum(w) == 1, risk == 1.0]
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
    return wv / wv.sum()


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
    rf_daily = (1.0 + float(rf_annual)) ** (1.0 / TRADING_DAYS) - 1.0
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

    st.markdown('<div class="sidebar-section"><div class="section-label">Benchmark (Beta)</div></div>', unsafe_allow_html=True)
    benchmark = st.text_input("Market benchmark ticker", "SPY").strip().upper()

    st.markdown('<div class="sidebar-section"><div class="section-label">Risk-Free & Objective</div></div>', unsafe_allow_html=True)
    rf_pct = st.slider("Risk-Free rate (annual %) ", 0.0, 20.0, 2.0, 0.1)
    rf = float(rf_pct) / 100.0
    include_rf = st.checkbox("Include Risk-Free asset (CML)", value=True)
    leverage_cap = st.slider("Max risky exposure alpha (CML)", 0.0, 3.0, 2.0, 0.05)

    st.markdown('<div class="sidebar-section"><div class="section-label">Constraints</div></div>', unsafe_allow_html=True)
    long_only = st.checkbox("Long-only (no shorting)", value=True)
    max_w = st.slider("Max weight per asset", 0.10, 1.00, 1.00, 0.05)

    st.markdown('<div class="sidebar-section"><div class="section-label">Capital Allocation</div></div>', unsafe_allow_html=True)
    currency_choice = st.selectbox("Currency", ["EUR", "USD", "GBP", "CAD", "AUD", "JPY"], index=0)
    investment_amount = st.number_input("Investment amount ($)", min_value=0.0, value=10000.0, step=100.0, format="%.2f")

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
    prices = download_close_prices(tickers, str(start), str(end))
    if prices.empty or prices.shape[0] < 60:
        st.error("Not enough price data. Try different tickers or a wider date range.")
        st.stop()

    rets, mu, cov = annual_stats(prices)
    cum_returns = (1 + rets).cumprod() * 100

    cov_matrix = cov.copy()
    corr_matrix = rets.corr()

    # Betas vs benchmark
    betas = None
    capm_df = pd.DataFrame()
    bench_prices = download_close_prices([benchmark], str(start), str(end))
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
        # User-defined weights (risky assets + optional risk-free)
        total_pct = sum(custom_weights_input.values()) + custom_rf_weight
        w_raw = np.array([custom_weights_input.get(t, 0.0) for t in prices.columns])
        w_rf_raw = custom_rf_weight
        if total_pct > 0:
            scale = 1.0 / total_pct  # normalise percentages to fractions summing to 1
            w_custom = w_raw * scale
            w_rf = w_rf_raw * scale
        else:
            w_custom = np.ones(len(prices.columns)) / (len(prices.columns) + (1 if include_rf else 0))
            w_rf = 1.0 / (len(prices.columns) + 1) if include_rf else 0.0

        if include_rf and w_rf > 1e-9:
            # Blend: portfolio return = w_rf * rf + (1 - w_rf) * risky_return
            alpha = 1.0 - w_rf  # fraction in risky assets
            w_risky_normed = w_custom / w_custom.sum() if w_custom.sum() > 1e-12 else w_custom
            port_r_risky, port_v_risky = portfolio_metrics_risky(w_risky_normed, mu, cov)
            port_r = w_rf * rf + alpha * port_r_risky
            port_v = alpha * port_v_risky
            port_s = (port_r - rf) / port_v if port_v > 0 else np.nan
            selected_label = f"Custom Allocation (RF {w_rf:.0%})"
            weights_df = pd.DataFrame(
                {"Asset": ["RISK-FREE"] + list(prices.columns),
                 "Weight": [w_rf] + list(w_custom)}
            )
        else:
            if w_raw.sum() > 0:
                w_custom = w_raw / w_raw.sum()
            else:
                w_custom = np.ones(len(prices.columns)) / len(prices.columns)
            port_r, port_v = portfolio_metrics_risky(w_custom, mu, cov)
            port_s = (port_r - rf) / port_v if port_v > 0 else np.nan
            selected_label = "Custom Allocation"
            weights_df = pd.DataFrame({"Asset": list(prices.columns), "Weight": list(w_custom)})

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

    # Compute total (cumulative) period return from weighted daily returns
    w_risky_only = weights_df[weights_df["Asset"] != "RISK-FREE"].copy()
    port_daily_rets = pd.Series(0.0, index=rets.index)
    for _, row in w_risky_only.iterrows():
        asset = row["Asset"]
        wgt = float(row["Weight"])
        if asset in rets.columns:
            port_daily_rets = port_daily_rets + rets[asset] * wgt
    total_period_return = float((1 + port_daily_rets).prod() - 1)
    n_years = len(rets) / TRADING_DAYS
    total_return_cash = float(investment_amount) * total_period_return

    # ── Additional risk metrics ──
    # Max drawdown
    port_cum = (1 + port_daily_rets).cumprod()
    running_max = port_cum.cummax()
    drawdown_series = (port_cum - running_max) / running_max
    max_drawdown = float(drawdown_series.min()) if len(drawdown_series) > 0 else 0.0

    # Sortino Ratio (downside deviation)
    rf_daily = (1.0 + float(rf)) ** (1.0 / TRADING_DAYS) - 1.0
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
            # Portfolio-level Alpha, Beta, R²
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

st.markdown("---")

# =========================
# PDF Report Download
# =========================
def _build_report_data() -> dict:
    """Assemble report_data dict from current optimizer state."""
    # Weighted portfolio daily returns
    w_risky_only = weights_df[weights_df["Asset"] != "RISK-FREE"].copy()
    port_daily = pd.Series(0.0, index=rets.index)
    for _, row in w_risky_only.iterrows():
        asset = row["Asset"]
        wgt = float(row["Weight"])
        if asset in rets.columns:
            port_daily = port_daily + rets[asset] * wgt
    port_cum = (1 + port_daily).cumprod()

    # Performance: compute weighted period returns from cumulative series
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

    # Holdings table
    h_df = weights_df.copy()
    h_df.columns = ["Ticker", "Weight"]
    h_df["Market Value"] = h_df["Weight"] * float(investment_amount)
    h_df["P/L"] = h_df["Market Value"] * float(port_r)

    # Benchmark cumulative
    bench_cum = None
    try:
        if not bench_prices.empty:
            b_rets = bench_prices.pct_change().dropna().iloc[:, 0]
            common = port_daily.index.intersection(b_rets.index)
            if len(common) > 10:
                bench_cum = (1 + b_rets.loc[common]).cumprod() * 100
    except Exception:
        pass

    # Max drawdown
    running_max = port_cum.cummax()
    drawdown = (port_cum - running_max) / running_max
    max_dd = float(drawdown.min()) if len(drawdown) > 0 else 0.0

    # CML curve data for frontier chart
    _cml_v, _cml_r = None, None
    if include_rf:
        _alphas = np.linspace(0.0, leverage_cap, 80)
        _cml_v, _cml_r = [], []
        for _a in _alphas:
            _rr, _vv, _, _, _ = cml_metrics(float(_a), w_tan, mu, cov, rf)
            _cml_r.append(float(_rr))
            _cml_v.append(float(_vv))

    # Auto-generated commentary
    _commentary = [
        f"Strategy: {selected_label}. "
        f"Expected annual return {port_r:.2%} with {port_v:.2%} volatility (Sharpe {port_s:.2f}).",
        f"Assets analyzed over {start} – {end}. "
        f"Risk-free rate assumed at {rf:.2%} annually. Benchmark: {benchmark}.",
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
            "Max Drawdown": max_drawdown,
            "Tracking Error": float(tracking_error) if np.isfinite(tracking_error) else None,
            "Information Ratio": float(information_ratio) if np.isfinite(information_ratio) else None,
            "Alpha": float(port_alpha) if np.isfinite(port_alpha) else None,
            "Beta": float(port_beta) if np.isfinite(port_beta) else None,
            "R-Squared": float(port_r_squared) if np.isfinite(port_r_squared) else None,
        },
        "calendar_year_returns": cal_year_returns,
        "inception_date": str(start),
        "num_holdings": len([w for w in weights_df["Weight"] if abs(float(w)) > 0.001]),
        "weights": {
            row["Asset"]: float(row["Weight"])
            for _, row in weights_df.iterrows()
        },
        "holdings_df": h_df,
        "portfolio_series": port_cum_chart,
        "benchmark_series": bench_cum,
        "portfolio_label": selected_label,
        "benchmark_label": benchmark,
        "frontier": {
            "pv_cloud": pv_cloud.tolist() if hasattr(pv_cloud, 'tolist') else list(pv_cloud),
            "pr_cloud": pr_cloud.tolist() if hasattr(pr_cloud, 'tolist') else list(pr_cloud),
            "sharpe_cloud": sharpe_cloud.tolist() if hasattr(sharpe_cloud, 'tolist') else list(sharpe_cloud),
            "port_v": float(port_v),
            "port_r": float(port_r),
            "cml_v": _cml_v,
            "cml_r": _cml_r,
            "selected_label": selected_label,
        },
        "cum_returns": cum_returns,
        "corr_matrix": corr_matrix,
        "cov_matrix": cov_matrix,
        "betas": betas,
        "capm_df": capm_df if capm_df is not None and not capm_df.empty else None,
        "commentary": _commentary,
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

    st.write("**Selected Asset CAPM Scatter**")
    if capm_df is not None and not capm_df.empty and not bench_prices.empty:
        capm_table = capm_df.copy()
        capm_table["Alpha (annualized)"] = (1.0 + capm_table["Alpha (daily)"]) ** TRADING_DAYS - 1.0
        capm_table = capm_table[["Asset", "Alpha (annualized)", "Beta", "R^2", "Corr (excess vs mkt)"]]
        st.dataframe(capm_table, width='stretch', height=220)

        asset_choice = st.selectbox("Asset", list(rets.columns), key="capm_asset_select")

        rf_daily = (1.0 + float(rf)) ** (1.0 / TRADING_DAYS) - 1.0
        bench_rets = bench_prices.pct_change().dropna().iloc[:, 0]
        asset_rets = rets[asset_choice].dropna()

        common_idx = asset_rets.index.intersection(bench_rets.index)
        if len(common_idx) >= 60:
            a = (asset_rets.loc[common_idx] - rf_daily).values
            m = (bench_rets.loc[common_idx] - rf_daily).values

            x = m
            y = a
            x_mean = float(np.mean(x))
            y_mean = float(np.mean(y))
            denom = float(np.sum((x - x_mean) ** 2))
            beta = float(np.sum((x - x_mean) * (y - y_mean)) / denom) if denom > 1e-12 else 0.0
            alpha = float(y_mean - beta * x_mean)
            corr = float(np.corrcoef(x, y)[0, 1]) if len(x) > 1 else np.nan
            r2 = float(corr ** 2) if np.isfinite(corr) else np.nan

            x_line = np.linspace(np.min(x), np.max(x), 60)
            y_line = alpha + beta * x_line

            sel_fig = go.Figure()
            sel_fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode="markers",
                    marker=dict(size=7, color="rgba(31, 119, 180, 0.6)"),
                    name="Daily excess returns",
                    hovertemplate=
                        f"<b>{asset_choice}</b><br>Excess mkt: %{{x:.2%}}<br>Excess asset: %{{y:.2%}}<extra></extra>",
                )
            )
            sel_fig.add_trace(
                go.Scatter(
                    x=x_line,
                    y=y_line,
                    mode="lines",
                    line=dict(color="orange", width=2),
                    name="CAPM fit",
                    hovertemplate="CAPM fit<extra></extra>",
                )
            )
            sel_fig.update_layout(
                height=360,
                margin=dict(l=10, r=10, t=40, b=10),
                xaxis_title=f"{benchmark} Excess Return (daily)",
                yaxis_title=f"{asset_choice} Excess Return (daily)",
                title=f"{asset_choice}: α={alpha:.4%}, β={beta:.2f}, R²={r2:.2f}",
            )
            st.plotly_chart(sel_fig, use_container_width=True)
        else:
            st.info("Not enough overlapping data to plot selected asset CAPM scatter.")
    else:
        st.info("Select assets and benchmark with sufficient data to view CAPM scatter.")

    show_corr = st.checkbox("Show correlation (instead of covariance)", value=False)
    mat = corr_matrix if show_corr else cov_matrix
    title = "Correlation Heatmap" if show_corr else "Covariance Heatmap (annualized)"
    cbar = "Corr" if show_corr else "Cov"

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

