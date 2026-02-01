import os
import re
import json
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go

TRADING_DAYS = 252

# ---------------- Page / Style ----------------
st.set_page_config(page_title="Portfolio Optimizer", layout="wide")

CSS = """
<style>
.block-container { padding-top: 1.1rem; padding-bottom: 2rem; max-width: 1350px; }
.small-muted { color: rgba(0,0,0,0.55); font-size: 0.9rem; }
.kpi { font-size: 1.05rem; font-weight: 700; margin: 0; }
.kpi-sub { color: rgba(0,0,0,0.55); font-size: 0.85rem; margin: 0; }
hr { border-color: rgba(0,0,0,0.1); }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# Header
st.write(
        '<span class="small-muted">Cumulative returns · Frontier + CML · Cov/Cor matrices · Betas · Actionable chat</span>',
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
    prices = prices.dropna(axis=1, how="any")  # keep complete history only
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


def capm_regression_analysis(asset_returns: pd.Series, benchmark_returns: pd.Series, rf: float) -> dict:
    """
    Perform CAPM regression analysis for a single asset.
    Returns: {alpha, beta, r_squared, asset_excess_returns, capm_predicted_returns}
    """
    # Excess returns
    asset_excess = asset_returns - rf
    market_excess = benchmark_returns - rf
    
    # OLS regression: asset_excess = alpha + beta * market_excess + epsilon
    X = market_excess.values.reshape(-1, 1)
    y = asset_excess.values
    
    # Add intercept
    X_with_const = np.column_stack([np.ones(len(X)), X])
    
    # Solve: beta_hat = (X'X)^-1 X'y
    XtX_inv = np.linalg.pinv(X_with_const.T @ X_with_const)
    beta_hat = XtX_inv @ X_with_const.T @ y
    
    alpha = beta_hat[0]
    beta = beta_hat[1]
    
    # Predictions
    y_pred = X_with_const @ beta_hat
    
    # R-squared
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    return {
        'alpha': alpha,
        'beta': beta,
        'r_squared': r_squared,
        'asset_excess_returns': asset_excess.values,
        'capm_predicted_returns': market_excess.values * beta,
        'market_excess_returns': market_excess.values,
    }


# ---------------- Chat: parsing + safe assistant ----------------
OUT_OF_SCOPE_TERMS = [
    "weather", "temperature", "forecast", "rain", "snow",
    "news", "politics", "sports", "flight", "restaurant",
]

def parse_target_vol(user_msg: str):
    """
    Extract target risk/volatility from phrases like:
      - "capped at 15% risk"
      - "target vol 12%"
      - "volatility 0.20"
    Returns decimal (0.15) or None.
    """
    msg = user_msg.lower()

    # percent form
    m = re.search(r"(\d+(?:\.\d+)?)\s*%\s*(risk|vol|volatility)", msg)
    if not m:
        m = re.search(r"(risk|vol|volatility)\s*(cap|capped|target)?\s*at?\s*(\d+(?:\.\d+)?)\s*%", msg)
        if m:
            val = float(m.group(3)) / 100.0
            return val if 0.01 <= val <= 2.0 else None
        return None

    val = float(m.group(1)) / 100.0
    return val if 0.01 <= val <= 2.0 else None


def explain_correlation(weights_df: pd.DataFrame, corr_matrix: pd.DataFrame, top_n: int = 4) -> str:
    if corr_matrix is None or corr_matrix.empty:
        return "Correlation matrix is not available."

    assets = [a for a in weights_df["Asset"].tolist() if a != "RISK-FREE"]
    if len(assets) < 2:
        return "Not enough risky assets to discuss correlation."

    w = weights_df[weights_df["Asset"] != "RISK-FREE"].copy()
    w = w.sort_values("Weight", ascending=False).head(top_n)
    top_assets = w["Asset"].tolist()

    pairs = []
    for i in range(len(top_assets)):
        for j in range(i + 1, len(top_assets)):
            a, b = top_assets[i], top_assets[j]
            if a in corr_matrix.index and b in corr_matrix.columns:
                pairs.append((a, b, float(corr_matrix.loc[a, b])))

    pairs.sort(key=lambda x: abs(x[2]), reverse=True)

    m = corr_matrix.copy()
    np.fill_diagonal(m.values, np.nan)
    avg_corr = float(np.nanmean(m.values))

    lines = []
    for a, b, c in pairs[:5]:
        lines.append(f"- Corr({a}, {b}) = **{c:.2f}**")

    return (
        f"**Diversification / Correlation**\n"
        f"- Average pairwise correlation ≈ **{avg_corr:.2f}** (lower usually means better diversification)\n"
        f"- Strongest relationships among top weights:\n" + ("\n".join(lines) if lines else "- (not enough data)")
    )


def safe_assistant(user_msg: str, ctx: dict) -> str:
    """
    Non-hallucinating assistant:
      - If user asks out-of-scope (e.g. weather), refuses safely.
      - If user asks for risk cap/target vol, it triggers an app action.
      - Otherwise answers only from ctx (weights/corr/betas/metrics).
    """
    msg = user_msg.strip()
    low = msg.lower()

    # Out of scope guard
    if any(t in low for t in OUT_OF_SCOPE_TERMS):
        return (
            "I can’t answer that reliably from this portfolio app (no external data source connected).\n\n"
            "I *can* help with:\n"
            "- Target risk/volatility portfolios (e.g., “cap me at 15% risk”)\n"
            "- Explaining weights, Sharpe, betas, covariance/correlation, diversification\n"
            "- Interpreting the frontier and Risk-Free (CML) mix"
        )

    # Action: target volatility
    target = parse_target_vol(msg)
    if target is not None:
        st.session_state.ai_target_vol = float(target)
        st.session_state.ai_action = "target_vol"
        return f"Done. I'll cap your risk at **{target:.0%} volatility**. I'm recomputing the portfolio now..."

    # If ctx missing, refuse
    if not ctx or "weights_df" not in ctx:
        return "I don’t have portfolio results available yet. Please select assets/dates first."
    # Comprehensive "explain" - summarize everything
    if low == "explain" or low == "explain.":
        wdf = ctx["weights_df"]
        top = wdf.head(10)
        w_lines = "\n".join([f"- {r['Asset']}: **{r['Weight']:.1%}**" for _, r in top.iterrows()])
        
        # Get correlation insights
        corr_text = explain_correlation(wdf, ctx.get("corr_matrix"), top_n=4)
        
        # Get beta insights
        betas = ctx.get("betas")
        bench = ctx.get("benchmark", "SPY")
        beta_text = ""
        if betas is not None:
            s = betas.sort_values(ascending=False)
            beta_text = (
                f"\n\n**Market Sensitivity (Betas vs {bench})**\n"
                f"- Highest: **{s.index[0]}** (β={s.iloc[0]:.2f})\n"
                f"- Lowest: **{s.index[-1]}** (β={s.iloc[-1]:.2f})\n"
                f"- β>1 = more volatile than market; β<1 = more defensive"
            )
        
        # Sharpe explanation
        sharpe_text = (
            f"\n\n**Risk-Adjusted Performance**\n"
            f"- Sharpe Ratio: **{ctx.get('port_s'):.2f}** = (Return - Risk-Free) / Volatility\n"
            f"- This measures how much excess return you get per unit of risk"
        )
        
        return (
            f"# Complete Portfolio Summary\n\n"
            f"**Optimization Method**: {ctx.get('selected_label','')}\n"
            f"- Expected Annual Return: **{ctx.get('port_r'):.2%}**\n"
            f"- Annual Volatility (Risk): **{ctx.get('port_v'):.2%}**\n"
            f"- Sharpe Ratio: **{ctx.get('port_s'):.2f}**\n"
            f"- Risk-Free Rate: **{ctx.get('rf'):.2%}**\n\n"
            f"**Portfolio Allocations** (Top Holdings)\n{w_lines}\n\n"
            f"{corr_text}{beta_text}{sharpe_text}\n\n"
            f"**Key Insights**\n"
            f"- Your portfolio is optimized for {ctx.get('selected_label','').lower()}\n"
            f"- Total assets: {len(wdf[wdf['Asset'] != 'RISK-FREE'])} risky assets\n"
            f"- {'Includes risk-free asset allocation' if any(wdf['Asset'] == 'RISK-FREE') else 'Fully invested in risky assets'}"
        )
    # Explanation intents
    if any(k in low for k in ["allocation", "weights", "explain my results", "summary", "explain results"]):
        wdf = ctx["weights_df"]
        top = wdf.head(6)
        w_lines = "\n".join([f"- {r['Asset']}: **{r['Weight']:.1%}**" for _, r in top.iterrows()])
        corr_text = explain_correlation(wdf, ctx.get("corr_matrix"), top_n=4)

        return (
            f"**Portfolio summary**\n"
            f"- Selected: **{ctx.get('selected_label','')}**\n"
            f"- Expected return (annual): **{ctx.get('port_r'):.2%}**\n"
            f"- Volatility (annual): **{ctx.get('port_v'):.2%}**\n"
            f"- Sharpe (vs rf): **{ctx.get('port_s'):.2f}**\n\n"
            f"**Top allocations**\n{w_lines}\n\n{corr_text}"
        )

    if "beta" in low:
        betas = ctx.get("betas")
        bench = ctx.get("benchmark", "SPY")
        if betas is None:
            return f"Betas are not available (benchmark {bench} failed to download or overlap was too short)."
        s = betas.sort_values(ascending=False)
        return (
            f"**Betas vs {bench}** (market sensitivity)\n"
            f"- Highest beta: **{s.index[0]}** (β={s.iloc[0]:.2f})\n"
            f"- Lowest beta: **{s.index[-1]}** (β={s.iloc[-1]:.2f})\n"
            "β>1 tends to move more than the market; β<1 is more defensive."
        )

    if any(k in low for k in ["correlation", "covariance", "diversified", "diversification"]):
        return explain_correlation(ctx["weights_df"], ctx.get("corr_matrix"), top_n=4)

    if "sharpe" in low:
        return (
            "Sharpe measures risk-adjusted performance: **(Return - rf) / Volatility**.\n"
            f"Here: return={ctx.get('port_r'):.2%}, rf={ctx.get('rf'):.2%}, vol={ctx.get('port_v'):.2%} "
            f"=> Sharpe={ctx.get('port_s'):.2f}."
        )


    return (
        "I can help with portfolio results. Try:\n"
        "- 'explain' for complete portfolio summary\n"
        "- 'I want to be capped at 15% risk'\n"
        "- 'Explain allocations and correlation'\n"
        "- 'Explain betas'\n"
        "- 'Am I diversified?'"
    )


# =========================
# UI (Inputs)
# =========================
st.title("Portfolio Optimizer")
st.write(
    '<span class="small-muted">Cumulative returns · Frontier + CML · Cov/Cor matrices · Betas · Actionable chat</span>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown('<div class="sidebar-section"><div class="section-label">Assets</div></div>', unsafe_allow_html=True)
    tickers_text = st.text_input("Tickers (comma-separated)", "AAPL,MSFT,GOOGL,AMZN")
    tickers = [t.strip().upper() for t in tickers_text.split(",") if t.strip()]

    st.markdown('<div class="sidebar-section"><div class="section-label">Period</div></div>', unsafe_allow_html=True)
    start = st.date_input("Start", value=pd.to_datetime("2020-01-01"))
    end = st.date_input("End", value=pd.to_datetime("today"))

    st.markdown('<div class="sidebar-section"><div class="section-label">Benchmark (Beta)</div></div>', unsafe_allow_html=True)
    benchmark = st.text_input("Market benchmark ticker", "SPY").strip().upper()

    st.markdown('<div class="sidebar-section"><div class="section-label">Risk-Free & Objective</div></div>', unsafe_allow_html=True)
    rf_pct = st.slider("Risk-Free rate (annual %)", 0.0, 20.0, 2.0, 0.5)
    rf = rf_pct / 100.0
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
        ["Max Sharpe", "Minimum Variance", "Target Volatility"]
    )
    target_vol_input = None
    if portfolio_strategy == "Target Volatility":
        # slider inputs shown as percent in UI; keep decimal used elsewhere
        tv_pct = st.slider("Target Volatility (annual %)", 1, 40, 15)
        target_vol_input = float(tv_pct) / 100.0
    
    # ---------------- Figma section ----------------
    # (no additional sidebar integrations)

# Get OpenAI API key from environment
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')
if not OPENAI_API_KEY:
    st.warning("OPENAI_API_KEY not set. Set it as an environment variable to enable AI assistant.")

# Guard
if len(tickers) < 2:
    st.warning("Add at least **2 tickers**.")
    st.stop()

# =========================
# Compute results
# =========================
try:
    # Include benchmark in the download if it's not already there
    tickers_with_bench = list(tickers)
    if benchmark and benchmark not in tickers_with_bench:
        tickers_with_bench.append(benchmark)
    
    prices = download_close_prices(tickers_with_bench, str(start), str(end))
    if prices.empty or prices.shape[0] < 60:
        st.error("Not enough price data. Try different tickers or a wider date range.")
        st.stop()

    rets, mu, cov = annual_stats(prices)
    cum_returns = (1 + rets).cumprod() * 100

    cov_matrix = cov.copy()
    corr_matrix = rets.corr()

    # Betas vs benchmark
    betas = None
    if benchmark and benchmark in prices.columns:
        bench_rets = prices[benchmark].pct_change().dropna()
        common_idx = rets.index.intersection(bench_rets.index)
        if len(common_idx) >= 60:
            betas = compute_betas(rets.loc[common_idx], bench_rets.loc[common_idx])
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
        st.info("cvxpy not installed - target-vol risky-only uses simulation. (pip install cvxpy)")

    # Tangency portfolio (risky-only)
    if use_cvxpy and cvxpy_ok:
        try:
            w_tan = solve_max_sharpe(mu, cov, rf, long_only=long_only, max_w=max_w)
        except Exception:
            w_tan = W_cloud[int(np.nanargmax(sharpe_cloud))]
    else:
        w_tan = W_cloud[int(np.nanargmax(sharpe_cloud))]

    r_tan, v_tan = portfolio_metrics_risky(w_tan, mu, cov)

    # --------- ACTION OVERRIDE from chat (target volatility) ----------
    ai_action = st.session_state.get("ai_action")
    ai_target_vol = st.session_state.get("ai_target_vol")

    # Determine portfolio strategy (AI action overrides, then user selection, then defaults)
    active_strategy = portfolio_strategy
    active_target_vol = target_vol_input
    
    if ai_action == "target_vol" and ai_target_vol is not None:
        active_strategy = "Target Volatility"
        active_target_vol = float(ai_target_vol)
    
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

    # Save context for chat responses
    st.session_state.app_context = {
        "selected_label": selected_label,
        "rf": rf,
        "port_r": port_r,
        "port_v": port_v,
        "port_s": port_s,
        "weights_df": weights_df,
        "betas": betas,
        "benchmark": benchmark,
        "cov_matrix": cov_matrix,
        "corr_matrix": corr_matrix,
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
k1, k2, k3, k4, k5 = st.columns(5)
with k1:
    st.metric("Expected Return", f"{port_r:.2%}")
with k2:
    st.metric("Volatility", f"{port_v:.2%}")
with k3:
    st.metric("Your Sharpe", f"{port_s:.2f}")
with k4:
    st.metric("Max Sharpe", f"{max_sharpe_cloud:.2f}")
with k5:
    st.info(f"📋 {selected_label}")

st.markdown("---")

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
    st.plotly_chart(fig_w, config={'responsive': True})

    st.subheader("Asset Performance & CAPM Analysis")
    
    # Build asset performance table
    risky_assets = [a for a in tickers if a != "RISK-FREE"]
    if len(risky_assets) > 0 and betas is not None:
        asset_perf_data = []
        for ticker in risky_assets:
            total_ret = ((prices[ticker].iloc[-1] / prices[ticker].iloc[0]) - 1)
            ann_ret = (1 + total_ret) ** (TRADING_DAYS / len(prices)) - 1
            asset_perf_data.append({
                'Asset': ticker,
                'Period Return': f"{total_ret:.2%}",
                'Annualized Return': f"{ann_ret:.2%}",
                'Beta': f"{betas[ticker]:.3f}" if ticker in betas.index else "N/A"
            })
        
        perf_df = pd.DataFrame(asset_perf_data)
        st.dataframe(perf_df, width='stretch', height=200)
        
        # Interactive CAPM analysis selector
        st.write("**Click an asset to see CAPM Regression Analysis:**")
        selected_asset = st.selectbox("Select asset for detailed analysis:", risky_assets, key="capm_select")
        
        if selected_asset and selected_asset in prices.columns and selected_asset in rets.columns:
            if benchmark and benchmark in prices.columns:
                try:
                    # Get benchmark returns (daily, not annualized)
                    bench_prices = prices[benchmark]
                    bench_rets_daily = bench_prices.pct_change().dropna()
                    
                    # Get asset returns (daily)
                    asset_rets_daily = rets[selected_asset]
                    
                    # Align returns by date
                    common_idx = asset_rets_daily.index.intersection(bench_rets_daily.index)
                    
                    if len(common_idx) > 20:
                        asset_rets_aligned = asset_rets_daily[common_idx].values
                        bench_rets_aligned = bench_rets_daily[common_idx].values
                        
                        # Calculate excess returns
                        rf_daily = rf / TRADING_DAYS
                        asset_excess = asset_rets_aligned - rf_daily
                        bench_excess = bench_rets_aligned - rf_daily
                        
                        # Simple linear regression
                        X = bench_excess.reshape(-1, 1)
                        y = asset_excess
                        X_with_const = np.column_stack([np.ones(len(X)), X])
                        XtX_inv = np.linalg.pinv(X_with_const.T @ X_with_const)
                        beta_hat = XtX_inv @ X_with_const.T @ y
                        
                        alpha_daily = beta_hat[0]
                        beta_val = beta_hat[1]
                        
                        # R-squared
                        y_pred = X_with_const @ beta_hat
                        ss_res = np.sum((y - y_pred) ** 2)
                        ss_tot = np.sum((y - y.mean()) ** 2)
                        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                        
                        # Display metrics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Alpha (annualized)", f"{alpha_daily*TRADING_DAYS:.2%}")
                        with col2:
                            st.metric("Beta", f"{beta_val:.3f}")
                        with col3:
                            st.metric("R-squared", f"{r_squared:.3f}")
                        
                        # Scatter plot
                        fig_capm = go.Figure()
                        fig_capm.add_trace(go.Scatter(
                            x=bench_excess,
                            y=asset_excess,
                            mode='markers',
                            marker=dict(size=5, opacity=0.5, color='steelblue'),
                            name='Actual Returns',
                            hovertemplate='Market Excess: %{x:.4f}<br>Asset Excess: %{y:.4f}<extra></extra>'
                        ))
                        
                        # Regression line
                        x_line = np.array([bench_excess.min(), bench_excess.max()])
                        y_line = alpha_daily + beta_val * x_line
                        fig_capm.add_trace(go.Scatter(
                            x=x_line,
                            y=y_line,
                            mode='lines',
                            line=dict(color='red', width=2),
                            name=f"CAPM: α={alpha_daily*TRADING_DAYS:.2%}, β={beta_val:.3f}",
                            hovertemplate='Regression Line<extra></extra>'
                        ))
                        
                        fig_capm.update_layout(
                            title=f"CAPM Regression: {selected_asset} vs {benchmark}",
                            xaxis_title=f"{benchmark} Excess Return (daily)",
                            yaxis_title=f"{selected_asset} Excess Return (daily)",
                            height=450,
                            margin=dict(l=10, r=10, t=40, b=10),
                            hovermode='closest'
                        )
                        st.plotly_chart(fig_capm)
                    else:
                        st.warning(f"Not enough overlapping data between {selected_asset} and {benchmark}")
                except Exception as e:
                    st.error(f"Error analyzing {selected_asset}: {str(e)}")
            else:
                st.warning(f"Benchmark {benchmark} not available")

    st.subheader("Betas & Matrices")
    
    show_corr = st.checkbox("Show correlation (instead of covariance)", value=False)
    mat = corr_matrix if show_corr else cov_matrix
    title = "Correlation Heatmap" if show_corr else "Covariance Heatmap (annualized)"
    cbar = "Corr" if show_corr else "Cov"

    st.dataframe(mat, width='stretch', height=250)
    fig_mat = go.Figure(data=go.Heatmap(z=mat.values, x=mat.columns, y=mat.index, colorscale="Reds", colorbar=dict(title=cbar)))
    fig_mat.update_layout(height=320, margin=dict(l=10, r=10, t=40, b=10), title=title)
    st.plotly_chart(fig_mat, config={'responsive': True})

with tab2:
    st.subheader("Cumulative Returns (Base = 100)")
    fig_ret = go.Figure()
    for t in cum_returns.columns:
        fig_ret.add_trace(go.Scatter(x=cum_returns.index, y=cum_returns[t], mode="lines", name=t))
    fig_ret.update_layout(height=360, margin=dict(l=10, r=10, t=40, b=10), yaxis_title="Growth of 100")
    st.plotly_chart(fig_ret, config={'responsive': True})

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
    st.plotly_chart(fig, config={'responsive': True})

# ======================================================================
# CHAT — Top-level (required by Streamlit) + actionable commands
# ======================================================================
st.markdown("---")
st.markdown(
    """
    <div class="chat-card">
        <div class="chat-title">Portfolio Assistant</div>
        <div class="chat-hint">
            Try: "explain" for complete summary • "I want to be capped at 15% risk" • "Explain allocations".
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# quick prompts
q1, q2, q3, q4, q5 = st.columns(5)
if q1.button("Explain"):
    st.session_state.prefill = "explain"
if q2.button("Cap at 15% risk"):
    st.session_state.prefill = "I want to be capped at 15% risk"
if q3.button("Explain allocations"):
    st.session_state.prefill = "Explain allocations"
if q4.button("Explain correlation"):
    st.session_state.prefill = "Explain correlation"
if q5.button("Explain betas"):
    st.session_state.prefill = "Explain betas"

if "chat" not in st.session_state:
    st.session_state.chat = []

# render history
for role, content in st.session_state.chat:
    with st.chat_message(role):
        st.markdown(content)

# top-level chat input
prefill = st.session_state.pop("prefill", "")
user_msg = st.chat_input("Ask about your portfolio… (e.g., capped at 15% risk)")

if user_msg:
    st.session_state.chat.append(("user", user_msg))
    with st.chat_message("user"):
        st.markdown(user_msg)

    ctx = st.session_state.get("app_context", {})

    # Try OpenAI assistant first (if key available), then fall back to local safe_assistant
    answer = None
    action = None
    if OPENAI_API_KEY:
        try:
            from app_ai_helpers import call_openai_assistant
            resp_text, act = call_openai_assistant(user_msg, ctx, OPENAI_API_KEY)
            if resp_text is None:
                # resp_text None indicates failure; fall back to safe_assistant
                pass
            else:
                answer = resp_text
                action = act
        except Exception:
            # Silently fall back on error
            pass

    if answer is None:
        answer = safe_assistant(user_msg, ctx)

    st.session_state.chat.append(("assistant", answer))
    with st.chat_message("assistant"):
        st.markdown(answer)

    # If assistant triggered an action, apply it and rerun
    if action and isinstance(action, dict):
        if action.get('action') == 'target_vol' and 'target_vol' in action:
            try:
                tv = float(action.get('target_vol'))
                st.session_state.ai_target_vol = float(tv)
                st.session_state.ai_action = 'target_vol'
                st.rerun()
            except Exception:
                pass
    else:
        # Also check free-text parsing from safe_assistant
        if parse_target_vol(answer) is not None:
            st.session_state.ai_target_vol = float(parse_target_vol(answer))
            st.session_state.ai_action = 'target_vol'
            st.rerun()

# Footer
# Footer removed — reverting to original layout
