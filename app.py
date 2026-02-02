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
.kpi-amount { font-size: 0.95rem; font-weight: 700; margin: 0; white-space: nowrap; }
.kpi-amount-label { color: rgba(0,0,0,0.55); font-size: 0.8rem; margin: 0; }
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
        return f"✅ Done. I’ll cap your risk at **{target:.0%} volatility**. I’m recomputing the portfolio now…"

    # If ctx missing, refuse
    if not ctx or "weights_df" not in ctx:
        return "I don’t have portfolio results available yet. Please select assets/dates first."

    # Explanation intents
    if any(k in low for k in ["allocation", "weights", "explain my results", "summary", "explain results"]):
        wdf = ctx["weights_df"]
        top = wdf.head(6)
        w_lines = "\n".join([f"- {r['Asset']}: **{r['Weight']:.1%}**" for _, r in top.iterrows()])
        corr_text = explain_correlation(wdf, ctx.get("corr_matrix"), top_n=4)
        invest_amt = float(ctx.get("investment_amount") or 0.0)
        exp_dollars = ctx.get("expected_return_dollars")
        curr = ctx.get("currency_choice", "USD")
        dollars_line = ""
        if invest_amt > 0 and exp_dollars is not None:
            dollars_line = f"- Expected annual return on {invest_amt:,.2f} {curr}: **{float(exp_dollars):,.2f} {curr}**\n"

        return (
            f"**Portfolio summary**\n"
            f"- Selected: **{ctx.get('selected_label','')}**\n"
            f"- Expected return (annual): **{ctx.get('port_r'):.2%}**\n"
            f"{dollars_line}"
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
            "Sharpe measures risk-adjusted performance: **(Return − rf) / Volatility**.\n"
            f"Here: return={ctx.get('port_r'):.2%}, rf={ctx.get('rf'):.2%}, vol={ctx.get('port_v'):.2%} "
            f"→ Sharpe={ctx.get('port_s'):.2f}."
        )

    return (
        "I can help with portfolio results. Try:\n"
        "- “I want to be capped at 15% risk”\n"
        "- “Explain allocations and correlation”\n"
        "- “Explain betas”\n"
        "- “Am I diversified?”"
    )


# =========================
# UI (Inputs)
# =========================
st.title("📈 Portfolio Optimizer")
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
    rf_pct = st.slider("Risk-Free rate (annual %) ", 0.0, 20.0, 2.0, 0.1)
    rf = float(rf_pct) / 100.0
    include_rf = st.checkbox("Include Risk-Free asset (CML)", value=True)
    leverage_cap = st.slider("Max risky exposure alpha (CML)", 0.0, 3.0, 2.0, 0.05)

    st.markdown('<div class="sidebar-section"><div class="section-label">Constraints</div></div>', unsafe_allow_html=True)
    long_only = st.checkbox("Long-only (no shorting)", value=True)
    max_w = st.slider("Max weight per asset", 0.10, 1.00, 1.00, 0.05)

    st.markdown('<div class="sidebar-section"><div class="section-label">Capital Allocation</div></div>', unsafe_allow_html=True)
    currency_choice = st.selectbox("Currency", ["USD", "EUR", "GBP", "CAD", "AUD", "JPY"], index=0)
    investment_amount = st.number_input("Investment amount ($)", min_value=0.0, value=10000.0, step=100.0, format="%.2f")

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
    st.warning("⚠️ OPENAI_API_KEY not set. Set it as an environment variable to enable AI assistant.")

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
        "investment_amount": float(investment_amount),
        "expected_return_dollars": float(investment_amount) * float(port_r),
        "currency_choice": currency_choice,
        "betas": betas,
        "capm_df": capm_df,
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
k1, k2, k3, k4, k5, k6 = st.columns(6)
with k1:
    st.metric("Expected Return", f"{port_r:.2%}")
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
        f"<div class=\"kpi-amount-label\">Expected Return ({currency_choice})</div>",
        unsafe_allow_html=True,
    )
with k6:
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
    st.plotly_chart(fig_w, use_container_width=True)

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

# ======================================================================
# CHAT — Top-level (required by Streamlit) + actionable commands
# ======================================================================
st.markdown("---")
st.markdown(
    """
    <div class="chat-card">
        <div class="chat-title">💬 Portfolio Assistant</div>
        <div class="chat-hint">
            Try: “I want to be capped at 15% risk” → then “Explain allocations and correlation”.
            Out-of-scope (e.g., weather) will be refused safely.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# quick prompts
q1, q2, q3, q4 = st.columns(4)
if q1.button("Cap at 15% risk"):
    st.session_state.prefill = "I want to be capped at 15% risk"
if q2.button("Explain allocations"):
    st.session_state.prefill = "Explain allocations"
if q3.button("Explain correlation"):
    st.session_state.prefill = "Explain correlation"
if q4.button("Explain betas"):
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
