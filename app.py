import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go

TRADING_DAYS = 252

# ---------- Page / Style ----------
st.set_page_config(page_title="Portfolio Optimizer", layout="wide")

CSS = """
<style>
.block-container { padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1300px; }
h1, h2, h3 { letter-spacing: -0.02em; }
.section { margin-top: 0.4rem; margin-bottom: 0.7rem; }
.card {
  border: 1px solid rgba(255,255,255,0.10);
  border-radius: 18px;
  padding: 14px 16px;
  background: rgba(255,255,255,0.03);
}
.kpi { font-size: 1.15rem; font-weight: 700; margin: 0; }
.kpi-sub { color: rgba(255,255,255,0.65); font-size: 0.85rem; margin: 0; }
.small-muted { color: rgba(255,255,255,0.65); font-size: 0.9rem; }
hr { border-color: rgba(255,255,255,0.08); }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ---------- Data ----------
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
    prices = prices.dropna(axis=1, how="any")  # keep tickers with full history
    return prices


def annual_stats(prices):
    rets = prices.pct_change().dropna()
    mu = rets.mean() * TRADING_DAYS
    cov = rets.cov() * TRADING_DAYS
    return rets, mu, cov


# ---------- Portfolio math ----------
def random_portfolios(mu, cov, n, max_w, seed=42):
    rng = np.random.default_rng(seed)
    k = len(mu)

    W = rng.random((n, k))
    W = W / W.sum(axis=1, keepdims=True)

    if max_w < 1.0:
        mask = (W <= max_w).all(axis=1)
        W = W[mask]
        if W.shape[0] < max(2000, n // 25):
            W = rng.random((n, k))
            W = W / W.sum(axis=1, keepdims=True)

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


# ---------- cvxpy (optional) ----------
def solve_min_variance(mu, cov, long_only=True, max_w=1.0):
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
    Convex proxy: maximize excess return subject to risk <= 1.
    Returns tangency weights over risky assets.
    """
    import cvxpy as cp
    n = len(mu)
    w = cp.Variable(n)
    excess = (mu.values - rf) @ w
    risk = cp.quad_form(w, cov.values)

    cons = [cp.sum(w) == 1, risk <= 1]
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


# ---------- UI ----------
st.title("📈 Portfolio Optimizer (One Page)")
st.write(
    '<span class="small-muted">Compare cumulative returns, optimize risky portfolios, and (optionally) mix with a Risk-Free asset along the Capital Market Line.</span>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Assets")
    tickers_text = st.text_input("Tickers (comma-separated)", "AAPL,MSFT,GOOGL,AMZN")
    tickers = [t.strip().upper() for t in tickers_text.split(",") if t.strip()]

    st.header("Period")
    start = st.date_input("Start", value=pd.to_datetime("2020-01-01"))
    end = st.date_input("End", value=pd.to_datetime("today"))

    st.header("Risk-Free / Objective")
    rf = st.number_input("Risk-Free rate (annual)", min_value=0.0, max_value=0.20, value=0.02, step=0.005)
    include_rf = st.checkbox("Include Risk-Free asset (CML)", value=True)

    if include_rf:
        cml_mode = st.selectbox("Mode", ["Max Sharpe (CML)", "Target Volatility (CML)", "Custom Risk Exposure (CML)"])
        leverage_cap = st.slider("Max risky exposure alpha", 0.0, 3.0, 2.0, 0.05)
        if cml_mode == "Target Volatility (CML)":
            target_vol_cml = st.slider("Target Volatility (annual)", 0.01, 1.00, 0.25, 0.01)
        elif cml_mode == "Custom Risk Exposure (CML)":
            alpha_custom = st.slider("Alpha (risky exposure)", 0.0, 3.0, 1.0, 0.05)
    else:
        objective = st.selectbox("Objective (risky-only)", ["Max Sharpe", "Min Variance", "Target Volatility"])
        target_vol = None
        if objective == "Target Volatility":
            target_vol = st.slider("Target Volatility (annual)", 0.05, 0.80, 0.25, 0.01)

    st.header("Constraints")
    long_only = st.checkbox("Long-only (no shorting)", value=True)
    max_w = st.slider("Max weight per asset", 0.10, 1.00, 1.00, 0.05)

    st.header("Engine")
    use_cvxpy = st.checkbox("Use true optimizer (cvxpy)", value=True)
    n_sims = st.slider("Simulation count (frontier cloud)", 2000, 80000, 25000, 2500)

if len(tickers) < 2:
    st.warning("Add at least **2 tickers**.")
    st.stop()

try:
    prices = download_close_prices(tickers, str(start), str(end))
    if prices.empty or prices.shape[0] < 60:
        st.error("Not enough price data. Try different tickers or a wider date range.")
        st.stop()

    rets, mu, cov = annual_stats(prices)

    # Returns comparison chart (base 100)
    cum_returns = (1 + rets).cumprod() * 100

    # Cloud (always)
    W_cloud, pr_cloud, pv_cloud = random_portfolios(mu, cov, n_sims, max_w=max_w)
    sharpe_cloud = (pr_cloud - rf) / pv_cloud

    # Tangency (risky) + Min Var (risky)
    # Tangency
    if use_cvxpy:
        try:
            w_tan = solve_max_sharpe(mu, cov, rf, long_only=long_only, max_w=max_w)
        except Exception as e:
            st.warning(f"cvxpy failed for Max Sharpe → using simulation. Reason: {e}")
            w_tan = W_cloud[int(np.nanargmax(sharpe_cloud))]
    else:
        w_tan = W_cloud[int(np.nanargmax(sharpe_cloud))]

    r_tan, v_tan = portfolio_metrics_risky(w_tan, mu, cov)

    # Min variance (for reference and warnings)
    if use_cvxpy:
        try:
            w_min = solve_min_variance(mu, cov, long_only=long_only, max_w=max_w)
        except Exception:
            w_min = W_cloud[int(np.nanargmin(pv_cloud))]
    else:
        w_min = W_cloud[int(np.nanargmin(pv_cloud))]

    r_min, v_min = portfolio_metrics_risky(w_min, mu, cov)

    # ---- Selected portfolio ----
    selected_label = ""
    w_rf = 0.0
    w_risky = None

    if include_rf:
        # Choose alpha
        if cml_mode == "Max Sharpe (CML)":
            alpha = 1.0
            selected_label = "CML @ alpha=1 (RF + Tangency)"
        elif cml_mode == "Target Volatility (CML)":
            alpha = 0.0 if v_tan <= 1e-12 else float(target_vol_cml / v_tan)
            alpha = max(0.0, min(alpha, leverage_cap))
            selected_label = f"CML Target Vol @ alpha={alpha:.2f}"
            if target_vol_cml < v_min:
                st.info(
                    f"Target vol ({target_vol_cml:.2%}) is below min risky-only vol ({v_min:.2%}). "
                    "This is achievable thanks to Risk-Free mixing (CML)."
                )
        else:
            alpha = float(alpha_custom)
            alpha = max(0.0, min(alpha, leverage_cap))
            selected_label = f"CML Custom @ alpha={alpha:.2f}"

        w_rf = 1.0 - alpha
        w_risky = alpha * w_tan

        port_r, port_v, port_s, _, _ = cml_metrics(alpha, w_tan, mu, cov, rf)

        # weights table includes RF
        weights_df = pd.DataFrame(
            {"Asset": ["RISK-FREE"] + list(prices.columns),
             "Weight": [w_rf] + list(w_risky)}
        )

    else:
        # risky-only selection
        if use_cvxpy:
            try:
                if objective == "Min Variance":
                    w_risky = w_min
                    selected_label = "Min Variance (risky-only)"
                elif objective == "Target Volatility":
                    if target_vol <= v_min + 1e-4:
                        st.warning(
                            f"Target vol ({target_vol:.2%}) is below/equal to min risky-only vol ({v_min:.2%}). "
                            "Result matches Min Variance."
                        )
                        w_risky = w_min
                        selected_label = "Target Vol -> Min Variance (risky-only)"
                    else:
                        w_risky = solve_target_vol_max_return(mu, cov, target_vol, long_only=long_only, max_w=max_w)
                        selected_label = "Target Volatility (risky-only)"
                else:
                    w_risky = w_tan
                    selected_label = "Max Sharpe (risky-only)"
            except Exception as e:
                st.warning(f"cvxpy failed → using simulation. Reason: {e}")
                # fallback: simulation pick
                if objective == "Max Sharpe":
                    w_risky = W_cloud[int(np.nanargmax(sharpe_cloud))]
                    selected_label = "Max Sharpe (simulation)"
                elif objective == "Min Variance":
                    w_risky = W_cloud[int(np.nanargmin(pv_cloud))]
                    selected_label = "Min Variance (simulation)"
                else:
                    idx = int(np.nanargmin(np.abs(pv_cloud - target_vol)))
                    w_risky = W_cloud[idx]
                    selected_label = "Target Volatility (simulation)"
        else:
            if objective == "Max Sharpe":
                w_risky = W_cloud[int(np.nanargmax(sharpe_cloud))]
            elif objective == "Min Variance":
                w_risky = W_cloud[int(np.nanargmin(pv_cloud))]
            else:
                idx = int(np.nanargmin(np.abs(pv_cloud - target_vol)))
                w_risky = W_cloud[idx]
            selected_label = f"{objective} (simulation)"

        port_r, port_v = portfolio_metrics_risky(w_risky, mu, cov)
        port_s = (port_r - rf) / port_v if port_v > 0 else np.nan

        weights_df = pd.DataFrame({"Asset": list(prices.columns), "Weight": list(w_risky)})

    # Clean weights table
    weights_df["Weight"] = weights_df["Weight"].astype(float)
    weights_df = weights_df.sort_values("Weight", ascending=False).reset_index(drop=True)

    # ---------- TOP KPIs ----------
    k1, k2, k3, k4 = st.columns(4)
    k1.markdown(f'<div class="card"><p class="kpi">{port_r:.2%}</p><p class="kpi-sub">Expected Return (annual)</p></div>', unsafe_allow_html=True)
    k2.markdown(f'<div class="card"><p class="kpi">{port_v:.2%}</p><p class="kpi-sub">Volatility (annual)</p></div>', unsafe_allow_html=True)
    k3.markdown(f'<div class="card"><p class="kpi">{port_s:.2f}</p><p class="kpi-sub">Sharpe (vs rf)</p></div>', unsafe_allow_html=True)
    k4.markdown(f'<div class="card"><p class="kpi">{selected_label}</p><p class="kpi-sub">Selected Portfolio</p></div>', unsafe_allow_html=True)

    st.markdown("---")

    # ---------- ONE PAGE LAYOUT ----------
    left, right = st.columns([0.42, 0.58], gap="large")

    with left:
        st.subheader("Weights")
        st.dataframe(weights_df, use_container_width=True, height=260)

        fig_w = go.Figure()
        fig_w.add_trace(go.Bar(x=weights_df["Asset"], y=weights_df["Weight"]))
        fig_w.update_layout(
            height=320,
            margin=dict(l=10, r=10, t=40, b=10),
            yaxis_title="Weight",
            title="Allocation",
        )
        st.plotly_chart(fig_w, use_container_width=True)

        st.subheader("Reference (risky-only)")
        rcol1, rcol2, rcol3 = st.columns(3)
        rcol1.metric("Tangency Return", f"{r_tan:.2%}")
        rcol2.metric("Tangency Vol", f"{v_tan:.2%}")
        rcol3.metric("Min Var Vol", f"{v_min:.2%}")

    with right:
        st.subheader("Cumulative Returns (Base = 100)")
        fig_ret = go.Figure()
        for t in cum_returns.columns:
            fig_ret.add_trace(go.Scatter(x=cum_returns.index, y=cum_returns[t], mode="lines", name=t))
        fig_ret.update_layout(
            height=360,
            margin=dict(l=10, r=10, t=40, b=10),
            yaxis_title="Growth of 100",
        )
        st.plotly_chart(fig_ret, use_container_width=True)

        st.subheader("Frontier (risky-only) + CML (if enabled)")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=pv_cloud, y=pr_cloud, mode="markers", marker=dict(size=4), name="RISKY portfolios"))

        # cloud markers
        i_min_cloud = int(np.nanargmin(pv_cloud))
        i_sh_cloud = int(np.nanargmax(sharpe_cloud))
        fig.add_trace(go.Scatter(x=[pv_cloud[i_min_cloud]], y=[pr_cloud[i_min_cloud]],
                                 mode="markers", marker=dict(size=10, symbol="x"), name="Min Var (cloud)"))
        fig.add_trace(go.Scatter(x=[pv_cloud[i_sh_cloud]], y=[pr_cloud[i_sh_cloud]],
                                 mode="markers", marker=dict(size=10, symbol="star"), name="Max Sharpe (cloud)"))

        # Selected point
        fig.add_trace(go.Scatter(x=[port_v], y=[port_r],
                                 mode="markers", marker=dict(size=14, symbol="diamond"), name="Selected"))

        # CML line if RF enabled
        if include_rf:
            alphas = np.linspace(0.0, leverage_cap, 80)
            cml_r = []
            cml_v = []
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

except Exception as e:
    st.error("The app crashed while loading. Here is the exact error:")
    st.exception(e)
    st.stop()
