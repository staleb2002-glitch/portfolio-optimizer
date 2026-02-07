"""
Helper functions for optional OpenAI chat integration.
This file is loaded by app.py when available.
"""
import os
import re
import json


def call_openai_assistant(user_msg: str, ctx: dict, api_key: str, model: str = "gpt-3.5-turbo"):
    try:
        from openai import OpenAI
    except Exception:
        return None, "OpenAI library not installed"

    if not api_key:
        return None, "No API key provided"

    client = OpenAI(api_key=api_key)

    # Build compact context summary
    sel = ctx.get('selected_label','')
    rf = ctx.get('rf', 0.0)
    port_r = ctx.get('port_r', None)
    port_v = ctx.get('port_v', None)
    port_s = ctx.get('port_s', None)
    weights = ctx.get('weights_df')
    top_weights = []
    if weights is not None:
        for _, r in weights.head(6).iterrows():
            top_weights.append({'asset': r['Asset'], 'weight': float(r['Weight'])})

    context_summary = json.dumps({
        'selected_label': sel,
        'rf': rf,
        'port_r': port_r,
        'port_v': port_v,
        'port_s': port_s,
        'top_weights': top_weights,
    }, default=str)

    system_prompt = (
        "You are a helpful portfolio assistant embedded in a web app. "
        "You can answer any question the user asks, both portfolio-related and general knowledge. "
        "If the user asks to set or cap their portfolio volatility/risk at any level, respond with a JSON object on a single line: "
        "{\"action\": \"target_vol\", \"target_vol\": 0.15} "
        "Use the exact percentage/value the user specifies (e.g., user says '20% risk' → use 0.20, '10%' → use 0.10, '25% volatility' → use 0.25). "
        "After the JSON (or if no action needed), provide a helpful answer. Keep responses concise but friendly."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Context: {context_summary}\nUser: {user_msg}"}
    ]

    try:
        resp = client.chat.completions.create(model=model, messages=messages, max_tokens=400, temperature=0.0)
        text = resp.choices[0].message.content.strip()
    except Exception as e:
        return None, f"OpenAI call failed: {e}"

    # Try to extract JSON action
    act = None
    m = re.search(r"(\{\s*\"action\".*?\})", text, re.S)
    if m:
        try:
            act_json = m.group(1)
            act = json.loads(act_json)
        except Exception:
            act = None

    return text, act


def generate_ai_commentary(ctx: dict, api_key: str, model: str = "gpt-4o") -> list:
    """
    Use OpenAI to generate a comprehensive portfolio analysis summary
    for inclusion in PDF/PPTX reports.

    Returns a list of paragraph strings.
    Falls back to a static summary if OpenAI is unavailable.
    """
    # Build a rich context for the AI
    sel = ctx.get("selected_label", "N/A")
    port_r = ctx.get("port_r", 0)
    port_v = ctx.get("port_v", 0)
    port_s = ctx.get("port_s", 0)
    rf = ctx.get("rf", 0)
    total_ret = ctx.get("total_period_return", None)
    n_years = ctx.get("n_years", None)
    invest = ctx.get("investment_amount", 0)
    currency = ctx.get("currency_choice", "USD")
    benchmark = ctx.get("benchmark", "SPY")

    weights_df = ctx.get("weights_df")
    holdings = []
    if weights_df is not None:
        for _, r in weights_df.iterrows():
            holdings.append({"asset": r["Asset"], "weight": float(r["Weight"])})

    betas = ctx.get("betas")
    beta_info = {}
    if betas is not None:
        for asset, b in betas.items():
            beta_info[asset] = round(float(b), 3)

    corr_matrix = ctx.get("corr_matrix")
    avg_corr = None
    if corr_matrix is not None and hasattr(corr_matrix, "values"):
        import numpy as np
        m = corr_matrix.values.copy()
        np.fill_diagonal(m, np.nan)
        avg_corr = round(float(np.nanmean(m)), 3)

    risk = ctx.get("risk", {})

    data_for_ai = json.dumps({
        "strategy": sel,
        "annual_return": port_r,
        "annual_volatility": port_v,
        "sharpe_ratio": port_s,
        "risk_free_rate": rf,
        "total_period_return": total_ret,
        "period_years": n_years,
        "investment_amount": invest,
        "currency": currency,
        "benchmark": benchmark,
        "holdings": holdings,
        "betas_vs_benchmark": beta_info,
        "avg_pairwise_correlation": avg_corr,
        "max_drawdown": risk.get("Max Drawdown"),
    }, default=str, indent=2)

    system_prompt = (
        "You are a senior portfolio analyst writing a professional investment report commentary. "
        "Write a comprehensive yet concise analysis (4-6 paragraphs) covering:\n"
        "1. Portfolio strategy overview and rationale\n"
        "2. Return analysis: annual expected return, total period return, comparison to benchmark\n"
        "3. Risk analysis: volatility, Sharpe ratio, max drawdown, interpretation\n"
        "4. Diversification assessment: correlation structure, beta exposure, geographic/asset class diversification\n"
        "5. Key holdings and their contribution\n"
        "6. Forward-looking considerations and recommendations\n\n"
        "Use professional financial language. Be specific with numbers. "
        "Do not use markdown formatting (no **, ##, etc) — output plain text paragraphs only. "
        "Each paragraph should be separated by a newline."
    )

    # Try OpenAI
    if api_key:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Portfolio data:\n{data_for_ai}"},
                ],
                max_tokens=1200,
                temperature=0.3,
            )
            text = resp.choices[0].message.content.strip()
            # Split into paragraphs
            paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
            if paragraphs:
                return paragraphs
        except Exception:
            pass  # fall through to static

    # Fallback: build a static comprehensive summary
    paras = []
    paras.append(
        f"Strategy: {sel}. This portfolio targets an expected annual return of "
        f"{port_r:.2%} with {port_v:.2%} annualized volatility, yielding a "
        f"Sharpe ratio of {port_s:.2f} against a {rf:.2%} risk-free rate."
    )

    if total_ret is not None and n_years is not None:
        paras.append(
            f"Over the analyzed period of approximately {n_years:.1f} years, the "
            f"portfolio delivered a total cumulative return of {total_ret:.2%}. "
            f"On an investment of {invest:,.0f} {currency}, this corresponds to "
            f"a total profit/loss of {invest * total_ret:+,.0f} {currency}."
        )

    dd = risk.get("Max Drawdown")
    if dd is not None:
        paras.append(
            f"The maximum drawdown observed was {dd:.2%}, indicating the largest "
            f"peak-to-trough decline during the period. A Sharpe ratio of {port_s:.2f} "
            f"suggests {'strong' if port_s > 1 else 'moderate' if port_s > 0.5 else 'weak'} "
            f"risk-adjusted performance relative to the risk-free rate."
        )

    if avg_corr is not None:
        paras.append(
            f"The average pairwise correlation across portfolio assets is {avg_corr:.2f}. "
            f"{'This relatively low correlation suggests good diversification benefits.' if avg_corr < 0.5 else 'The moderate-to-high correlation indicates concentrated factor exposure.'} "
            f"Benchmark: {benchmark}."
        )

    if holdings:
        top_3 = sorted(holdings, key=lambda x: x["weight"], reverse=True)[:3]
        names = ", ".join(f"{h['asset']} ({h['weight']:.1%})" for h in top_3)
        paras.append(
            f"The top holdings are {names}. These positions form the core of the "
            f"portfolio and drive the majority of both return and risk contributions."
        )

    paras.append(
        "This report is for informational purposes only and does not constitute "
        "investment advice. Past performance is not indicative of future results. "
        "Investors should consider their own objectives and risk tolerance."
    )

    return paras
