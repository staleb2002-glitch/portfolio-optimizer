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
