"""
portfolio_factsheet_pdf.py
──────────────────────────
Professional fund-factsheet-style PDF (3 pages, Invesco-inspired layout).

Usage:
    from portfolio_factsheet_pdf import generate_factsheet
    generate_factsheet(report_data, "factsheet.pdf")

Dependencies: reportlab, matplotlib, pandas, numpy, Pillow
"""

from __future__ import annotations

import io
import math
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm, mm
from reportlab.pdfgen import canvas

# ──────────────────────────────────────────────
# Brand colours (Invesco-inspired dark-blue palette)
# ──────────────────────────────────────────────
NAVY         = colors.HexColor("#003366")
DARK_NAVY    = colors.HexColor("#001F3F")
ACCENT_BLUE  = colors.HexColor("#0055A4")
LIGHT_BLUE   = colors.HexColor("#E8F0FE")
MID_BLUE     = colors.HexColor("#4A90D9")
WHITE        = colors.white
BLACK        = colors.HexColor("#1A1A1A")
BODY_TEXT    = colors.HexColor("#333333")
MUTED_TEXT   = colors.HexColor("#666666")
LIGHT_GREY   = colors.HexColor("#F5F5F7")
BORDER_GREY  = colors.HexColor("#CCCCCC")
POSITIVE     = colors.HexColor("#2E7D32")
NEGATIVE     = colors.HexColor("#C62828")
HEADER_BG    = colors.HexColor("#E8ECF0")
ROW_ALT      = colors.HexColor("#F7F9FB")

CHART_COLORS = [
    "#003366", "#0055A4", "#4A90D9", "#E94560",
    "#FCA311", "#6A994E", "#9B5DE5", "#F15BB5",
    "#00B4D8", "#FB8500", "#219EBC", "#8338EC",
]

# Page geometry
PAGE_W, PAGE_H = A4  # 595.27 × 841.89 pt
MARGIN_L   = 40
MARGIN_R   = 40
MARGIN_T   = 50
MARGIN_B   = 45
CONTENT_W  = PAGE_W - MARGIN_L - MARGIN_R

# Font constants
FONT_SANS      = "Helvetica"
FONT_SANS_BOLD = "Helvetica-Bold"
FONT_SANS_OBL  = "Helvetica-Oblique"


# ──────────────────────────────────────────────
# Low-level drawing helpers
# ──────────────────────────────────────────────

def _draw_header_bar(c: canvas.Canvas, page_title: str = ""):
    """Top navy banner with fund name / page label."""
    bar_h = 36
    c.setFillColor(NAVY)
    c.rect(0, PAGE_H - bar_h, PAGE_W, bar_h, fill=1, stroke=0)
    c.setFillColor(WHITE)
    c.setFont(FONT_SANS_BOLD, 10)
    c.drawString(MARGIN_L, PAGE_H - bar_h + 12, page_title)
    report_date = datetime.today().strftime("%B %Y")
    c.setFont(FONT_SANS, 8)
    c.drawRightString(PAGE_W - MARGIN_R, PAGE_H - bar_h + 12, f"As of {report_date}")


def _draw_footer(c: canvas.Canvas, page_num: int, total_pages: int = 3):
    """Bottom footer with disclaimer + page number."""
    y = MARGIN_B - 18
    c.setStrokeColor(BORDER_GREY)
    c.setLineWidth(0.4)
    c.line(MARGIN_L, y + 14, PAGE_W - MARGIN_R, y + 14)
    c.setFillColor(MUTED_TEXT)
    c.setFont(FONT_SANS, 6)
    c.drawString(MARGIN_L, y,
                 "This document is for informational purposes only and does not constitute investment advice. "
                 "Past performance is not indicative of future results.")
    c.drawRightString(PAGE_W - MARGIN_R, y, f"Page {page_num} of {total_pages}")


def _draw_section_heading(c: canvas.Canvas, x: float, y: float, text: str,
                          width: float = CONTENT_W) -> float:
    """Dark blue section heading with underline. Returns y after heading."""
    c.setFillColor(NAVY)
    c.setFont(FONT_SANS_BOLD, 11)
    c.drawString(x, y, text)
    c.setStrokeColor(ACCENT_BLUE)
    c.setLineWidth(1.2)
    c.line(x, y - 3, x + width, y - 3)
    return y - 18


def _draw_key_value_row(c: canvas.Canvas, x: float, y: float,
                        label: str, value: str,
                        label_w: float = 140, row_w: float = 250,
                        font_size: float = 8.5) -> float:
    """Draw a label: value row. Returns next y."""
    c.setFillColor(MUTED_TEXT)
    c.setFont(FONT_SANS, font_size)
    c.drawString(x, y, label)
    c.setFillColor(BLACK)
    c.setFont(FONT_SANS_BOLD, font_size)
    c.drawString(x + label_w, y, value)
    return y - 14


def _embed_chart(c: canvas.Canvas, fig: plt.Figure, x: float, y: float,
                 width: float, dpi: int = 180) -> float:
    """Render matplotlib figure to PNG and embed on canvas. Returns y below chart."""
    from PIL import Image as PILImage
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", pad_inches=0.12,
                facecolor="white", edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    pil = PILImage.open(buf)
    px_w, px_h = pil.size
    buf.seek(0)
    img_h = width * (px_h / px_w)
    from reportlab.lib.utils import ImageReader
    c.drawImage(ImageReader(buf), x, y - img_h, width=width, height=img_h, mask="auto")
    return y - img_h - 6


def _draw_table(c: canvas.Canvas, x: float, y: float,
                headers: List[str], rows: List[List[str]],
                col_widths: List[float],
                col_aligns: Optional[List[str]] = None,
                font_size: float = 8, header_font_size: float = 8,
                row_height: float = 16) -> float:
    """Draw a styled table. Returns y below last row."""
    if col_aligns is None:
        col_aligns = ["LEFT"] * len(headers)
    total_w = sum(col_widths)

    # Header background
    c.setFillColor(NAVY)
    c.rect(x, y - row_height + 3, total_w, row_height, fill=1, stroke=0)
    c.setFillColor(WHITE)
    c.setFont(FONT_SANS_BOLD, header_font_size)
    cx = x
    for i, h in enumerate(headers):
        if col_aligns[i] == "RIGHT":
            c.drawRightString(cx + col_widths[i] - 4, y - row_height + 7, h)
        elif col_aligns[i] == "CENTER":
            c.drawCentredString(cx + col_widths[i] / 2, y - row_height + 7, h)
        else:
            c.drawString(cx + 4, y - row_height + 7, h)
        cx += col_widths[i]
    y -= row_height

    # Data rows
    for ri, row in enumerate(rows):
        if ri % 2 == 0:
            c.setFillColor(ROW_ALT)
            c.rect(x, y - row_height + 3, total_w, row_height, fill=1, stroke=0)
        else:
            c.setFillColor(WHITE)
            c.rect(x, y - row_height + 3, total_w, row_height, fill=1, stroke=0)

        cx = x
        for i, cell in enumerate(row):
            # Color P/L values
            text_color = BODY_TEXT
            if headers[i] in ("P/L", "Return", "1M", "YTD", "1Y", "Since Inception"):
                try:
                    v = float(cell.replace("%", "").replace(",", "").replace("+", ""))
                    text_color = POSITIVE if v >= 0 else NEGATIVE
                except (ValueError, AttributeError):
                    pass
            c.setFillColor(text_color)
            c.setFont(FONT_SANS, font_size)
            if col_aligns[i] == "RIGHT":
                c.drawRightString(cx + col_widths[i] - 4, y - row_height + 7, str(cell))
            elif col_aligns[i] == "CENTER":
                c.drawCentredString(cx + col_widths[i] / 2, y - row_height + 7, str(cell))
            else:
                c.drawString(cx + 4, y - row_height + 7, str(cell))
            cx += col_widths[i]

        # Subtle bottom border
        c.setStrokeColor(colors.Color(0, 0, 0, 0.06))
        c.setLineWidth(0.3)
        c.line(x, y - row_height + 3, x + total_w, y - row_height + 3)
        y -= row_height

    return y - 4


def _wrap_text(c_canvas: canvas.Canvas, text: str, x: float, y: float,
               max_width: float, font: str = FONT_SANS,
               font_size: float = 8.5, leading: float = 12,
               color=BODY_TEXT) -> float:
    """Draw word-wrapped text. Returns y after last line."""
    c_canvas.setFont(font, font_size)
    c_canvas.setFillColor(color)
    words = text.split()
    line = ""
    for w in words:
        test = f"{line} {w}".strip()
        if c_canvas.stringWidth(test, font, font_size) > max_width:
            c_canvas.drawString(x, y, line)
            y -= leading
            line = w
        else:
            line = test
    if line:
        c_canvas.drawString(x, y, line)
        y -= leading
    return y


# ──────────────────────────────────────────────
# Chart generators (factsheet-specific)
# ──────────────────────────────────────────────

def _make_performance_chart(portfolio_series: pd.Series,
                            benchmark_series: Optional[pd.Series],
                            portfolio_label: str,
                            benchmark_label: str) -> plt.Figure:
    """Growth-of-100 line chart (factsheet style)."""
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    ax.plot(portfolio_series.index, portfolio_series.values,
            color="#003366", linewidth=1.8, label=portfolio_label)
    if benchmark_series is not None and not benchmark_series.empty:
        ax.plot(benchmark_series.index, benchmark_series.values,
                color="#E94560", linewidth=1.3, linestyle="--", label=benchmark_label)
    ax.axhline(100, color="#CCCCCC", linewidth=0.5, linestyle=":")
    ax.set_ylabel("Growth of $100", fontsize=8, color="#333")
    ax.legend(fontsize=7.5, frameon=False, loc="upper left")
    ax.grid(axis="y", alpha=0.25, linewidth=0.4)
    ax.tick_params(labelsize=7, colors="#555")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.autofmt_xdate(rotation=25, ha="right")
    fig.tight_layout()
    return fig


def _make_donut_chart(weights: Dict[str, float], title: str = "Asset Allocation") -> plt.Figure:
    """Donut pie chart for allocation breakdown."""
    filtered = {k: abs(v) for k, v in weights.items() if abs(v) > 0.001}
    if not filtered:
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.axis("off")
        return fig
    labels = list(filtered.keys())
    sizes = list(filtered.values())
    clrs = CHART_COLORS[:len(labels)]

    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    wedges, _, autotexts = ax.pie(
        sizes, labels=None, autopct="%1.1f%%", startangle=140,
        colors=clrs, pctdistance=0.78,
        wedgeprops=dict(width=0.40, edgecolor="white", linewidth=1.2),
    )
    for at in autotexts:
        at.set_fontsize(7)
        at.set_color("white")
        at.set_fontweight("bold")
    ax.legend(wedges, labels, loc="center left", bbox_to_anchor=(1.02, 0.5),
              fontsize=7.5, frameon=False)
    ax.set_title(title, fontsize=10, fontweight="bold", color="#003366", pad=8)
    fig.subplots_adjust(right=0.62)
    return fig


def _make_sector_bar_chart(weights: Dict[str, float]) -> plt.Figure:
    """Horizontal bar chart for sector / asset breakdown."""
    filtered = {k: abs(v) for k, v in weights.items() if abs(v) > 0.001}
    labels = list(filtered.keys())
    vals = [v * 100 for v in filtered.values()]

    fig, ax = plt.subplots(figsize=(4.5, max(2.0, len(labels) * 0.35 + 0.5)))
    bars = ax.barh(labels, vals, color="#003366", height=0.55, edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Weight (%)", fontsize=7.5, color="#333")
    ax.set_title("Allocation Breakdown", fontsize=10, fontweight="bold", color="#003366", pad=8)
    ax.tick_params(labelsize=7.5, colors="#444")
    ax.invert_yaxis()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", alpha=0.2, linewidth=0.3)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{v:.1f}%", va="center", fontsize=7, color="#333")
    fig.tight_layout()
    return fig


# ──────────────────────────────────────────────
# Commentary auto-generator
# ──────────────────────────────────────────────

def _auto_commentary(rd: dict) -> List[str]:
    """Build manager commentary paragraphs from portfolio data."""
    perf = rd.get("performance", {})
    risk = rd.get("risk", {})
    weights = rd.get("weights", {})
    vol = risk.get("Volatility", 0)
    sharpe = risk.get("Sharpe", 0)
    max_dd = risk.get("Max Drawdown", 0)
    pv = rd.get("portfolio_value", 0)
    plabel = rd.get("portfolio_label", "Portfolio")
    blabel = rd.get("benchmark_label", "Benchmark")

    top_holdings = sorted(weights.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
    top_str = ", ".join(f"{t} ({w:.1%})" for t, w in top_holdings)

    paras = []

    # Overview paragraph
    since_inc = perf.get("Since Inception")
    inc_str = f"{since_inc:.2%}" if isinstance(since_inc, (int, float)) else "N/A"
    paras.append(
        f"The {plabel} strategy delivered a total return of {inc_str} since inception. "
        f"Annualised volatility of {vol:.2%} and a Sharpe ratio of {sharpe:.2f} indicate "
        f"{'attractive' if sharpe > 0.8 else 'moderate' if sharpe > 0.4 else 'below-target'} "
        f"risk-adjusted performance relative to the {blabel} benchmark."
    )

    # Allocation paragraph
    paras.append(
        f"Top allocations are concentrated in {top_str}. "
        f"The portfolio is designed to balance growth potential and downside protection, "
        f"with maximum drawdown contained at {max_dd:.2%}."
    )

    # Outlook
    if sharpe > 1.0:
        outlook = ("We maintain a constructive outlook and expect the current positioning to "
                   "continue generating favourable risk-adjusted returns.")
    elif sharpe > 0.5:
        outlook = ("The strategy is positioned defensively while seeking selective opportunities "
                   "to enhance returns. We will monitor macro developments closely.")
    else:
        outlook = ("We are reviewing the current allocation to improve risk-adjusted outcomes "
                   "and will rebalance towards higher-conviction positions.")
    paras.append(outlook)

    return paras


# ──────────────────────────────────────────────
# PAGE BUILDERS
# ──────────────────────────────────────────────

def _page1(c: canvas.Canvas, rd: dict):
    """Page 1: Title, objective, key facts, performance table, performance chart."""
    portfolio_name = rd.get("client_name", "Investment Portfolio")
    _draw_header_bar(c, f"{portfolio_name}  —  Fund Factsheet")

    y = PAGE_H - MARGIN_T - 30

    # ── Title block ──
    c.setFillColor(NAVY)
    c.setFont(FONT_SANS_BOLD, 22)
    c.drawString(MARGIN_L, y, portfolio_name)
    y -= 18
    c.setFillColor(ACCENT_BLUE)
    c.setFont(FONT_SANS, 10)
    c.drawString(MARGIN_L, y, "Monthly Fund Factsheet")
    y -= 6
    c.setStrokeColor(ACCENT_BLUE)
    c.setLineWidth(2)
    c.line(MARGIN_L, y, MARGIN_L + 120, y)
    y -= 22

    # ── Investment Objective ──
    y = _draw_section_heading(c, MARGIN_L, y, "Investment Objective")
    strategy = rd.get("portfolio_label", "Diversified")
    obj_text = (
        f"The fund aims to achieve long-term capital appreciation through a {strategy} strategy, "
        f"investing across multiple asset classes to optimise risk-adjusted returns relative to "
        f"the {rd.get('benchmark_label', 'market')} benchmark."
    )
    y = _wrap_text(c, obj_text, MARGIN_L, y, CONTENT_W, font_size=9, leading=13)
    y -= 10

    # ── Key Facts (left column) & Performance Table (right column) ──
    col_left_w = CONTENT_W * 0.45
    col_right_w = CONTENT_W * 0.50
    col_right_x = MARGIN_L + CONTENT_W - col_right_w

    y_left = y
    y_right = y

    # Left: Key Facts
    y_left = _draw_section_heading(c, MARGIN_L, y_left, "Key Facts", col_left_w)

    pv = rd.get("portfolio_value")
    currency = rd.get("currency", "USD")
    rf_rate = rd.get("risk", {}).get("Risk-Free Rate", rd.get("rf", "N/A"))
    report_date = rd.get("report_date", datetime.today().strftime("%Y-%m-%d"))

    facts = [
        ("Portfolio Value", f"{pv:,.2f} {currency}" if pv else "N/A"),
        ("Report Date", report_date),
        ("Inception Date", rd.get("inception_date", "N/A")),
        ("No. of Holdings", str(rd.get("num_holdings", "N/A"))),
        ("Strategy", rd.get("portfolio_label", "N/A")),
        ("Benchmark", rd.get("benchmark_label", "N/A")),
        ("Currency", currency),
        ("Risk-Free Rate", f"{rf_rate:.2%}" if isinstance(rf_rate, (int, float)) else str(rf_rate)),
    ]
    for label, val in facts:
        y_left = _draw_key_value_row(c, MARGIN_L, y_left, label, val,
                                     label_w=110, font_size=8.5)

    # Right: Performance Summary Table
    y_right = _draw_section_heading(c, col_right_x, y_right, "Performance Summary", col_right_w)

    perf = rd.get("performance", {})
    perf_headers = ["Period", "Return"]
    perf_rows = []
    for period in ["1M", "YTD", "1Y", "3Y Ann.", "5Y Ann.", "Since Inception"]:
        ret = perf.get(period)
        if ret is not None and isinstance(ret, (int, float)):
            perf_rows.append([period, f"{ret:+.2%}"])
        elif ret is not None:
            perf_rows.append([period, str(ret)])

    if perf_rows:
        cw = [col_right_w * 0.45, col_right_w * 0.55]
        y_right = _draw_table(c, col_right_x, y_right, perf_headers, perf_rows,
                              col_widths=cw, col_aligns=["LEFT", "RIGHT"],
                              font_size=8.5, row_height=17)

    y = min(y_left, y_right) - 16

    # ── Performance Chart ──
    y = _draw_section_heading(c, MARGIN_L, y, "Cumulative Performance")

    portfolio_series = rd.get("portfolio_series")
    benchmark_series = rd.get("benchmark_series")
    if portfolio_series is not None and not portfolio_series.empty:
        fig = _make_performance_chart(
            portfolio_series, benchmark_series,
            rd.get("portfolio_label", "Portfolio"),
            rd.get("benchmark_label", "Benchmark"),
        )
        chart_w = CONTENT_W
        y = _embed_chart(c, fig, MARGIN_L, y, chart_w)
    else:
        c.setFillColor(MUTED_TEXT)
        c.setFont(FONT_SANS_OBL, 9)
        c.drawString(MARGIN_L, y - 12, "Performance data not available.")
        y -= 20

    _draw_footer(c, 1)


def _page2(c: canvas.Canvas, rd: dict):
    """Page 2: Top holdings, allocation donut, sector chart, risk metrics."""
    portfolio_name = rd.get("client_name", "Investment Portfolio")
    _draw_header_bar(c, f"{portfolio_name}  —  Portfolio Analytics")

    y = PAGE_H - MARGIN_T - 30

    # ── Top Holdings Table ──
    y = _draw_section_heading(c, MARGIN_L, y, "Top Holdings")

    holdings_df = rd.get("holdings_df")
    if holdings_df is not None and isinstance(holdings_df, pd.DataFrame) and not holdings_df.empty:
        h_headers = list(holdings_df.columns)
        h_rows = []
        for _, row in holdings_df.head(10).iterrows():
            formatted = []
            for col in h_headers:
                val = row[col]
                if col == "Weight":
                    formatted.append(f"{val:.1%}" if isinstance(val, (int, float)) else str(val))
                elif col in ("Market Value", "P/L"):
                    if isinstance(val, (int, float)):
                        prefix = "+" if val > 0 and col == "P/L" else ""
                        formatted.append(f"{prefix}{val:,.2f}")
                    else:
                        formatted.append(str(val))
                else:
                    formatted.append(str(val))
            h_rows.append(formatted)

        n_cols = len(h_headers)
        cw = [CONTENT_W / n_cols] * n_cols
        aligns = ["LEFT"] * n_cols
        for i, h in enumerate(h_headers):
            if h in ("Weight", "Market Value", "P/L"):
                aligns[i] = "RIGHT"
        y = _draw_table(c, MARGIN_L, y, h_headers, h_rows,
                        col_widths=cw, col_aligns=aligns, font_size=8, row_height=16)
    else:
        c.setFillColor(MUTED_TEXT)
        c.setFont(FONT_SANS_OBL, 9)
        c.drawString(MARGIN_L, y - 12, "Holdings data not available.")
        y -= 20

    y -= 10

    # ── Charts side by side: Donut (left) & Sector bars (right) ──
    weights = rd.get("weights", {})
    chart_half_w = CONTENT_W / 2 - 6

    if weights:
        y_charts = y

        # Left: Donut
        y_left = _draw_section_heading(c, MARGIN_L, y_charts, "Asset Allocation", chart_half_w)
        fig_donut = _make_donut_chart(weights)
        y_left = _embed_chart(c, fig_donut, MARGIN_L, y_left, chart_half_w)

        # Right: Sector bars
        right_x = MARGIN_L + CONTENT_W / 2 + 6
        y_right = _draw_section_heading(c, right_x, y_charts, "Allocation Breakdown", chart_half_w)
        fig_sector = _make_sector_bar_chart(weights)
        y_right = _embed_chart(c, fig_sector, right_x, y_right, chart_half_w)

        y = min(y_left, y_right) - 10
    else:
        y -= 20

    # ── Calendar Year Returns ──
    cal_year = rd.get("calendar_year_returns", {})
    if cal_year:
        y = _draw_section_heading(c, MARGIN_L, y, "Calendar Year Returns")
        cal_headers = ["Year", "Return"]
        cal_rows = [[yr, f"{ret:+.2%}"] for yr, ret in sorted(cal_year.items(), reverse=True)]
        cw_cal = [CONTENT_W * 0.45, CONTENT_W * 0.55]
        y = _draw_table(c, MARGIN_L, y, cal_headers, cal_rows,
                        col_widths=cw_cal, col_aligns=["LEFT", "RIGHT"],
                        font_size=8.5, row_height=17)
        y -= 10

    # ── Portfolio Risk Metrics ──
    y = _draw_section_heading(c, MARGIN_L, y, "Risk Metrics")

    risk = rd.get("risk", {})
    risk_headers = ["Metric", "Value"]
    risk_rows = []
    metric_map = {
        "Volatility": lambda v: f"{v:.2%}",
        "Sharpe": lambda v: f"{v:.2f}",
        "Sharpe Ratio": lambda v: f"{v:.2f}",
        "Max Drawdown": lambda v: f"{v:.2%}",
        "Correlation": lambda v: f"{v:.2f}",
        "Beta": lambda v: f"{v:.2f}",
        "Sortino": lambda v: f"{v:.2f}",
        "Sortino Ratio": lambda v: f"{v:.2f}",
        "Tracking Error": lambda v: f"{v:.2%}",
        "Information Ratio": lambda v: f"{v:.2f}",
        "Alpha": lambda v: f"{v:.2%}",
        "R-Squared": lambda v: f"{v:.2f}",
        "R²": lambda v: f"{v:.2f}",
    }
    for metric, val in risk.items():
        if val is None:
            continue
        if isinstance(val, (int, float)):
            fmt_fn = metric_map.get(metric, lambda v: f"{v:.4f}")
            risk_rows.append([metric, fmt_fn(val)])
        else:
            risk_rows.append([metric, str(val)])

    if risk_rows:
        cw = [CONTENT_W * 0.55, CONTENT_W * 0.45]
        y = _draw_table(c, MARGIN_L, y, risk_headers, risk_rows,
                        col_widths=cw, col_aligns=["LEFT", "RIGHT"],
                        font_size=8.5, row_height=17)

    _draw_footer(c, 2)


def _page3(c: canvas.Canvas, rd: dict):
    """Page 3: Manager commentary, strategy description, risk disclaimer."""
    portfolio_name = rd.get("client_name", "Investment Portfolio")
    _draw_header_bar(c, f"{portfolio_name}  —  Commentary & Disclosures")

    y = PAGE_H - MARGIN_T - 30

    # ── Manager Commentary ──
    y = _draw_section_heading(c, MARGIN_L, y, "Manager Commentary")

    commentary = rd.get("commentary")
    if not commentary:
        commentary = _auto_commentary(rd)
    if isinstance(commentary, str):
        commentary = [commentary]

    for para in commentary:
        y = _wrap_text(c, para, MARGIN_L, y, CONTENT_W,
                       font_size=9, leading=13, color=BODY_TEXT)
        y -= 8

    y -= 8

    # ── Strategy Description ──
    y = _draw_section_heading(c, MARGIN_L, y, "Strategy Description")

    strategy = rd.get("portfolio_label", "Diversified Multi-Asset")
    benchmark = rd.get("benchmark_label", "Benchmark")
    risk = rd.get("risk", {})
    vol = risk.get("Volatility", 0)
    weights = rd.get("weights", {})
    n_assets = len([w for w in weights.values() if abs(w) > 0.001])

    strategy_paras = [
        f"The {strategy} strategy constructs a portfolio of {n_assets} assets, "
        f"optimised to maximise risk-adjusted returns as measured by the Sharpe ratio. "
        f"The strategy targets an annualised volatility of approximately {vol:.1%} "
        f"while maintaining broad diversification across asset classes.",

        f"Portfolio weights are determined using mean-variance optimisation with "
        f"constraints to prevent excessive concentration. The strategy is benchmarked "
        f"against the {benchmark} and is rebalanced periodically to maintain target allocations.",

        f"The investment process combines quantitative optimisation with risk management "
        f"overlays, including drawdown monitoring and correlation analysis, to ensure "
        f"the portfolio remains within its risk mandate."
    ]

    for para in strategy_paras:
        y = _wrap_text(c, para, MARGIN_L, y, CONTENT_W,
                       font_size=9, leading=13, color=BODY_TEXT)
        y -= 8

    y -= 10

    # ── Decorative separator ──
    c.setStrokeColor(ACCENT_BLUE)
    c.setLineWidth(0.6)
    c.line(MARGIN_L, y, MARGIN_L + CONTENT_W, y)
    y -= 18

    # ── Risk Disclaimer ──
    y = _draw_section_heading(c, MARGIN_L, y, "Important Information & Risk Disclosures")

    disclaimers = [
        "This document is provided for informational purposes only and does not constitute "
        "an offer or solicitation to buy or sell any securities. The information contained herein "
        "is based on sources believed to be reliable but is not guaranteed as to accuracy or completeness.",

        "Past performance is not indicative of future results. The value of investments and the "
        "income derived from them can go down as well as up, and investors may not get back the "
        "amount originally invested. Returns shown are hypothetical and based on backtested data; "
        "actual trading results may differ materially.",

        "All portfolio optimisation metrics, including Sharpe ratio, volatility, and maximum drawdown, "
        "are based on historical data and may not reflect future risk characteristics. Diversification "
        "does not guarantee a profit or protect against a loss.",

        "This report was generated by Portfolio Optimizer, an analytical tool for educational and "
        "informational purposes. It should not be relied upon as the sole basis for an investment decision. "
        "Investors should consult with a qualified financial advisor before making any investment decisions.",

        "© " + str(datetime.today().year) + " Portfolio Optimizer. All rights reserved."
    ]

    for para in disclaimers:
        y = _wrap_text(c, para, MARGIN_L, y, CONTENT_W,
                       font=FONT_SANS, font_size=7.5, leading=10.5, color=MUTED_TEXT)
        y -= 6

    _draw_footer(c, 3)


# ──────────────────────────────────────────────
# PUBLIC API
# ──────────────────────────────────────────────

def generate_factsheet(report_data: dict, output_path: str) -> str:
    """
    Generate a 3-page fund-factsheet-style PDF.

    Parameters
    ----------
    report_data : dict
        Same schema as used by generate_portfolio_report().
    output_path : str
        Destination file path for the PDF.

    Returns
    -------
    str
        Absolute path to the generated PDF.
    """
    c = canvas.Canvas(output_path, pagesize=A4)
    c.setTitle(f"{report_data.get('client_name', 'Portfolio')} — Fund Factsheet")
    c.setAuthor("Portfolio Optimizer")

    _page1(c, report_data)
    c.showPage()

    _page2(c, report_data)
    c.showPage()

    _page3(c, report_data)
    c.showPage()

    c.save()
    return os.path.abspath(output_path)


# ──────────────────────────────────────────────
# Standalone demo
# ──────────────────────────────────────────────

if __name__ == "__main__":
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=252, freq="B")
    cum = (1 + pd.Series(np.random.normal(0.0004, 0.012, len(dates)), index=dates)).cumprod() * 100
    bench = (1 + pd.Series(np.random.normal(0.0003, 0.011, len(dates)), index=dates)).cumprod() * 100

    holdings = pd.DataFrame({
        "Ticker": ["SPY", "AGG", "GLD", "VNQ", "QQQ"],
        "Weight": [0.30, 0.25, 0.20, 0.15, 0.10],
        "Market Value": [75000.0, 62500.0, 50000.0, 37500.0, 25000.0],
        "P/L": [4200.0, 1500.0, 2100.0, -300.0, 1800.0],
    })

    sample_data = {
        "client_name": "Global Balanced Fund",
        "report_date": "2026-02-07",
        "portfolio_value": 250000,
        "currency": "USD",
        "portfolio_label": "Max Sharpe Optimised",
        "benchmark_label": "S&P 500 (SPY)",
        "performance": {
            "1M": 0.012,
            "YTD": 0.034,
            "1Y": 0.091,
            "Since Inception": 0.184,
        },
        "risk": {
            "Volatility": 0.145,
            "Sharpe": 1.12,
            "Max Drawdown": -0.087,
            "Correlation": 0.82,
        },
        "weights": {
            "SPY": 0.30, "AGG": 0.25, "GLD": 0.20,
            "VNQ": 0.15, "QQQ": 0.10,
        },
        "holdings_df": holdings,
        "portfolio_series": cum,
        "benchmark_series": bench,
        "commentary": None,  # will auto-generate
    }

    path = generate_factsheet(sample_data, "fund_factsheet.pdf")
    print(f"✅ Factsheet generated: {path}")
