"""
portfolio_report.py
───────────────────
Client-ready PDF portfolio report generator.

Usage:
    from portfolio_report import generate_portfolio_report
    generate_portfolio_report(report_data, "portfolio_report.pdf")

Dependencies: reportlab, matplotlib, pandas, numpy
"""

from __future__ import annotations

import io
import os
import tempfile
from datetime import datetime
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm, mm
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

# ──────────────────────────────────────────────
# Color palette
# ──────────────────────────────────────────────
BRAND_DARK = colors.HexColor("#1a1a2e")
BRAND_ACCENT = colors.HexColor("#16213e")
BRAND_BLUE = colors.HexColor("#0f3460")
BRAND_HIGHLIGHT = colors.HexColor("#e94560")
HEADER_BG = colors.HexColor("#f0f0f5")
ROW_ALT = colors.HexColor("#f9f9fc")
TEXT_DARK = colors.HexColor("#222222")
TEXT_MUTED = colors.HexColor("#666666")

PIE_COLORS = [
    "#0f3460", "#e94560", "#00b4d8", "#fca311",
    "#6a994e", "#9b5de5", "#f15bb5", "#00f5d4",
    "#fb8500", "#219ebc", "#8338ec", "#ff006e",
]

LINE_COLORS = ["#0f3460", "#e94560", "#00b4d8", "#fca311", "#6a994e", "#9b5de5"]


# ──────────────────────────────────────────────
# Styles
# ──────────────────────────────────────────────
def _build_styles() -> dict:
    """Return a dict of reusable ParagraphStyles."""
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "ReportTitle",
            parent=base["Title"],
            fontSize=26,
            leading=32,
            textColor=BRAND_DARK,
            spaceAfter=4 * mm,
        ),
        "subtitle": ParagraphStyle(
            "ReportSubtitle",
            parent=base["Normal"],
            fontSize=11,
            leading=15,
            textColor=TEXT_MUTED,
            spaceAfter=2 * mm,
        ),
        "section": ParagraphStyle(
            "SectionHeader",
            parent=base["Heading2"],
            fontSize=15,
            leading=20,
            textColor=BRAND_DARK,
            spaceBefore=10 * mm,
            spaceAfter=4 * mm,
            borderPadding=(0, 0, 2 * mm, 0),
        ),
        "body": ParagraphStyle(
            "BodyText",
            parent=base["Normal"],
            fontSize=10,
            leading=14,
            textColor=TEXT_DARK,
            spaceAfter=3 * mm,
        ),
        "kpi_value": ParagraphStyle(
            "KPIValue",
            parent=base["Normal"],
            fontSize=22,
            leading=26,
            textColor=BRAND_DARK,
            alignment=TA_CENTER,
        ),
        "kpi_label": ParagraphStyle(
            "KPILabel",
            parent=base["Normal"],
            fontSize=9,
            leading=12,
            textColor=TEXT_MUTED,
            alignment=TA_CENTER,
        ),
        "disclaimer": ParagraphStyle(
            "Disclaimer",
            parent=base["Normal"],
            fontSize=7,
            leading=9,
            textColor=TEXT_MUTED,
            spaceAfter=2 * mm,
        ),
    }


# ──────────────────────────────────────────────
# Chart generation helpers
# ──────────────────────────────────────────────

def _chart_to_image(fig: plt.Figure, width_cm: float = 16, dpi: int = 180) -> Image:
    """Render a matplotlib figure into a reportlab Image flowable."""
    buf = io.BytesIO()
    fig_w, fig_h = fig.get_size_inches()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    buf.seek(0)

    # Read actual rendered image size so the PDF matches what was saved
    from PIL import Image as PILImage
    pil_img = PILImage.open(buf)
    px_w, px_h = pil_img.size
    buf.seek(0)

    img_w = width_cm * cm
    img_h = img_w * (px_h / px_w)
    return Image(buf, width=img_w, height=img_h)


def generate_pie_chart(weights: Dict[str, float], title: str = "Asset Allocation") -> plt.Figure:
    """Create a donut-style pie chart for portfolio weights."""
    labels = list(weights.keys())
    sizes = [abs(v) for v in weights.values()]

    # Filter out zero / negligible weights
    filtered = [(l, s) for l, s in zip(labels, sizes) if s > 0.001]
    if not filtered:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No allocations", ha="center", va="center", fontsize=12)
        ax.axis("off")
        return fig

    labels, sizes = zip(*filtered)
    clrs = PIE_COLORS[: len(labels)]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=None,
        autopct="%1.1f%%",
        startangle=140,
        colors=clrs,
        pctdistance=0.78,
        wedgeprops=dict(width=0.42, edgecolor="white", linewidth=1.5),
    )
    for at in autotexts:
        at.set_fontsize(8)
        at.set_color("white")
        at.set_fontweight("bold")

    ax.legend(
        wedges,
        labels,
        loc="center left",
        bbox_to_anchor=(1.05, 0.5),
        fontsize=9,
        frameon=False,
    )
    ax.set_title(title, fontsize=13, fontweight="bold", color="#1a1a2e", pad=14)
    fig.subplots_adjust(right=0.65)
    return fig


def generate_frontier_chart(
    pv_cloud,
    pr_cloud,
    port_v: float,
    port_r: float,
    cml_v=None,
    cml_r=None,
    selected_label: str = "Selected",
) -> plt.Figure:
    """Efficient frontier scatter chart for the report."""
    fig, ax = plt.subplots(figsize=(7, 4.2))
    pv = np.asarray(pv_cloud)
    pr = np.asarray(pr_cloud)

    ax.scatter(pv, pr, s=2, alpha=0.25, c="#a0b4d0", edgecolors="none",
               label="Simulated portfolios", rasterized=True)

    if cml_v and cml_r:
        ax.plot(cml_v, cml_r, color="#fca311", linewidth=2, label="CML", zorder=5)

    sharpe_c = np.where(pv > 1e-12, pr / pv, np.nan)
    i_best = int(np.nanargmax(sharpe_c))
    ax.scatter([pv[i_best]], [pr[i_best]], s=120, marker="*",
               c="#fca311", edgecolors="#333", linewidths=0.5,
               label="Max Sharpe", zorder=6)

    ax.scatter([port_v], [port_r], s=80, marker="D", c="#e94560",
               edgecolors="white", linewidths=1, label=selected_label, zorder=7)

    ax.set_xlabel("Annualized Volatility", fontsize=9, color="#333")
    ax.set_ylabel("Annualized Return", fontsize=9, color="#333")
    import matplotlib.ticker as mtick
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))
    ax.tick_params(labelsize=8, colors="#666")
    ax.legend(fontsize=7, frameon=True, edgecolor="#ccc", loc="upper left")
    ax.grid(True, alpha=0.2, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig


def generate_performance_chart(
    portfolio_series: Optional[pd.Series] = None,
    benchmark_series: Optional[pd.Series] = None,
    portfolio_label: str = "Portfolio",
    benchmark_label: str = "Benchmark",
) -> Optional[plt.Figure]:
    """Line chart of cumulative indexed returns (base = 100)."""
    if portfolio_series is None or portfolio_series.empty:
        return None

    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.plot(
        portfolio_series.index,
        portfolio_series.values,
        color=LINE_COLORS[0],
        linewidth=1.8,
        label=portfolio_label,
    )
    if benchmark_series is not None and not benchmark_series.empty:
        ax.plot(
            benchmark_series.index,
            benchmark_series.values,
            color=LINE_COLORS[1],
            linewidth=1.4,
            linestyle="--",
            label=benchmark_label,
        )

    ax.axhline(100, color="#cccccc", linewidth=0.6, linestyle=":")
    ax.set_ylabel("Growth of 100", fontsize=9)
    ax.set_title("Cumulative Performance", fontsize=13, fontweight="bold", color="#1a1a2e", pad=10)
    ax.legend(fontsize=9, frameon=False)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.autofmt_xdate(rotation=30, ha="right")
    fig.tight_layout()
    return fig


# ──────────────────────────────────────────────
# Table formatting helpers
# ──────────────────────────────────────────────

def _format_table(
    data: List[List[str]],
    col_widths: Optional[List[float]] = None,
    right_align_cols: Optional[List[int]] = None,
) -> Table:
    """
    Build a reportlab Table with professional styling:
    - light gray header row
    - alternating row colors
    - subtle borders
    """
    if col_widths is None:
        n_cols = len(data[0]) if data else 1
        col_widths = [None] * n_cols

    tbl = Table(data, colWidths=col_widths, repeatRows=1)

    style_cmds: list = [
        # Header
        ("BACKGROUND", (0, 0), (-1, 0), HEADER_BG),
        ("TEXTCOLOR", (0, 0), (-1, 0), BRAND_DARK),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 9),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 6),
        ("TOPPADDING", (0, 0), (-1, 0), 6),
        # Body
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 1), (-1, -1), 9),
        ("TOPPADDING", (0, 1), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 1), (-1, -1), 4),
        # Grid
        ("LINEBELOW", (0, 0), (-1, 0), 0.8, BRAND_DARK),
        ("LINEBELOW", (0, -1), (-1, -1), 0.5, colors.grey),
        ("LINEBELOW", (0, 1), (-1, -2), 0.25, colors.Color(0, 0, 0, 0.08)),
        # Alignment
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]

    # Alternating row colors
    for i in range(1, len(data)):
        if i % 2 == 0:
            style_cmds.append(("BACKGROUND", (0, i), (-1, i), ROW_ALT))

    # Right-align numeric columns
    if right_align_cols:
        for c in right_align_cols:
            style_cmds.append(("ALIGN", (c, 0), (c, -1), "RIGHT"))

    tbl.setStyle(TableStyle(style_cmds))
    return tbl


def _kv_table(metrics: Dict[str, Any], fmt: str = "{}") -> Table:
    """Simple two-column key/value table."""
    rows = [["Metric", "Value"]]
    for k, v in metrics.items():
        rows.append([str(k), fmt.format(v) if not isinstance(v, str) else v])
    return _format_table(rows, col_widths=[8 * cm, 6 * cm], right_align_cols=[1])


def _holdings_table(df: pd.DataFrame) -> Table:
    """Format a holdings DataFrame into a styled Table."""
    headers = list(df.columns)
    rows = [headers]
    numeric_cols = []

    for idx, col in enumerate(headers):
        if col in ("Weight", "Market Value", "P/L"):
            numeric_cols.append(idx)

    for _, row in df.iterrows():
        formatted = []
        for col in headers:
            val = row[col]
            if col == "Weight":
                formatted.append(f"{val:.1%}" if isinstance(val, (int, float)) else str(val))
            elif col in ("Market Value", "P/L"):
                formatted.append(f"{val:,.2f}" if isinstance(val, (int, float)) else str(val))
            else:
                formatted.append(str(val))
        rows.append(formatted)

    n = len(headers)
    widths = [max(3.5 * cm, 14 * cm / n)] * n
    return _format_table(rows, col_widths=widths, right_align_cols=numeric_cols)


# ──────────────────────────────────────────────
# PDF assembly
# ──────────────────────────────────────────────

def _footer(canvas, doc):
    """Draw page footer with page number and disclaimer line."""
    canvas.saveState()
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(TEXT_MUTED)
    canvas.drawString(
        2 * cm,
        1.2 * cm,
        "This report is for informational purposes only and does not constitute investment advice.",
    )
    canvas.drawRightString(
        A4[0] - 2 * cm,
        1.2 * cm,
        f"Page {doc.page}",
    )
    # Thin line above footer
    canvas.setStrokeColor(colors.Color(0, 0, 0, 0.1))
    canvas.setLineWidth(0.4)
    canvas.line(2 * cm, 1.6 * cm, A4[0] - 2 * cm, 1.6 * cm)
    canvas.restoreState()


def generate_portfolio_report(report_data: dict, output_path: str) -> str:
    """
    Generate a client-ready PDF portfolio report.

    Parameters
    ----------
    report_data : dict
        Dictionary containing all report inputs (see module docstring for schema).
    output_path : str
        File path for the output PDF.

    Returns
    -------
    str
        The absolute path to the generated PDF.
    """
    styles = _build_styles()
    elements: list = []

    # ── helpers ──
    def add_section(title: str):
        elements.append(Paragraph(title, styles["section"]))

    def add_spacer(h_mm: float = 4):
        elements.append(Spacer(1, h_mm * mm))

    # ── 1) Cover header ──────────────────────────
    client_name = report_data.get("client_name", "Portfolio Report")
    report_date = report_data.get("report_date", datetime.today().strftime("%Y-%m-%d"))
    portfolio_value = report_data.get("portfolio_value")
    currency = report_data.get("currency", "USD")

    elements.append(Spacer(1, 12 * mm))
    elements.append(Paragraph(f"Portfolio Report", styles["title"]))
    elements.append(Paragraph(f"Prepared for <b>{client_name}</b>", styles["subtitle"]))
    elements.append(Paragraph(f"Report date: {report_date}", styles["subtitle"]))
    if portfolio_value is not None:
        elements.append(
            Paragraph(
                f"Portfolio value: <b>{portfolio_value:,.2f} {currency}</b>",
                styles["subtitle"],
            )
        )
    add_spacer(6)

    # Decorative rule
    rule_data = [["" ]]
    rule = Table(rule_data, colWidths=[16 * cm])
    rule.setStyle(
        TableStyle([
            ("LINEBELOW", (0, 0), (-1, 0), 1.2, BRAND_HIGHLIGHT),
        ])
    )
    elements.append(rule)
    add_spacer(8)

    # ── 2) Performance summary ────────────────────
    performance = report_data.get("performance")
    if performance:
        add_section("Performance Summary")
        perf_rows = [["Period", "Return"]]
        for period, ret in performance.items():
            if isinstance(ret, (int, float)):
                perf_rows.append([str(period), f"{ret:.2%}"])
            else:
                perf_rows.append([str(period), str(ret)])
        elements.append(
            _format_table(perf_rows, col_widths=[8 * cm, 6 * cm], right_align_cols=[1])
        )
        add_spacer(4)

    # ── 3) Risk metrics ───────────────────────────
    risk = report_data.get("risk")
    if risk:
        add_section("Risk Metrics")
        risk_rows = [["Metric", "Value"]]
        for metric, val in risk.items():
            if isinstance(val, (int, float)):
                if "sharpe" in metric.lower():
                    risk_rows.append([str(metric), f"{val:.2f}"])
                else:
                    risk_rows.append([str(metric), f"{val:.2%}"])
            else:
                risk_rows.append([str(metric), str(val)])
        elements.append(
            _format_table(risk_rows, col_widths=[8 * cm, 6 * cm], right_align_cols=[1])
        )
        add_spacer(4)

    # ── 4) Asset allocation pie chart ─────────────
    weights = report_data.get("weights")
    if weights:
        add_section("Asset Allocation")
        fig_pie = generate_pie_chart(weights)
        elements.append(_chart_to_image(fig_pie, width_cm=14))
        add_spacer(4)

    # ── 5) Holdings table ─────────────────────────
    holdings_df = report_data.get("holdings_df")
    if holdings_df is not None and isinstance(holdings_df, pd.DataFrame) and not holdings_df.empty:
        add_section("Holdings")
        elements.append(_holdings_table(holdings_df))
        add_spacer(4)

    # ── 6) Performance chart ──────────────────────
    portfolio_series = report_data.get("portfolio_series")
    benchmark_series = report_data.get("benchmark_series")
    if portfolio_series is not None:
        add_section("Performance Chart")
        fig_perf = generate_performance_chart(
            portfolio_series=portfolio_series,
            benchmark_series=benchmark_series,
            portfolio_label=report_data.get("portfolio_label", "Portfolio"),
            benchmark_label=report_data.get("benchmark_label", "Benchmark"),
        )
        if fig_perf is not None:
            elements.append(_chart_to_image(fig_perf, width_cm=16))
            add_spacer(4)

    # ── 7) Efficient frontier chart ─────────────
    frontier = report_data.get("frontier")
    if frontier:
        add_section("Efficient Frontier")
        fig_front = generate_frontier_chart(
            pv_cloud=frontier["pv_cloud"],
            pr_cloud=frontier["pr_cloud"],
            port_v=frontier["port_v"],
            port_r=frontier["port_r"],
            cml_v=frontier.get("cml_v"),
            cml_r=frontier.get("cml_r"),
            selected_label=frontier.get("selected_label", "Selected"),
        )
        elements.append(_chart_to_image(fig_front, width_cm=16))
        add_spacer(4)

    # ── 8) Manager commentary ─────────────────────
    commentary = report_data.get("commentary")
    if commentary:
        add_section("Manager Commentary")
        if isinstance(commentary, str):
            commentary = [commentary]
        for para_text in commentary:
            elements.append(Paragraph(para_text, styles["body"]))
            add_spacer(1)

    # ── 8) Transactions (optional) ────────────────
    transactions_df = report_data.get("transactions_df")
    if transactions_df is not None and isinstance(transactions_df, pd.DataFrame) and not transactions_df.empty:
        add_section("Recent Transactions")
        headers = list(transactions_df.columns)
        rows = [headers]
        for _, row in transactions_df.iterrows():
            rows.append([str(row[c]) for c in headers])
        n = len(headers)
        widths = [max(3 * cm, 16 * cm / n)] * n
        elements.append(_format_table(rows, col_widths=widths))
        add_spacer(4)

    # ── Build PDF ─────────────────────────────────
    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2.5 * cm,
        title=f"Portfolio Report — {client_name}",
        author="Portfolio Optimizer",
    )
    doc.build(elements, onFirstPage=_footer, onLaterPages=_footer)
    return os.path.abspath(output_path)


# ──────────────────────────────────────────────
# Example / standalone usage
# ──────────────────────────────────────────────
if __name__ == "__main__":
    # Build sample data
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=252, freq="B")
    cum = (1 + pd.Series(np.random.normal(0.0004, 0.012, len(dates)), index=dates)).cumprod() * 100
    bench = (1 + pd.Series(np.random.normal(0.0003, 0.011, len(dates)), index=dates)).cumprod() * 100

    holdings = pd.DataFrame({
        "Ticker": ["SPY", "AGG", "GLD", "VNQ"],
        "Weight": [0.35, 0.40, 0.15, 0.10],
        "Market Value": [87500.0, 100000.0, 37500.0, 25000.0],
        "P/L": [4200.0, 1500.0, 2100.0, -300.0],
    })

    report_data = {
        "client_name": "Jane Doe",
        "report_date": "2026-02-06",
        "portfolio_value": 250000,
        "currency": "EUR",
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
        },
        "weights": {
            "SPY": 0.35,
            "AGG": 0.40,
            "GLD": 0.15,
            "VNQ": 0.10,
        },
        "holdings_df": holdings,
        "portfolio_series": cum,
        "benchmark_series": bench,
        "portfolio_label": "Portfolio",
        "benchmark_label": "S&P 500",
        "commentary": [
            "The portfolio maintained a defensive tilt throughout Q4, with a 40% allocation to "
            "investment-grade bonds providing stability during the volatility in December.",
            "Equity exposure was concentrated in broad market ETFs to capture the late-year rally. "
            "Gold allocation contributed positively as real yields declined.",
        ],
    }

    path = generate_portfolio_report(report_data, "portfolio_report.pdf")
    print(f"✅ Report generated: {path}")
