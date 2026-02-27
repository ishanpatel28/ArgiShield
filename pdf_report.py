"""
pdf_report.py
=============
AgriShield – PDF Report Generator (ReportLab)
----------------------------------------------
Generates a clean, print-ready PDF containing:
  - Farm header and seasonal context
  - Hedging summary (plain language)
  - Per-crop risk breakdown (revenue, P50/P10 loss, yield distribution)
  - Hedge recommendations per crop
  - Farm-level summary and metadata

Returns bytes suitable for streaming in a FastAPI response.
"""

import io
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    BaseDocTemplate, Frame, HRFlowable, PageTemplate, Paragraph,
    Spacer, Table, TableStyle, KeepTogether,
)
from reportlab.platypus.flowables import Flowable

# ── Color palette (mirrors CSS) ───────────────────────────────────────────
GREEN_DARK   = colors.HexColor("#2d4a2d")
GREEN_MID    = colors.HexColor("#4a6741")
GREEN_MUTED  = colors.HexColor("#6b8f5e")
GREEN_LIGHT  = colors.HexColor("#a8c49a")
GREEN_WASH   = colors.HexColor("#eaf0e5")
EARTH_DARK   = colors.HexColor("#5c4033")
EARTH_MID    = colors.HexColor("#8b6555")
EARTH_WASH   = colors.HexColor("#f5ede0")
EARTH_LIGHT  = colors.HexColor("#c4a882")
AMBER        = colors.HexColor("#c4862a")
WARN_BG      = colors.HexColor("#fdf3dc")
DANGER_BG    = colors.HexColor("#fce9e0")
DANGER_TEXT  = colors.HexColor("#8b3a1e")
TEXT_PRIMARY = colors.HexColor("#1e2a1e")
TEXT_SEC     = colors.HexColor("#4a5a48")
TEXT_MUTED   = colors.HexColor("#7a8a78")
BG_PAGE      = colors.HexColor("#f7f2ea")
BORDER       = colors.HexColor("#d4ccc0")
WHITE        = colors.white


def _fmt(n) -> str:
    """Format a number with thousand separators."""
    try:
        return f"${int(n):,}"
    except (TypeError, ValueError):
        return "—"


def _pct(n) -> str:
    try:
        return f"{float(n):.1f}%"
    except (TypeError, ValueError):
        return "—"


# ── Styles ────────────────────────────────────────────────────────────────

def _make_styles():
    styles = getSampleStyleSheet()

    base = ParagraphStyle(
        "base",
        fontName="Times-Roman",
        fontSize=10,
        leading=15,
        textColor=TEXT_PRIMARY,
    )

    title = ParagraphStyle(
        "title",
        parent=base,
        fontName="Times-Bold",
        fontSize=22,
        leading=26,
        textColor=GREEN_DARK,
        spaceAfter=4,
    )

    subtitle = ParagraphStyle(
        "subtitle",
        parent=base,
        fontSize=11,
        textColor=TEXT_SEC,
        spaceAfter=2,
    )

    section_head = ParagraphStyle(
        "section_head",
        parent=base,
        fontName="Times-Bold",
        fontSize=13,
        leading=17,
        textColor=GREEN_DARK,
        spaceBefore=16,
        spaceAfter=6,
    )

    crop_head = ParagraphStyle(
        "crop_head",
        parent=base,
        fontName="Times-Bold",
        fontSize=11,
        textColor=GREEN_DARK,
        spaceAfter=4,
    )

    body = ParagraphStyle(
        "body",
        parent=base,
        fontSize=10,
        leading=15,
        textColor=TEXT_SEC,
        spaceAfter=6,
    )

    label = ParagraphStyle(
        "label",
        parent=base,
        fontSize=8,
        leading=11,
        textColor=TEXT_MUTED,
    )

    hedge_text = ParagraphStyle(
        "hedge_text",
        parent=body,
        fontSize=9.5,
        textColor=TEXT_SEC,
        leftIndent=8,
        borderPad=4,
    )

    narrative = ParagraphStyle(
        "narrative",
        parent=base,
        fontSize=10.5,
        leading=17,
        textColor=TEXT_PRIMARY,
        leftIndent=12,
        rightIndent=4,
        borderPad=6,
    )

    footer_style = ParagraphStyle(
        "footer_style",
        parent=base,
        fontSize=8,
        textColor=TEXT_MUTED,
        alignment=TA_CENTER,
    )

    return {
        "title": title, "subtitle": subtitle, "section_head": section_head,
        "crop_head": crop_head, "body": body, "label": label,
        "hedge_text": hedge_text, "narrative": narrative, "footer": footer_style,
    }


# ── Page template ─────────────────────────────────────────────────────────

def _header_footer(canvas, doc):
    """Draw page header stripe and footer on every page."""
    w, h = letter
    canvas.saveState()

    # Header bar
    canvas.setFillColor(GREEN_DARK)
    canvas.rect(0, h - 0.45 * inch, w, 0.45 * inch, fill=1, stroke=0)
    canvas.setFillColor(WHITE)
    canvas.setFont("Times-Bold", 11)
    canvas.drawString(0.6 * inch, h - 0.30 * inch, "AgriShield")
    canvas.setFont("Times-Roman", 9)
    canvas.setFillColor(GREEN_LIGHT)
    canvas.drawRightString(w - 0.6 * inch, h - 0.30 * inch, "Climate Risk & Hedge Report")

    # Footer
    canvas.setFillColor(TEXT_MUTED)
    canvas.setFont("Times-Roman", 8)
    canvas.drawCentredString(w / 2, 0.35 * inch, f"Page {doc.page}  •  Generated {datetime.now().strftime('%B %d, %Y')}")

    canvas.restoreState()


# ── Section: Farm header ──────────────────────────────────────────────────

def _section_farm_header(result: dict, styles: dict) -> list:
    farm    = result.get("farm_input", {}).get("farm", {})
    crops   = result.get("farm_input", {}).get("crops", [])
    frs     = result.get("risk_result", {}).get("farm_risk_summary", {})
    seasonal = result.get("seasonal_outlook", {})
    meta    = result.get("meta", {})

    loc = farm.get("location_raw") or f"{farm.get('lat',''):.4f}, {farm.get('lon',''):.4f}"
    crop_str = ", ".join(
        f"{c['name']} ({int(c['acres'])} ac)" for c in crops
    )
    enso_desc = seasonal.get("enso", {}).get("enso_desc", "Neutral")

    flowables = [
        Paragraph(f"Farm Risk Report — {loc}", styles["title"]),
        Paragraph(crop_str, styles["subtitle"]),
        Paragraph(
            f"ENSO: {enso_desc}  •  Generated in {meta.get('elapsed_seconds', '—')}s  •  "
            f"IBM Granite: {'active' if meta.get('granite_call1_used') or meta.get('granite_call2_used') else 'fallback mode'}",
            styles["label"],
        ),
        Spacer(1, 8),
        HRFlowable(width="100%", thickness=1, color=GREEN_LIGHT, spaceAfter=10),
    ]

    # Summary numbers table
    sev   = frs.get("overall_severity", "unknown").title()
    table_data = [
        ["Expected Revenue", "Median Loss (P50)", "Worst-Case (P10)", "Risk Level"],
        [
            _fmt(frs.get("total_full_revenue")),
            f"{_fmt(frs.get('total_p50_loss'))} ({_pct(frs.get('total_p50_loss_pct'))})",
            f"{_fmt(frs.get('total_p10_loss'))} ({_pct(frs.get('total_p10_loss_pct'))})",
            sev,
        ],
    ]

    sev_bg = {
        "Critical": DANGER_BG, "High": colors.HexColor("#fdf0e0"),
        "Moderate": WARN_BG,   "Low": GREEN_WASH,
    }.get(sev, EARTH_WASH)

    t = Table(table_data, colWidths=[1.6 * inch, 2.0 * inch, 2.0 * inch, 1.4 * inch])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), GREEN_DARK),
        ("TEXTCOLOR",  (0, 0), (-1, 0), WHITE),
        ("FONTNAME",   (0, 0), (-1, 0), "Times-Bold"),
        ("FONTSIZE",   (0, 0), (-1, 0), 8),
        ("FONTNAME",   (0, 1), (-1, 1), "Times-Roman"),
        ("FONTSIZE",   (0, 1), (-1, 1), 10),
        ("BACKGROUND", (0, 1), (-1, 1), sev_bg),
        ("BACKGROUND", (3, 1), (3, 1), sev_bg),
        ("ALIGN",      (0, 0), (-1, -1), "CENTER"),
        ("VALIGN",     (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [None, None]),
        ("GRID",       (0, 0), (-1, -1), 0.5, BORDER),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    flowables.append(t)
    return flowables


# ── Section: Hedging summary ──────────────────────────────────────────────

def _section_summary(result: dict, styles: dict) -> list:
    advisory = result.get("advisory", {})
    short    = advisory.get("short_summary", "")
    detailed = advisory.get("detailed_narrative", "")

    flowables = [
        Paragraph("What You Should Do This Season", styles["section_head"]),
        HRFlowable(width="100%", thickness=0.5, color=GREEN_LIGHT, spaceAfter=8),
    ]

    # Summary box
    if short:
        summary_table = Table(
            [[Paragraph(short, styles["narrative"])]],
            colWidths=[7.0 * inch],
        )
        summary_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), GREEN_WASH),
            ("LEFTPADDING",  (0, 0), (-1, -1), 14),
            ("RIGHTPADDING", (0, 0), (-1, -1), 10),
            ("TOPPADDING",   (0, 0), (-1, -1), 10),
            ("BOTTOMPADDING",(0, 0), (-1, -1), 10),
            ("BOX",          (0, 0), (-1, -1), 1, GREEN_MUTED),
            ("LINEBEFORE",   (0, 0), (0, -1), 3, GREEN_MID),
        ]))
        flowables.append(summary_table)
        flowables.append(Spacer(1, 10))

    if detailed:
        flowables.append(Paragraph("Detailed Advisory", styles["section_head"]))
        for para in detailed.split("\n\n"):
            para = para.strip()
            if not para:
                continue
            # Section headers within detailed text
            if para.isupper() or (len(para) < 40 and "\n" not in para and not para[0].isdigit()):
                flowables.append(Paragraph(para, styles["crop_head"]))
            else:
                flowables.append(Paragraph(para, styles["body"]))

    return flowables


# ── Section: Per-crop breakdown ───────────────────────────────────────────

def _section_crops(result: dict, styles: dict) -> list:
    crops  = result.get("farm_input", {}).get("crops", [])
    ls_all = result.get("risk_result", {}).get("loss_scenarios", {})
    mc_all = result.get("monte_carlo", {})
    recs   = result.get("hedge_result", {}).get("recommendations", [])

    flowables = [
        Paragraph("Crop-by-Crop Risk Analysis", styles["section_head"]),
        HRFlowable(width="100%", thickness=0.5, color=GREEN_LIGHT, spaceAfter=8),
    ]

    for crop_entry in crops:
        name   = crop_entry["name"]
        acres  = int(crop_entry.get("acres", 0))
        ls     = ls_all.get(name, {})
        sc     = ls.get("scenarios", {})
        p50    = sc.get("median_p50", {})
        p10    = sc.get("bad_year_p10", {})
        p90    = sc.get("good_year_p90", {})
        mc     = mc_all.get(name, {}).get("yield_distribution", {})
        sev    = ls.get("severity", "unknown").title()
        rec    = next((r for r in recs if r["crop"] == name), {})
        full_rev = ls.get("full_revenue", 0)

        sev_bg = {
            "Critical": DANGER_BG, "High": colors.HexColor("#fdf0e0"),
            "Moderate": WARN_BG,   "Low": GREEN_WASH,
        }.get(sev, EARTH_WASH)
        sev_color = {
            "Critical": DANGER_TEXT, "High": EARTH_DARK,
            "Moderate": AMBER,       "Low": GREEN_MID,
        }.get(sev, TEXT_MUTED)

        # Crop header row
        header_data = [[
            Paragraph(f"<b>{name}</b>", styles["crop_head"]),
            Paragraph(f"{acres} acres", styles["label"]),
            Paragraph(f"<b>{sev} Risk</b>", ParagraphStyle(
                "sev", parent=styles["label"], textColor=sev_color, fontSize=9,
                fontName="Times-Bold",
            )),
        ]]
        header_t = Table(header_data, colWidths=[3.5 * inch, 2.0 * inch, 1.5 * inch])
        header_t.setStyle(TableStyle([
            ("BACKGROUND",     (0, 0), (-1, -1), sev_bg),
            ("TOPPADDING",     (0, 0), (-1, -1), 7),
            ("BOTTOMPADDING",  (0, 0), (-1, -1), 7),
            ("LEFTPADDING",    (0, 0), (0, -1), 10),
            ("ALIGN",          (1, 0), (-1, -1), "RIGHT"),
            ("RIGHTPADDING",   (-1, 0), (-1, -1), 10),
            ("LINEBELOW",      (0, 0), (-1, -1), 0.5, BORDER),
        ]))

        # Financial table
        fin_data = [
            ["", "Revenue", "Yield %", "Loss"],
            ["Full season",
             _fmt(full_rev),
             "100%",
             "—"],
            ["Good year (P90)",
             _fmt(p90.get("revenue", 0)),
             f"{p90.get('yield_pct', '—')}%",
             f"({_fmt(p90.get('loss', 0))})"],
            ["Median (P50)",
             _fmt(p50.get("revenue", 0)),
             f"{p50.get('yield_pct', '—')}%",
             _fmt(p50.get("loss", 0))],
            ["Bad year (P10)",
             _fmt(p10.get("revenue", 0)),
             f"{p10.get('yield_pct', '—')}%",
             _fmt(p10.get("loss", 0))],
        ]

        if mc:
            fin_data.append([
                "Distribution (P10 / P50 / P90)",
                f"{mc.get('p10', '—')}% / {mc.get('p50', '—')}% / {mc.get('p90', '—')}%",
                f"σ = {mc.get('std_dev', '—')}%",
                f"skew = {mc.get('skewness', '—')}",
            ])

        fin_t = Table(
            fin_data,
            colWidths=[2.0 * inch, 1.8 * inch, 1.2 * inch, 2.0 * inch],
        )
        fin_t.setStyle(TableStyle([
            ("FONTNAME",    (0, 0), (-1, 0), "Times-Bold"),
            ("FONTSIZE",    (0, 0), (-1, 0), 8),
            ("FONTNAME",    (0, 1), (-1, -1), "Times-Roman"),
            ("FONTSIZE",    (0, 1), (-1, -1), 9),
            ("TEXTCOLOR",   (0, 0), (-1, 0), TEXT_MUTED),
            ("BACKGROUND",  (0, 0), (-1, 0), colors.HexColor("#f0ece5")),
            ("BACKGROUND",  (0, 3), (-1, 3), WARN_BG),   # P50
            ("BACKGROUND",  (0, 4), (-1, 4), DANGER_BG), # P10
            ("ALIGN",       (1, 0), (-1, -1), "RIGHT"),
            ("LEFTPADDING", (0, 0), (0, -1), 8),
            ("TOPPADDING",  (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("LINEBELOW",   (0, 0), (-1, 0), 0.5, BORDER),
        ]))

        # Hedge recommendation
        hedge_text = rec.get("recommendation", "")
        items = [header_t, Spacer(1, 1), fin_t]
        if hedge_text:
            items.append(Spacer(1, 6))
            items.append(Paragraph(
                f"<b>Hedge recommendation:</b> {hedge_text}",
                styles["hedge_text"],
            ))

        block = KeepTogether([
            *items,
            Spacer(1, 10),
        ])
        flowables.append(block)

    return flowables


# ── Section: Hedge summary ────────────────────────────────────────────────

def _section_hedge_summary(result: dict, styles: dict) -> list:
    fhs  = result.get("hedge_result", {}).get("farm_hedge_summary", {})
    recs = result.get("hedge_result", {}).get("recommendations", [])

    flowables = [
        Paragraph("Hedge Portfolio Summary", styles["section_head"]),
        HRFlowable(width="100%", thickness=0.5, color=GREEN_LIGHT, spaceAfter=8),
    ]

    table_data = [["Crop", "Strategy", "Contracts", "Strike", "Premium", "Cost %"]]
    for rec in recs:
        if not rec.get("has_futures"):
            table_data.append([
                rec["crop"], "No futures", "—", "—", "—", "—",
            ])
            continue
        siz = rec.get("sizing", {})
        inp = rec.get("inputs", {})
        table_data.append([
            rec["crop"],
            "Protective Put" if rec.get("strategy") == "protective_put" else rec.get("strategy", "—"),
            str(siz.get("contracts_recommended", "—")),
            f"${inp.get('strike_usd', '—')}",
            f"${siz.get('total_premium_usd', 0):,.0f}",
            f"{siz.get('hedge_cost_pct_revenue', 0):.1f}%",
        ])

    # Totals row
    table_data.append([
        "TOTAL", "", "",  "",
        f"${fhs.get('total_hedge_premium', 0):,.0f}",
        f"{fhs.get('hedge_cost_pct_revenue', 0):.1f}%",
    ])

    t = Table(table_data, colWidths=[1.2 * inch, 1.5 * inch, .9 * inch, .9 * inch, 1.3 * inch, .8 * inch])
    t.setStyle(TableStyle([
        ("BACKGROUND",  (0, 0), (-1, 0), GREEN_DARK),
        ("TEXTCOLOR",   (0, 0), (-1, 0), WHITE),
        ("FONTNAME",    (0, 0), (-1, 0), "Times-Bold"),
        ("FONTSIZE",    (0, 0), (-1, 0), 8),
        ("FONTNAME",    (0, 1), (-1, -1), "Times-Roman"),
        ("FONTSIZE",    (0, 1), (-1, -1), 9),
        ("ALIGN",       (1, 0), (-1, -1), "CENTER"),
        ("LEFTPADDING", (0, 0), (0, -1), 8),
        ("TOPPADDING",  (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("ROWBACKGROUNDS", (0, 1), (-1, -2), [WHITE, colors.HexColor("#f7f4ef")]),
        ("BACKGROUND",  (0, -1), (-1, -1), EARTH_WASH),
        ("FONTNAME",    (0, -1), (-1, -1), "Times-Bold"),
        ("LINEABOVE",   (0, -1), (-1, -1), 1, BORDER),
        ("GRID",        (0, 0), (-1, -1), 0.4, BORDER),
    ]))
    flowables.append(t)

    # Budget note
    budget = fhs.get("total_budget", 0)
    premium = fhs.get("total_hedge_premium", 0)
    if budget > 0:
        used_pct = premium / budget * 100 if budget > 0 else 0
        flowables.append(Spacer(1, 6))
        flowables.append(Paragraph(
            f"Your hedge budget: ${budget:,.0f}  •  "
            f"Recommended premium: ${premium:,.0f}  •  "
            f"Budget utilization: {used_pct:.0f}%",
            styles["label"],
        ))

    return flowables


# ── Section: Seasonal context ─────────────────────────────────────────────

def _section_seasonal(result: dict, styles: dict) -> list:
    seasonal = result.get("seasonal_outlook", {})
    enso     = seasonal.get("enso", {})
    outlook  = seasonal.get("seasonal_outlook", {})
    fcst     = seasonal.get("forecast_14d", {})
    anom     = seasonal.get("current_anomaly", {})

    flowables = [
        Paragraph("Seasonal Climate Context", styles["section_head"]),
        HRFlowable(width="100%", thickness=0.5, color=GREEN_LIGHT, spaceAfter=8),
    ]

    ctx_data = [
        ["ENSO Phase", enso.get("enso_desc", "—"), "ONI", str(enso.get("oni_value", "—"))],
        ["30-day anomaly", anom.get("anomaly_signal", "—").title(),
         "Precip anomaly", f"{anom.get('precip_anomaly_pct', '—')}%"],
        ["Temp anomaly", f"+{anom.get('temp_anomaly_c', '—')}°C",
         "Current dry streak", f"{anom.get('current_dry_streak', '—')} days"],
        ["14-day precip", f"{fcst.get('total_precip_mm', '—')} mm",
         "Heat stress days", str(fcst.get("heat_stress_days", "—"))],
        ["Temp > normal prob", f"{outlook.get('temp_above_normal_prob_pct', '—')}%",
         "Precip < normal prob", f"{outlook.get('precip_below_normal_prob_pct', '—')}%"],
        ["Drought risk", "Elevated" if outlook.get("drought_risk_elevated") else "Normal", "", ""],
    ]

    ct = Table(ctx_data, colWidths=[2.0 * inch, 1.6 * inch, 2.0 * inch, 1.4 * inch])
    ct.setStyle(TableStyle([
        ("FONTNAME",    (0, 0), (0, -1), "Times-Bold"),
        ("FONTNAME",    (2, 0), (2, -1), "Times-Bold"),
        ("FONTSIZE",    (0, 0), (-1, -1), 9),
        ("TEXTCOLOR",   (0, 0), (0, -1), TEXT_SEC),
        ("TEXTCOLOR",   (2, 0), (2, -1), TEXT_SEC),
        ("TOPPADDING",  (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (0, -1), 4),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [WHITE, colors.HexColor("#f7f4ef")]),
        ("GRID",        (0, 0), (-1, -1), 0.3, BORDER),
    ]))
    flowables.append(ct)

    narrative = outlook.get("risk_narrative", "")
    if narrative:
        flowables.append(Spacer(1, 8))
        flowables.append(Paragraph(narrative, styles["body"]))

    return flowables


# ── Main builder ──────────────────────────────────────────────────────────

def build_pdf(result: dict) -> bytes:
    """
    Build the full PDF report from pipeline result dict.
    Returns raw PDF bytes.
    """
    buf = io.BytesIO()
    margin = 0.6 * inch
    doc = BaseDocTemplate(
        buf,
        pagesize=letter,
        leftMargin=margin,
        rightMargin=margin,
        topMargin=0.7 * inch,
        bottomMargin=0.6 * inch,
        title="AgriShield Risk Report",
        author="AgriShield",
    )

    frame = Frame(
        doc.leftMargin, doc.bottomMargin,
        doc.width, doc.height,
        id="main",
    )
    template = PageTemplate(id="main", frames=[frame], onPage=_header_footer)
    doc.addPageTemplates([template])

    styles = _make_styles()

    story = []
    story += _section_farm_header(result, styles)
    story.append(Spacer(1, 12))
    story += _section_summary(result, styles)
    story.append(Spacer(1, 8))
    story += _section_crops(result, styles)
    story += _section_hedge_summary(result, styles)
    story.append(Spacer(1, 8))
    story += _section_seasonal(result, styles)

    doc.build(story)
    return buf.getvalue()
