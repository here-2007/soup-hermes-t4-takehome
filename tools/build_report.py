#!/usr/bin/env python3
"""Build the two-page report and figures from the report source and CPU review."""

import csv
import html
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pypdf import PdfReader
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


ROOT = Path(__file__).resolve().parents[1]
NAVY = "#18344A"
TEAL = "#12746F"
RUST = "#A33F32"
GREY = "#596673"


def inline(text):
    value = html.escape(text)
    value = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", value)
    value = re.sub(r"`([^`]+)`", r'<font name="Courier" size="8">\1</font>', value)
    return value


def build_pdf():
    styles = {
        "body": ParagraphStyle("body", fontName="Helvetica", fontSize=9.15, leading=11.65,
                               spaceAfter=5, textColor=colors.HexColor(NAVY)),
        "title": ParagraphStyle("title", fontName="Helvetica-Bold", fontSize=22, leading=25,
                                spaceAfter=7, textColor=colors.HexColor(NAVY)),
        "heading": ParagraphStyle("heading", fontName="Helvetica-Bold", fontSize=11.2, leading=13,
                                  spaceBefore=8, spaceAfter=5, textColor=colors.HexColor(TEAL), keepWithNext=True),
        "cell": ParagraphStyle("cell", fontName="Helvetica", fontSize=8.1, leading=10.1,
                               textColor=colors.HexColor(NAVY)),
        "headcell": ParagraphStyle("headcell", fontName="Helvetica-Bold", fontSize=8.1, leading=10.1,
                                   textColor=colors.white),
        "source": ParagraphStyle("source", fontName="Helvetica", fontSize=7.1, leading=9,
                                 spaceBefore=3, textColor=colors.HexColor(GREY)),
    }
    width = A4[0] - 80
    lines = (ROOT / "REPORT.md").read_text().splitlines()
    story = []
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        if not line:
            index += 1
            continue
        if line == "<!-- PAGE BREAK -->":
            story.append(PageBreak()); index += 1; continue
        if line.startswith("# "):
            story.append(Paragraph(inline(line[2:]), styles["title"])); index += 1; continue
        if line.startswith("## "):
            story.append(Paragraph(inline(line[3:]), styles["heading"])); index += 1; continue
        if line.startswith("|"):
            rows = []
            while index < len(lines) and lines[index].strip().startswith("|"):
                row = [x.strip() for x in lines[index].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-+:?", x) for x in row):
                    rows.append(row)
                index += 1
            count = len(rows[0])
            fractions = {2: [.19, .81], 3: [.30, .56, .14], 4: [.37, .21, .21, .21], 5: [.16, .20, .23, .19, .22]}[count]
            cells = [[Paragraph(inline(value), styles["headcell" if n == 0 else "cell"])
                      for value in row] for n, row in enumerate(rows)]
            table = Table(cells, colWidths=[width*f for f in fractions], hAlign="LEFT", repeatRows=1)
            table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(NAVY)),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#F0F5F7"), colors.white]),
                ("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6), ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("LINEBELOW", (0, 0), (-1, 0), .3, colors.HexColor(NAVY)),
            ]))
            story.extend([table, Spacer(1, 6)])
            continue
        paragraph = [lines[index]]
        index += 1
        while index < len(lines) and lines[index].strip() and not lines[index].strip().startswith(("#", "|", "<!--")):
            paragraph.append(lines[index]); index += 1
        text = " ".join(paragraph)
        style = styles["source"] if text.startswith("Evidence:") else styles["body"]
        rendered = "".join(inline(part.strip()) + ("<br/>" if part.endswith("  ") else " ")
                           for part in paragraph).strip()
        story.append(Paragraph(rendered, style))

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#CBD7DF")); canvas.setLineWidth(.5)
        canvas.line(40, 30, A4[0]-40, 30)
        canvas.setFont("Helvetica", 7); canvas.setFillColor(colors.HexColor(GREY))
        canvas.drawString(40, 19, "PERNAV JAIN  |  SOUP AI ENGINEER TAKE-HOME  |  10 NOV 2026")
        canvas.drawRightString(A4[0]-40, 19, f"{doc.page} / 2")
        canvas.restoreState()

    output = ROOT / "report.pdf"
    document = SimpleDocTemplate(str(output), pagesize=A4, leftMargin=40, rightMargin=40,
                                 topMargin=33, bottomMargin=40, title="Tool-call gains did not survive NF4",
                                 author="Pernav Jain", subject="Soup AI Engineer take-home: DON'T SHIP")
    document.build(story, onFirstPage=footer, onLaterPages=footer)
    pages = PdfReader(output).pages
    if len(pages) != 2:
        raise ValueError(f"Report must contain two pages; rendered {len(pages)}")
    print(f"Built {output}: two pages")


def build_figures():
    data = json.loads((ROOT / "analysis/review.json").read_text())
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.labelcolor": NAVY, "text.color": NAVY,
                         "xtick.color": GREY, "ytick.color": GREY, "axes.edgecolor": "#C5D1D8"})
    figure, axes = plt.subplots(1, 2, figsize=(11, 3.4), gridspec_kw={"width_ratios": [1.05, 1.3]})
    variants = ["B", "M", "Q"]
    accuracy = [data["metrics"][v]["complete"]*100 for v in variants]
    bars = axes[0].bar(["Base B", "Dense M", "NF4 Q"], accuracy, color=[GREY, TEAL, RUST], width=.58)
    axes[0].set_ylim(0, 75); axes[0].set_ylabel("Exact reference correctness (%)")
    axes[0].set_title("150 held-out turns", loc="left", fontweight="bold", fontsize=12)
    for bar, value, variant in zip(bars, accuracy, variants):
        axes[0].text(bar.get_x()+bar.get_width()/2, value+1.8,
                     f"{value:.1f}%\n{round(value*1.5)}/150", ha="center", va="bottom", fontsize=10)
    names = ["M minus B", "Q minus M", "Q minus B"]
    keys = ["M_minus_B", "Q_minus_M", "Q_minus_B"]
    for y, key, color in zip([2, 1, 0], keys, [TEAL, RUST, GREY]):
        pair = data["paired_bootstrap"][key]; point = pair["delta"]*100
        low, high = [x*100 for x in pair["ci95"]]
        axes[1].errorbar(point, y, xerr=[[point-low], [high-point]], fmt="o", color=color,
                         capsize=5, linewidth=2, markersize=7)
        axes[1].annotate(f"{point:+.2f} [{low:+.2f}, {high:+.2f}]", (point, y),
                         xytext=(0, 15), textcoords="offset points", ha="center", fontsize=9)
    axes[1].axvline(0, color="#ABB9C3", linewidth=1, linestyle="--")
    axes[1].set_yticks([2, 1, 0], names); axes[1].set_ylim(-.5, 2.8); axes[1].set_xlim(-23, 24)
    axes[1].set_xlabel("Difference in percentage points; 95% paired bootstrap CI")
    axes[1].set_title("Gains and quantization loss", loc="left", fontweight="bold", fontsize=12)
    figure.tight_layout(w_pad=3)
    for suffix in ("png", "svg"):
        figure.savefig(ROOT/f"figures/artifact-comparison.{suffix}", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    with (ROOT/"analysis/task-results.csv").open("w", newline="") as handle:
        writer = csv.writer(handle); writer.writerow(["artifact", "correct", "n", "exact_accuracy", "parse_rate", "assistant_nll"])
        for variant in variants:
            result = data["metrics"][variant]
            writer.writerow([variant, round(result["complete"]*result["n"]), result["n"], result["complete"], result["parse_ok"], result["mean_assistant_nll"]])


if __name__ == "__main__":
    build_pdf()
    build_figures()
