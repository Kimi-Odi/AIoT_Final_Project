# pdf_export.py
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.units import cm

# ⭐ 使用 Windows 內建「微軟正黑體」
pdfmetrics.registerFont(TTFont("MSJH", "C:/Windows/Fonts/msjh.ttc"))

def export_pdf(filename: str, report_text: str):
    styles = getSampleStyleSheet()

    chinese_style = ParagraphStyle(
        "Chinese",
        parent=styles["Normal"],
        fontName="MSJH",
        fontSize=11,
        leading=14,
    )

    doc = SimpleDocTemplate(filename, pagesize=A4)
    story = []

    for line in report_text.split("\n"):
        clean_line = line.replace("**", "")
        story.append(Paragraph(clean_line, chinese_style))
        story.append(Spacer(1, 0.35 * cm))

    doc.build(story)
