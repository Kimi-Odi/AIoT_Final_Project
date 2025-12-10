import io
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.units import cm
from reportlab.lib import colors

def export_pdf(filename: str, data: dict):
    """
    將包含所有分析結果的字典匯出為一個精緻的 PDF 報告。
    """
    # --- 註冊字型 ---
    try:
        pdfmetrics.registerFont(TTFont("MSJH", "/System/Library/Fonts/STHeiti Medium.ttc"))
    except Exception:
        raise RuntimeError("無法找到字型 '/System/Library/Fonts/STHeiti Medium.ttc'。請確認該字型存在或修改 pdf_export.py 中的路徑。")

    # --- 樣式定義 ---
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "Title", parent=styles["h1"], fontName="MSJH", fontSize=24, alignment=1, spaceAfter=20
    )
    heading1_style = ParagraphStyle(
        "Heading1", parent=styles["h2"], fontName="MSJH", fontSize=18, spaceAfter=10, spaceBefore=12
    )
    heading2_style = ParagraphStyle(
        "Heading2", parent=styles["h3"], fontName="MSJH", fontSize=14, spaceAfter=8, spaceBefore=10
    )
    body_style = ParagraphStyle(
        "Body", parent=styles["Normal"], fontName="MSJH", fontSize=11, leading=16, spaceAfter=10
    )
    bullet_style = ParagraphStyle(
        "Bullet", parent=body_style, firstLineIndent=0, leftIndent=20
    )

    # --- PDF 文件結構 ---
    doc = SimpleDocTemplate(filename, pagesize=A4, leftMargin=2*cm, rightMargin=2*cm, topMargin=2*cm, bottomMargin=2*cm)
    story = []

    # 1. 報告標題
    story.append(Paragraph("AI 虛擬面試報告", title_style))

    # 2. 基本資訊
    story.append(Paragraph("基本資訊", heading1_style))
    info_data = [
        ["受試者 ID:", data.get("candidate_id", "N/A")],
        ["應徵職缺:", data.get("job_role", "N/A")],
        ["面試日期:", data.get("timestamp", "N/A")],
    ]
    info_table = Table(info_data, colWidths=[3*cm, 12*cm])
    info_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (-1, -1), 'MSJH'),
        ('FONTSIZE', (0, 0), (-1, -1), 11),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(info_table)
    story.append(Spacer(1, 1*cm))

    # 3. 整體評分與雷達圖
    story.append(Paragraph("整體評分", heading1_style))
    
    scores = data.get("overall_scores", {})
    score_data = [
        ["技術能力 (Technical)", f"{scores.get('technical', 0)} / 5"],
        ["溝通表達 (Communication)", f"{scores.get('communication', 0)} / 5"],
        ["回答結構 (Structure)", f"{scores.get('structure', 0)} / 5"],
        ["職缺相關性 (Relevance)", f"{scores.get('relevance', 0)} / 5"],
        ["問題解決 (Problem Solving)", f"{scores.get('problem_solving', 0)} / 5"],
        ["個人潛力 (Growth Potential)", f"{scores.get('growth_potential', 0)} / 5"],
    ]
    score_table = Table(score_data, colWidths=[6*cm, 3*cm])
    score_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (-1, -1), 'MSJH'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
        ('ALIGN', (1, 0), (1, -1), 'CENTER'),
    ]))
    
    radar_image_bytes = data.get("radar_chart_image")
    if radar_image_bytes:
        radar_image = Image(io.BytesIO(radar_image_bytes), width=8*cm, height=8*cm)
        combined_table = Table([[score_table, radar_image]], colWidths=[9.5*cm, 8.5*cm], vAlign='TOP')
        story.append(combined_table)
    else:
        story.append(score_table)

    story.append(Spacer(1, 1*cm))

    # 4. 整體評論
    story.append(Paragraph("整體評論", heading2_style))
    story.append(Paragraph(data.get("summary", "N/A").replace("\n", "<br/>"), body_style))
    story.append(Spacer(1, 1*cm))

    # 5. 逐題回饋
    story.append(PageBreak())
    story.append(Paragraph("逐題詳細回饋", heading1_style))
    
    qa_list = data.get("qa_list", [])
    voice_list = data.get("voice_analysis_results", [])
    per_question_feedback = data.get("per_question_feedback", [])

    for i, qa in enumerate(qa_list):
        story.append(Paragraph(f"第 {i+1} 題", heading2_style))
        story.append(Paragraph(f"<b>題目：</b> {qa['question']}", body_style))
        story.append(Paragraph(f"<b>回答：</b> {qa['answer']}", body_style))
        
        # 逐題回饋 (從 per_question_feedback 取得)
        if i < len(per_question_feedback):
            item = per_question_feedback[i]
            if "score" in item:
                s = item['score']
                score_text = f"<b>分數：</b> 技 {s['technical']}/5, 達 {s['communication']}/5, 構 {s['structure']}/5, 關 {s['relevance']}/5, 解 {s['problem_solving']}/5, 潛 {s['growth_potential']}/5"
                story.append(Paragraph(score_text, body_style))
            
            if "feedback" in item:
                story.append(Paragraph(f"<b>AI 回饋：</b> {item['feedback']}", body_style))

        # 語音分析回饋
        if i < len(voice_list) and voice_list[i]:
            voice_result = voice_list[i]
            if "error" not in voice_result:
                story.append(Paragraph("<b>語音特徵分析：</b>", body_style))
                story.append(Paragraph(f"• 音調：{voice_result['pitch']}", bullet_style))
                story.append(Paragraph(f"• 音量：{voice_result['volume']}", bullet_style))
                story.append(Paragraph(f"• 語速：{voice_result['speech_rate']}", bullet_style))

        story.append(Spacer(1, 0.5*cm))

    # --- 產生 PDF ---
    doc.build(story)