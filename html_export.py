# html_export.py
from jinja2 import Template

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="zh-Hant">
<head>
<meta charset="UTF-8">
<title>AI 面試報告</title>
<style>
body { font-family: Arial, sans-serif; padding: 40px; }
h1 { color: #2c3e50; }
h2 { margin-top: 30px; }
pre { background-color: #f5f5f5; padding: 10px; border-radius: 5px; }
</style>
</head>
<body>

<h1>AI 虛擬面試官 — 面試報告</h1>

<pre>{{ content }}</pre>

</body>
</html>
"""

def export_html(report_text: str) -> str:
    """
    將報告內容嵌入 HTML 並回傳 HTML 字串
    """
    template = Template(HTML_TEMPLATE)
    html = template.render(content=report_text)
    return html
