"""HTML template for model predictions report."""

from datetime import datetime
from string import Template

HTML_TEMPLATE = Template("""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Model Predictions Report</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0d1117; color: #e6edf3; font-size: 16px; }
.container { max-width: 1000px; margin: 0 auto; padding: 24px; }
h1 { color: #58a6ff; margin-bottom: 24px; font-size: 28px; }
.tabs { display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 24px; border-bottom: 1px solid #30363d; padding-bottom: 16px; }
.tab { padding: 10px 20px; background: #21262d; border: 1px solid #30363d; border-radius: 6px; cursor: pointer; color: #8b949e; font-size: 16px; font-weight: 500; transition: all 0.2s; }
.tab:hover { background: #30363d; color: #e6edf3; }
.tab.active { background: #238636; border-color: #238636; color: #fff; }
.date-content { display: none; }
.date-content.active { display: block; }
.symbol-tabs { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 20px; }
.symbol-tab { padding: 8px 16px; background: #161b22; border: 1px solid #30363d; border-radius: 6px; cursor: pointer; color: #8b949e; font-size: 15px; font-weight: 500; }
.symbol-tab:hover { background: #21262d; }
.symbol-tab.active { background: #1f6feb; border-color: #1f6feb; color: #fff; }
.symbol-content { display: none; }
.symbol-content.active { display: block; }
table { width: 100%; border-collapse: collapse; background: #161b22; border-radius: 8px; overflow: hidden; margin-bottom: 24px; }
th { background: #21262d; color: #8b949e; font-weight: 600; text-align: center; padding: 12px 24px; font-size: 12px; text-transform: uppercase; min-width: 100px; border-right: 1px solid #30363d; }
th.polymarket { background: #1c2128; color: #a371f7; }
th:first-child { text-align: left; }
td { padding: 10px 24px; border-top: 1px solid #21262d; font-size: 15px; text-align: center; border-right: 1px solid #21262d; }
td:first-child { text-align: left; }
td.polymarket { background: rgba(163, 113, 247, 0.08); }
tr:hover td { background: #1c2128; }
tr:hover td.polymarket { background: rgba(163, 113, 247, 0.15); }
.prob { font-size: 15px; font-weight: 500; }
.prob-high { color: #3fb950; }
.prob-mid { color: #d29922; }
.prob-low { color: #f85149; }
.na { color: #484f58; }
.strike { font-weight: 600; color: #58a6ff; font-size: 15px; }
.conf { font-size: 12px; color: #8b949e; }
.conf-high { color: #3fb950; }
.conf-mid { color: #d29922; }
.conf-low { color: #f85149; }
.meta { color: #8b949e; font-size: 14px; margin-bottom: 20px; }
</style>
</head>
<body>
<div class="container">
<h1>📊 Model Predictions Report</h1>
<p class="meta">Generated: $generated_time</p>
<div class="tabs" id="date-tabs">
$date_tabs
</div>
$date_content
<script>
function showDate(date) {
    document.querySelectorAll('.date-content').forEach(el => el.classList.remove('active'));
    document.querySelectorAll('#date-tabs .tab').forEach(el => el.classList.remove('active'));
    document.getElementById('date-' + date).classList.add('active');
    document.querySelector('#date-tabs .tab[onclick*="' + date + '"]').classList.add('active');
}
function showSymbol(date, symbol) {
    const container = document.getElementById('date-' + date);
    container.querySelectorAll('.symbol-content').forEach(el => el.classList.remove('active'));
    container.querySelectorAll('.symbol-tab').forEach(el => el.classList.remove('active'));
    document.getElementById('symbol-' + date + '-' + symbol).classList.add('active');
    container.querySelector('.symbol-tab[onclick*="' + symbol + '"]').classList.add('active');
}
</script>
</div>
</body>
</html>""")


def render_html(date_tabs: str, date_content: str) -> str:
    """Render the HTML template with the given content."""
    return HTML_TEMPLATE.substitute(
        generated_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        date_tabs=date_tabs,
        date_content=date_content,
    )
