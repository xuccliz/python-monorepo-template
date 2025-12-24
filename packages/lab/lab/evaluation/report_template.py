"""HTML template for evaluation report."""

from datetime import datetime
from string import Template

HTML_TEMPLATE = Template("""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Model Evaluation Report</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0d1117; color: #e6edf3; font-size: 16px; }
.container { max-width: 1200px; margin: 0 auto; padding: 24px; }
h1 { color: #58a6ff; margin-bottom: 8px; font-size: 28px; }
h2 { color: #8b949e; margin: 32px 0 16px; font-size: 20px; border-bottom: 1px solid #30363d; padding-bottom: 8px; }
h3 { color: #58a6ff; margin: 24px 0 12px; font-size: 18px; }
.meta { color: #8b949e; font-size: 14px; margin-bottom: 24px; }
.summary-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; margin-bottom: 32px; }
.summary-card { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 16px; }
.summary-card .label { color: #8b949e; font-size: 12px; text-transform: uppercase; margin-bottom: 4px; }
.summary-card .value { color: #e6edf3; font-size: 24px; font-weight: 600; }
.summary-card .value.good { color: #3fb950; }
.summary-card .value.warn { color: #d29922; }
.summary-card .value.bad { color: #f85149; }
table { width: 100%; border-collapse: collapse; background: #161b22; border-radius: 8px; overflow: hidden; margin-bottom: 24px; }
th { background: #21262d; color: #8b949e; font-weight: 600; text-align: left; padding: 12px 16px; font-size: 12px; text-transform: uppercase; border-right: 1px solid #30363d; }
th:last-child { border-right: none; }
td { padding: 10px 16px; border-top: 1px solid #21262d; font-size: 15px; border-right: 1px solid #21262d; }
td:last-child { border-right: none; }
tr:hover td { background: #1c2128; }
.model-name { font-weight: 600; color: #58a6ff; }
.metric-good { color: #3fb950; }
.metric-warn { color: #d29922; }
.metric-bad { color: #f85149; }
.na { color: #484f58; }
.tabs { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 24px; }
.tab { padding: 8px 16px; background: #21262d; border: 1px solid #30363d; border-radius: 6px; cursor: pointer; color: #8b949e; font-size: 14px; }
.tab:hover { background: #30363d; }
.tab.active { background: #238636; border-color: #238636; color: #fff; }
.symbol-content { display: none; }
.symbol-content.active { display: block; }
.prediction-row { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 4px; }
.prediction-badge { padding: 2px 8px; border-radius: 4px; font-size: 12px; background: #21262d; }
.prediction-badge.correct { background: rgba(63, 185, 80, 0.2); color: #3fb950; }
.prediction-badge.wrong { background: rgba(248, 81, 73, 0.2); color: #f85149; }
</style>
</head>
<body>
<div class="container">
<h1>📊 Model Evaluation Report</h1>
<p class="meta">Generated: $generated_time</p>

<div class="summary-grid">
$summary_cards
</div>

<h2>Combined Metrics (All Symbols)</h2>
$combined_table

<h2>Per-Symbol Results</h2>
<div class="tabs" id="symbol-tabs">
$symbol_tabs
</div>
$symbol_content

<script>
function showSymbol(symbol) {
    document.querySelectorAll('.symbol-content').forEach(el => el.classList.remove('active'));
    document.querySelectorAll('#symbol-tabs .tab').forEach(el => el.classList.remove('active'));
    document.getElementById('symbol-' + symbol).classList.add('active');
    document.querySelector('#symbol-tabs .tab[onclick*="' + symbol + '"]').classList.add('active');
}
</script>
</div>
</body>
</html>""")


def render_evaluation_report(
    summary_cards: str,
    combined_table: str,
    symbol_tabs: str,
    symbol_content: str,
) -> str:
    """Render the evaluation report HTML."""
    return HTML_TEMPLATE.substitute(
        generated_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        summary_cards=summary_cards,
        combined_table=combined_table,
        symbol_tabs=symbol_tabs,
        symbol_content=symbol_content,
    )
