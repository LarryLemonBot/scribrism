"""Escape and render local detector reports."""
import html
import json
from pathlib import Path

def four_class_signals(scores):
    return {**scores, "ai_origin": scores["ai"] + scores["humanized"], "non_human": scores["ai"] + scores["ai_edited"] + scores["humanized"]}

def render(report, text):
    esc = lambda value: html.escape(str(value))
    result = report.get('result', {})
    val = report.get('validation', {})
    parts = ['<!doctype html><html lang="en"><meta charset="utf-8">',
             '<meta name="viewport" content="width=device-width, initial-scale=1">',
             '<title>Scribrism</title><style>body{font:17px/1.55 system-ui;max-width:1200px;margin:40px auto;padding:0 24px;color:#162d35;background:#f8faf9}aside{padding:18px;background:#fff0c9;border-left:4px solid #b98312}table{border-collapse:collapse;width:100%;background:white}td,th{padding:10px;text-align:left;border-bottom:1px solid #dde4e4;vertical-align:top}code{overflow-wrap:anywhere}pre{white-space:pre-wrap}.scroll{overflow-x:auto}small{color:#456}</style>',
             '<h1>Scribrism</h1>', '<p>' + esc(Path(report['input']['path']).name) + '</p>',
             '<aside>Experimental model signals can be wrong. These are not Turnitin/GPTZero results, calibrated authorship probabilities, or percentages of text written by AI.</aside>',
             '<h2>Scan ' + esc(report['scan_status']) + '</h2>',
             '<p>Experimental classification: ' + esc(result.get('label', 'unavailable')) + '</p>',
             '<p>' + esc(report.get('error', 'All extracted tokens were scanned.')) + '</p>',
             '<p>Validation: ' + esc(report['validation_status']) + '</p>',
             '<p>A detector result does not establish authorship or approve submission.</p>',
             '<p><small>Created ' + esc(report['created_at']) + ' · file SHA-256 <code>' + esc(report['input']['sha256']) + '</code></small></p>',
             '<p>Model: ' + esc(report['model_id']) + '</p>']
    metrics = val.get('metrics') or val.get('historical_metrics')
    if metrics:
        current = val.get('applicable_to_current_scoring', False)
        parts.append('<h2>' + ('Bound evaluation results' if current else 'Historical predecessor results') + '</h2>')
        parts.append('<p>' + ('This evaluation is bound to the current scoring implementation; its measured scope and limitations still apply.' if current else 'These results describe an earlier evaluated system. They are not measured accuracy for this implementation, extraction and runtime.') + '</p><ul>')
        for name, m in metrics.items():
            parts.append('<li>' + esc(name) + ': ' + esc(m['correct']) + '/' + esc(m['total']) + ' correct; ' + esc(m['false_human_accusations']) + '/' + esc(m['human_total']) + ' human samples flagged AI or mixed.</li>')
        parts.append('</ul>')
    if val.get('binding_reasons'):
        parts.append('<ul>' + ''.join('<li>' + esc(x) + '</li>' for x in val['binding_reasons']) + '</ul>')
    parts.append('<h2>Extraction notes</h2><ul>' + ''.join('<li>' + esc(x) + '</li>' for x in report.get('extraction_notes', [])) + '</ul>')
    means = result.get('mean_class_scores')
    if means:
        signals = four_class_signals(means)
        labels = [('human', 'Human class'), ('ai', 'Raw AI class'), ('ai_edited', 'AI-edited human class'),
                  ('humanized', 'Paraphrased AI class'), ('ai_origin', 'Combined AI-origin signal (ai + humanized)'),
                  ('non_human', 'Combined non-human-class signal (ai + ai_edited + humanized)')]
        parts.append('<h2>Document model signals</h2><p>These are averaged, uncalibrated class outputs. The AI candidate combines raw and paraphrased AI classes; the AI-edited human class maps to a mixed candidate. The region rule can also produce Mixed. None proves actual writing history.</p><ul>')
        parts.extend('<li>' + label + ': ' + f'{signals[key]:.3f}' + '</li>' for key, label in labels)
        parts.append('</ul>')
    parts.append('<h2>Passage signals</h2><p>Each row is a whole-window prediction. Windows overlap, so their values are not independent evidence and do not identify which individual word or sentence was written by AI.</p><div class="scroll"><table>')
    windows = result.get('windows', [])
    if means:
        parts.append('<tr><th>Window</th><th>Human</th><th>Raw AI</th><th>AI-edited human</th><th>Paraphrased AI</th><th>AI + paraphrased AI</th><th>All non-human classes</th><th>Text</th></tr>')
        for index, window in enumerate(windows, 1):
            s = four_class_signals(window['scores'])
            values = ''.join('<td>' + f'{s[key]:.3f}' + '</td>' for key in ['human', 'ai', 'ai_edited', 'humanized', 'ai_origin', 'non_human'])
            parts.append('<tr><td>' + str(index) + '</td>' + values + '<td>' + esc(text[window['char_start']:window['char_end']]) + '</td></tr>')
    else:
        parts.append('<tr><th>Window</th><th>Raw model score</th><th>Text</th></tr>')
        for index, window in enumerate(windows, 1):
            parts.append('<tr><td>' + str(index) + '</td><td>' + f'{window["score"]:.3f}' + '</td><td>' + esc(text[window['char_start']:window['char_end']]) + '</td></tr>')
    parts.append('</table></div><h2>Extracted text</h2><pre>' + esc(text) + '</pre></html>')
    return ''.join(parts)
