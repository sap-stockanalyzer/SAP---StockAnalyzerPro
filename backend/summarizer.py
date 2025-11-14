"""Summarization utils (transformers optional)."""
from __future__ import annotations

_PIPE = None
def _get_pipe():
    global _PIPE
    if _PIPE is not None: return _PIPE
    try:
        from transformers import pipeline  # type: ignore
        _PIPE = pipeline('summarization', model='sshleifer/distilbart-cnn-12-6')
    except Exception:
        _PIPE = None
    return _PIPE

def summarize(text: str, mode: str = 'concise', max_chars: int = 260) -> str:
    txt = (text or '').strip()
    if not txt: return ''
    p = _get_pipe()
    if p is None:
        # first sentence fallback
        if '.' in txt:
            first = txt.split('.')[0].strip() + '.'
            return first[:max_chars]
        return (txt[:max_chars] + ('...' if len(txt) > max_chars else ''))
    try:
        out = p(txt[:2000], max_length=96, min_length=18, do_sample=False)[0]['summary_text']
        return out.strip()
    except Exception:
        return (txt[:max_chars] + ('...' if len(txt) > max_chars else ''))

def why_it_matters(text: str) -> str:
    s = summarize(text)
    return ('Why it matters: ' + s) if s else ''
