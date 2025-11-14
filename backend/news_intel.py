"""Full News Intelligence Engine."""
from __future__ import annotations
import os, re, json, time, datetime
from typing import Dict, List, Tuple, Optional
import pandas as pd
from .data_pipeline import log, _read_rolling
from . import summarizer, vector_index
from .config import PATHS  # ✅ new unified path import

# ---------------------------------------------------------------------
# Directory + Path Setup
# ---------------------------------------------------------------------
RAW_DIR = PATHS["news"] / "raw"       # news_cache base
OUT_DIR = PATHS["news"]        # same output base
ML_DIR = PATHS["ml_data"]      # ml_data root
PRIORS_PATH = PATHS["backend_service"].parent / "event_priors.json"

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _today_tag() -> str:
    return datetime.datetime.utcnow().strftime('%Y%m%d')

def _load_today_raw() -> Dict[str, Dict]:
    """Loads today's raw news safely, waiting for any fetchers to finish writing."""
    path = RAW_DIR / f'news_raw_{_today_tag()}.json'
    if not path.exists():
        return {}

    lock_path = PATHS["news"] / "news_raw.lock"
    waited = 0
    while lock_path.exists() and waited < 60:  # wait up to 60s
        time.sleep(1)
        waited += 1

    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        time.sleep(2)
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    except Exception:
        return {}

def _load_priors() -> Dict:
    try:
        with open(PRIORS_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {'priors': {}, 'scalers': {}, 'sector_impulse_map': {}}

def _parse_future_date(text: str) -> Optional[str]:
    try:
        import dateparser  # type: ignore
        dt = dateparser.parse(text, settings={'PREFER_DATES_FROM': 'future'})
        if dt:
            return dt.date().isoformat()
    except Exception:
        pass
    m = re.search(r'in\s+(\d+)\s+(day|days|week|weeks|month|months)', (text or '').lower())
    if m:
        n = int(m.group(1))
        unit = m.group(2)
        delta = datetime.timedelta(days=n if 'day' in unit else (7 * n if 'week' in unit else 30 * n))
        return (datetime.date.today() + delta).isoformat()
    return None

def _extract_tickers(text: str, rolling: Dict[str, Dict]) -> List[str]:
    if not text:
        return []
    t = text.upper()
    hits = []
    for s in (rolling or {}).keys():
        if s in t:
            hits.append(s)
    for s, node in (rolling or {}).items():
        nm = (node or {}).get('name')
        if nm and nm.upper() in t and s not in hits:
            hits.append(s)
    return hits[:8]

def _event_type(title: str, summary: str) -> str:
    txt = f"{title} {summary}".lower()
    if ('phase 3' in txt or 'phase iii' in txt or 'pdufa' in txt or 'fda' in txt):
        if 'approval' in txt or 'approves' in txt:
            return 'fda_approval'
        return 'fda_phase'
    if 'guidance' in txt and any(k in txt for k in ['raise', 'hike', 'boost']):
        return 'guidance_up'
    if 'guidance' in txt and any(k in txt for k in ['cut', 'lower', 'reduce']):
        return 'guidance_down'
    if any(k in txt for k in ['merger', 'acquire', 'm&a']):
        return 'mna'
    if any(k in txt for k in ['earnings', 'eps']):
        return 'earnings'
    if any(k in txt for k in ['layoff', 'job cuts']):
        return 'layoffs'
    if any(k in txt for k in ['lawsuit', 'litigation']):
        return 'litigation'
    if 'cpi' in txt:
        return 'macro_cpi'
    if 'fed' in txt or 'fomc' in txt:
        return 'macro_fed'
    if 'payrolls' in txt or 'jobs report' in txt:
        return 'macro_jobs'
    return 'other'

def _stance(title: str, summary: str, etype: str) -> int:
    txt = f"{title} {summary}".lower()
    pos = any(k in txt for k in ['beat', 'beats', 'approval', 'approves', 'record', 'raise guidance', 'strong'])
    neg = any(k in txt for k in ['miss', 'misses', 'cut guidance', 'probe', 'investigation', 'recall', 'downgrade', 'lawsuit'])
    if etype in ('guidance_up', 'fda_approval', 'mna'):
        return 1
    if etype in ('guidance_down', 'litigation'):
        return -1
    if pos and not neg:
        return 1
    if neg and not pos:
        return -1
    return 0

def _credibility_weight(source: str) -> float:
    if not source:
        return 0.6
    s = source.lower()
    tier1 = ['reuters', 'bloomberg', 'wsj', 'financial times']
    tier2 = ['yahoo', 'marketwatch', 'seekingalpha', 'investors']
    if any(x in s for x in tier1):
        return 0.95
    if any(x in s for x in tier2):
        return 0.85
    return 0.7

def _headline_match(title: str, tickers: List[str]) -> float:
    t = (title or '').upper()
    return 1.0 if any(s in t for s in tickers) else 0.5

def _novelty(headline: str) -> float:
    sim = vector_index.search_similar(headline or '', k=1, score_threshold=0.92)
    return 0.6 if sim else 1.0

def _apply_priors(etype: str, stance: int, cred: float, evid: float, nov: float, head: float, buzz: float, pri: Dict):
    base = (pri.get('priors', {}).get(etype) or pri.get('priors', {}).get('other') or {'short': 0.0, 'mid': 0.0, 'long': 0.0})
    scal = pri.get('scalers', {})
    scale = lambda x: (x ** scal.get('credibility_weight', 1.0))
    s = base['short'] * stance * scale(cred) * (evid ** scal.get('evidence_strength', 1.0)) * \
        (nov ** scal.get('novelty', 0.7)) * (head ** scal.get('headline_match', 0.5)) * \
        (1.0 + min(buzz, 50) * 0.01 * scal.get('buzz', 0.5))
    m = base['mid'] * stance * scale(cred) * (evid ** scal.get('evidence_strength', 1.0)) * \
        (nov ** scal.get('novelty', 0.7)) * (head ** scal.get('headline_match', 0.5)) * \
        (1.0 + min(buzz, 50) * 0.01 * scal.get('buzz', 0.5))
    l = base['long'] * stance * scale(cred) * (evid ** scal.get('evidence_strength', 1.0)) * \
        (nov ** scal.get('novelty', 0.7)) * (head ** scal.get('headline_match', 1.0)) * \
        (1.0 + min(buzz, 50) * 0.01 * scal.get('buzz', 0.5))
    return s, m, l

# ---------------------------------------------------------------------
# Main Runner
# ---------------------------------------------------------------------
def run_news_intel() -> Dict:
    start = time.time()
    raw = _load_today_raw()
    if not raw:
        log('[news_intel] ℹ️ no raw news today — nothing to analyze.')
        return {}
    rolling = _read_rolling() or {}
    pri = _load_priors()
    rows = []
    per_ticker = {}

    for hid, a in raw.items():
        title = a.get('title', '') or ''
        summary = a.get('summary', '') or ''
        url = a.get('url', '') or ''
        source = a.get('source', '') or ''
        text = f"{title} {summary}".strip()

        tickers = _extract_tickers(text, rolling)
        etype = _event_type(title, summary)
        stance = _stance(title, summary, etype)
        cred = _credibility_weight(source)
        evid = 0.8 if stance != 0 else 0.6
        head = _headline_match(title, tickers)
        nov = _novelty(title)
        fut_date = _parse_future_date(text)

        summary_short = summarizer.summarize(text)
        why_text = summarizer.why_it_matters(text)

        if not tickers:
            tickers = ['__MARKET__']

        for sym in tickers:
            buzz = 1.0
            sh, md, lg = _apply_priors(etype, stance, cred, evid, nov, head, buzz, pri)
            sector = None
            if sym != '__MARKET__':
                sector = (rolling.get(sym) or {}).get('sector')
                smap = pri.get('sector_impulse_map', {}).get(etype) or {}
                mult = smap.get(str(sector), smap.get('default', 1.0))
                sh, md, lg = sh * mult, md * mult, lg * mult

            rows.append({
                'event_id': hid,
                'published_at': a.get('published_at'),
                'source': source,
                'target_type': 'ticker' if sym != '__MARKET__' else 'market',
                'ticker': None if sym == '__MARKET__' else sym,
                'sector': sector,
                'event_type': etype,
                'stance': stance,
                'evidence_strength': evid,
                'novelty': nov,
                'credibility_weight': cred,
                'headline_match': head,
                'future_date': fut_date,
                'impact_short': sh,
                'impact_mid': md,
                'impact_long': lg,
                'headline': title,
                'url': url,
                'summary_short': summary_short,
                'why_it_matters': why_text
            })

            if sym != '__MARKET__':
                srow = per_ticker.setdefault(
                    sym,
                    {'sent_sum': 0.0, 'sent_cnt': 0, 'buzz': 0, 'short': 0.0, 'mid': 0.0, 'long': 0.0,
                     'next_event_date': None, 'next_event_kind': None}
                )
                srow['sent_sum'] += stance
                srow['sent_cnt'] += 1
                srow['buzz'] += 1
                srow['short'] += sh
                srow['mid'] += md
                srow['long'] += lg
                if fut_date and not srow['next_event_date']:
                    srow['next_event_date'] = fut_date
                    srow['next_event_kind'] = etype

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ev_path = OUT_DIR / f'news_events_{_today_tag()}.parquet'
    try:
        pd.DataFrame(rows).to_parquet(ev_path, index=False)
    except Exception:
        ev_path = OUT_DIR / f'news_events_{_today_tag()}.json'
        with open(ev_path, 'w', encoding='utf-8') as f:
            json.dump(rows, f, indent=2, ensure_ascii=False)

    senti_map = {}
    for sym, agg in per_ticker.items():
        sentiment_score = (agg['sent_sum'] / max(1, agg['sent_cnt'])) if agg['sent_cnt'] else 0.0
        senti_map[sym] = {
            'sentiment': round(sentiment_score, 4),
            'buzz': int(agg['buzz']),
            'event_short_impulse': round(agg['short'], 6),
            'event_mid_impulse': round(agg['mid'], 6),
            'event_long_impulse': round(agg['long'], 6),
            'next_event_date': agg['next_event_date'],
            'next_event_kind': agg['next_event_kind']
        }

    sm_path = OUT_DIR / f'sentiment_map_{_today_tag()}.json'
    with open(sm_path, 'w', encoding='utf-8') as f:
        json.dump(senti_map, f, indent=2, ensure_ascii=False)

    ML_DIR.mkdir(parents=True, exist_ok=True)
    feat_rows = [{'symbol': k, **v} for k, v in senti_map.items()]
    feats_path = ML_DIR / f'news_features_{_today_tag()}.parquet'
    try:
        pd.DataFrame(feat_rows).to_parquet(feats_path, index=False)
    except Exception:
        feats_path = ML_DIR / f'news_features_{_today_tag()}.json'
        with open(feats_path, 'w', encoding='utf-8') as f:
            json.dump(feat_rows, f, indent=2, ensure_ascii=False)

    dur = time.time() - start
    log(f"[news_intel] ✅ events → {ev_path} | senti_map → {sm_path} | feats → {feats_path} | {dur:.1f}s")
    return {'events': str(ev_path), 'sentiment_map': str(sm_path), 'features': str(feats_path)}

if __name__ == '__main__':
    run_news_intel()
