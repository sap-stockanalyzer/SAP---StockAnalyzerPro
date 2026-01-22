"""
scheduler_config.py

Central schedule for AION Analytics.
"""

from __future__ import annotations

ENABLE   = True
TIMEZONE = "America/Denver"

# Let scheduler_runner resolve project root robustly
PROJECT_ROOT = None

SCHEDULE = [
    {
        "name": "news_collect_morning",
        "time": "07:45",
        "module": "backend.services.news_fetcher",
        "args": [],
        "cwd": PROJECT_ROOT,
        "description": "Market-wide news collection (morning).",
    },
    {
        "name": "news_collect_midday",
        "time": "11:45",
        "module": "backend.services.news_fetcher",
        "args": [],
        "cwd": PROJECT_ROOT,
        "description": "Market-wide news collection (midday).",
    },
    {
        "name": "news_collect_afternoon",
        "time": "14:45",
        "module": "backend.services.news_fetcher",
        "args": [],
        "cwd": PROJECT_ROOT,
        "description": "Market-wide news collection (afternoon).",
    },
    {
        "name": "nightly_full",
        "time": "17:30",
        "module": "backend.jobs.nightly_job",
        "args": [],
        "cwd": PROJECT_ROOT,
        "description": "Full nightly rebuild.",
    },
    {
        "name": "evening_insights",
        "time": "18:00",
        "module": "backend.services.insights_builder",
        "args": [],
        "cwd": PROJECT_ROOT,
        "description": "Rebuild nightly insights/top picks.",
    },
    {
        "name": "social_sentiment_evening",
        "time": "20:30",
        "module": "backend.services.social_sentiment_fetcher",
        "args": [],
        "cwd": PROJECT_ROOT,
        "description": "Evening social sentiment refresh.",
    },
    {
        "name": "dt_premarket_full",
        "time": "06:30",
        "module": "dt_backend.jobs.daytrading_job",
        "args": [],
        "cwd": PROJECT_ROOT,
        "description": "DT premarket full prep.",
    },
    *[
        {
            "name": f"dt_hourly_{t.replace(':','')}",
            "time": t,
            "module": "dt_backend.jobs.daytrading_job",
            "args": [],
            "cwd": PROJECT_ROOT,
            "description": f"DT hourly refresh {t}.",
        }
        for t in ["09:30", "10:30", "11:30", "12:30", "13:30", "14:30"]
    ],
    *[
        {
            "name": f"eod_{w}_full",
            "time": "06:00",
            "module": f"backend.bots.runner_{w}",
            "args": ["--mode", "full"],
            "cwd": PROJECT_ROOT,
            "description": f"Premarket rebalance for {w} bot.",
        }
        for w in ["1w", "2w", "4w"]
    ],
    *[
        {
            "name": f"bot_loop_{w}_1135",
            "time": "11:35",
            "module": f"backend.bots.runner_{w}",
            "args": ["--mode", "loop"],
            "cwd": PROJECT_ROOT,
            "description": f"{w} bot hourly loop.",
        }
        for w in ["1w", "2w", "4w"]
    ],
    *[
        {
            "name": f"bot_loop_{w}_1430",
            "time": "14:30",
            "module": f"backend.bots.runner_{w}",
            "args": ["--mode", "loop"],
            "cwd": PROJECT_ROOT,
            "description": f"{w} bot afternoon loop.",
        }
        for w in ["1w", "2w", "4w"]
    ],
    *[
        {
            "name": f"eod_{w}_close",
            "time": "16:15",
            "module": f"backend.bots.runner_{w}",
            "args": ["--mode", "full"],
            "cwd": PROJECT_ROOT,
            "description": f"Market close rebalance for {w} bot.",
        }
        for w in ["1w", "2w", "4w"]
    ],
]
