"""Live finance feed fetching for gold-relevant macro context."""

from __future__ import annotations

import json
import ssl
import urllib.request
from datetime import datetime

from config import NEWS_ENABLED, NEWS_FETCH_TIMEOUT, NEWS_MAX_ITEMS


RELEVANCE_KEYWORDS = (
    "黄金",
    "金价",
    "现货金",
    "xau",
    "gold",
    "美元",
    "美指",
    "美联储",
    "FOMC",
    "fed",
    "鲍威尔",
    "非农",
    "cpi",
    "pce",
    "利率决议",
    "降息",
    "加息",
    "通胀",
    "美债",
    "收益率",
    "避险",
    "地缘",
    "战争",
    "冲突",
    "制裁",
    "关税",
    "衰退",
    "石油",
    "原油",
)

_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"
)


def _get_json(url: str, timeout: float) -> dict:
    context = ssl.create_default_context()
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": _USER_AGENT,
            "Referer": "https://finance.sina.com.cn/",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout, context=context) as response:
        return json.loads(response.read().decode("utf-8", errors="ignore"))


def _sina_news() -> list[dict]:
    url = (
        "https://zhibo.sina.com.cn/api/zhibo/feed?"
        "page=1&page_size=30&zhibo_id=152&tag_id=0&dire=f&dpc=1"
    )
    payload = _get_json(url, NEWS_FETCH_TIMEOUT)
    items = (
        payload.get("result", {})
        .get("data", {})
        .get("feed", {})
        .get("list", [])
    )
    return [
        {
            "source": "新浪财经",
            "time": str(item.get("create_time", "")),
            "text": str(item.get("rich_text", "")).strip(),
            "url": str(item.get("docurl", "")),
        }
        for item in items
        if item.get("rich_text")
    ]


def _wallstreetcn_news() -> list[dict]:
    url = (
        "https://api-one.wallstcn.com/apiv1/content/lives?"
        "channel=global-channel&limit=30"
    )
    payload = _get_json(url, NEWS_FETCH_TIMEOUT)
    items = payload.get("data", {}).get("items", [])
    result = []
    for item in items:
        text = str(item.get("content_text") or item.get("title") or "").strip()
        if not text:
            continue
        timestamp = int(item.get("display_time") or 0)
        time_text = (
            datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")
            if timestamp
            else ""
        )
        result.append(
            {
                "source": "华尔街见闻",
                "time": time_text,
                "text": text,
                "url": str(item.get("uri") or ""),
            }
        )
    return result


def _is_relevant(text: str) -> bool:
    lower = text.lower()
    return any(keyword.lower() in lower for keyword in RELEVANCE_KEYWORDS)


def _dedupe(items: list[dict]) -> list[dict]:
    seen = set()
    unique = []
    for item in items:
        signature = "".join(item.get("text", "").split())[:80]
        if not signature or signature in seen:
            continue
        seen.add(signature)
        unique.append(item)
    return unique


def fetch_market_news(limit: int | None = None) -> list[dict]:
    """Return the newest gold/USD/Fed-related headlines, newest first."""
    if not NEWS_ENABLED:
        return []
    collected: list[dict] = []
    errors: list[str] = []
    for fetcher in (_wallstreetcn_news, _sina_news):
        try:
            collected.extend(fetcher())
        except Exception as exc:
            errors.append(f"{fetcher.__name__}: {type(exc).__name__}")
        if len(collected) >= NEWS_MAX_ITEMS * 2:
            break
    relevant = [item for item in _dedupe(collected) if _is_relevant(item.get("text", ""))]
    relevant.sort(key=lambda item: item.get("time", ""), reverse=True)
    limit = int(limit or NEWS_MAX_ITEMS)
    return relevant[:limit]
