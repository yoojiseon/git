"""API 키가 필요 없는 공개 데이터 수집.

- FRED (미국 연준 경제 데이터): fredgraph.csv 엔드포인트는 키 없이 받을 수 있음
- 네이버 금융 차트: 국내 상장 ETF 일별 시세와 종목명
- Google News RSS: 최신 경제 뉴스 헤드라인

응답은 .cache/ 폴더에 하루 단위로 저장해서 같은 날 다시 실행하면 재사용합니다.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import io
import re
import sys
import time
import urllib.parse
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import requests

CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; pension-agent/0.1)"}

# 이 도메인들에 접속할 수 있어야 데이터가 수집됩니다.
REQUIRED_HOSTS = ["fred.stlouisfed.org", "fchart.stock.naver.com", "news.google.com"]


class DataError(RuntimeError):
    pass


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _fetch(url: str, *, timeout: int = 15, retries: int = 2, use_cache: bool = True) -> bytes:
    key = hashlib.sha1(url.encode()).hexdigest()[:16]
    cache_file = CACHE_DIR / dt.date.today().isoformat() / key
    if use_cache and cache_file.exists():
        return cache_file.read_bytes()

    last: Exception | None = None
    for attempt in range(retries):
        started = time.time()
        try:
            resp = requests.get(url, headers=HEADERS, timeout=(5, timeout))
            resp.raise_for_status()
            if use_cache:
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                cache_file.write_bytes(resp.content)
            _log(f"  OK {time.time() - started:4.1f}s {url[:90]}")
            return resp.content
        except requests.RequestException as exc:  # 네트워크 오류는 재시도
            last = exc
            _log(f"  FAIL {time.time() - started:4.1f}s {url[:90]} ({type(exc).__name__})")
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    raise DataError(f"데이터 수집 실패: {url} ({last})")


# ---------------------------------------------------------------- FRED
def fred_series(series_id: str, start: str = "2010-01-01") -> pd.Series:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}&cosd={start}"
    df = pd.read_csv(io.BytesIO(_fetch(url)))
    if df.shape[1] < 2:
        raise DataError(f"FRED {series_id}: 예상과 다른 응답")
    values = pd.to_numeric(df.iloc[:, 1], errors="coerce")
    values.index = pd.to_datetime(df.iloc[:, 0])
    return values.dropna().rename(series_id)


def fred_many(series_ids: list[str]) -> tuple[dict[str, pd.Series], list[str]]:
    out, errors = {}, []
    for sid in series_ids:
        try:
            out[sid] = fred_series(sid)
        except (DataError, ValueError) as exc:
            errors.append(f"{sid}: {exc}")
    return out, errors


# ---------------------------------------------------------------- 네이버 금융
@dataclass
class PriceHistory:
    code: str
    name: str
    close: pd.Series  # DatetimeIndex → 종가(원)


_XML_DECL = re.compile(r"^\s*<\?xml[^>]*\?>")


def parse_naver_chart(raw: bytes, code: str) -> PriceHistory:
    text = None
    for enc in ("euc-kr", "cp949", "utf-8"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    if text is None:
        raise DataError(f"{code}: 응답 인코딩을 해석할 수 없음")
    root = ET.fromstring(_XML_DECL.sub("", text))
    chart = next(root.iter("chartdata"), None)
    if chart is None:
        raise DataError(f"{code}: 차트 데이터 없음")
    dates, closes = [], []
    for item in root.iter("item"):
        parts = (item.get("data") or "").split("|")
        if len(parts) >= 5 and parts[4] not in ("", "null"):
            dates.append(pd.to_datetime(parts[0], format="%Y%m%d"))
            closes.append(float(parts[4]))
    if not closes:
        raise DataError(f"{code}: 시세가 비어 있음 (종목코드 확인 필요)")
    close = pd.Series(closes, index=pd.DatetimeIndex(dates), name=code).sort_index()
    return PriceHistory(code=code, name=chart.get("name", "").strip(), close=close)


def naver_history(code: str, count: int = 4000) -> PriceHistory:
    url = (
        "https://fchart.stock.naver.com/sise.nhn?"
        f"symbol={code}&timeframe=day&count={count}&requestType=0"
    )
    return parse_naver_chart(_fetch(url), code)


def names_match(expected: str, actual: str) -> bool:
    """운용사 브랜드 변경(KOSEF→KIWOOM, KINDEX→ACE 등)을 감안한 느슨한 비교."""
    def norm(s: str) -> str:
        s = re.sub(r"[\s()·\-]", "", s.upper())
        return re.sub(r"^(KODEX|TIGER|ACE|KINDEX|KOSEF|KIWOOM|RISE|KBSTAR|SOL|PLUS|ARIRANG|HANARO)", "", s)

    a, b = norm(expected), norm(actual)
    return bool(a) and bool(b) and (a == b or a in b or b in a)


# ---------------------------------------------------------------- 뉴스
@dataclass
class NewsItem:
    topic: str
    title: str
    link: str
    published: str
    source: str


NEWS_TOPICS = {
    "미국 증시": "미국 증시 전망",
    "연준·금리": "연준 금리 인하 OR 인상",
    "한국은행": "한국은행 기준금리",
    "환율": "원달러 환율 전망",
    "금": "금값 전망",
    "연금·ETF": "연금 ETF 투자",
}


def google_news(topic: str, query: str, limit: int = 5) -> list[NewsItem]:
    q = urllib.parse.quote(f"{query} when:14d")
    url = f"https://news.google.com/rss/search?q={q}&hl=ko&gl=KR&ceid=KR:ko"
    root = ET.fromstring(_fetch(url))
    items = []
    for item in root.iter("item"):
        source = item.find("source")
        items.append(
            NewsItem(
                topic=topic,
                title=(item.findtext("title") or "").strip(),
                link=(item.findtext("link") or "").strip(),
                published=(item.findtext("pubDate") or "").strip(),
                source=source.text.strip() if source is not None and source.text else "",
            )
        )
        if len(items) >= limit:
            break
    return items


def latest_news(limit_per_topic: int = 5) -> tuple[list[NewsItem], list[str]]:
    news, errors = [], []
    for topic, query in NEWS_TOPICS.items():
        try:
            news.extend(google_news(topic, query, limit_per_topic))
        except (DataError, ET.ParseError) as exc:
            errors.append(f"뉴스({topic}): {exc}")
    return news, errors
