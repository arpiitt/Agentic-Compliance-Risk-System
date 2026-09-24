"""
SEC EDGAR API async tools with retry, rate-limiting, timeout, and caching.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
    before_sleep_log,
)

from app.cache.redis_cache import NS_EDGAR, get_cached, set_cached
from app.config import get_settings
from app.metrics.prometheus import edgar_request_total, retry_total

logger = logging.getLogger(__name__)
settings = get_settings()

EDGAR_BASE = "https://data.sec.gov"
TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"

# In-memory ticker→CIK cache to avoid repeated downloads
_ticker_cik_map: dict[str, str] = {}


# ─── Retry policy ─────────────────────────────────────────────────────────────
def _is_retryable(exc: Exception) -> bool:
    """Return True only for transient errors worth retrying.

    404 Not Found from EDGAR is a permanent failure — the filing does not exist —
    so it must NOT be retried to avoid wasting time on every run.
    """
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in (429, 500, 502, 503, 504)
    return isinstance(exc, (httpx.TimeoutException, httpx.NetworkError))


def _edgar_retry(**kw):
    return retry(
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.NetworkError)),
        stop=stop_after_attempt(settings.max_retries),
        wait=wait_exponential_jitter(initial=1, max=16),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )


def _get_http_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(
        headers={"User-Agent": settings.sec_user_agent},
        timeout=settings.timeout_seconds,
        follow_redirects=True,
    )


# ─── CIK lookup ───────────────────────────────────────────────────────────────
async def fetch_ticker_cik_map() -> dict[str, str]:
    """Download SEC ticker→CIK map (cached in-memory)."""
    global _ticker_cik_map
    if _ticker_cik_map:
        return _ticker_cik_map
    cached = await get_cached(NS_EDGAR, {"op": "ticker_map"})
    if cached:
        _ticker_cik_map = cached
        return _ticker_cik_map
    async with _get_http_client() as client:
        resp = await asyncio.wait_for(
            client.get(TICKER_MAP_URL),
            timeout=settings.timeout_seconds,
        )
        resp.raise_for_status()
        raw: dict[str, dict] = resp.json()
    # Transform to ticker → zero-padded CIK
    mapping = {
        v["ticker"].upper(): str(v["cik_str"]).zfill(10)
        for v in raw.values()
    }
    _ticker_cik_map = mapping
    await set_cached(NS_EDGAR, {"op": "ticker_map"}, mapping, ttl=3600)
    return mapping


async def resolve_cik(entity: str, ticker: str | None = None) -> str | None:
    """Resolve a ticker or company name to a CIK. Returns None if not found."""
    mapping = await fetch_ticker_cik_map()
    if ticker and ticker.upper() in mapping:
        return mapping[ticker.upper()]
    # Fuzzy: try entity as ticker
    entity_upper = entity.upper().split()[0]
    if entity_upper in mapping:
        return mapping[entity_upper]
    # Fallback: search EDGAR full-text search API
    try:
        async with _get_http_client() as client:
            resp = await asyncio.wait_for(
                client.get(
                    "https://efts.sec.gov/LATEST/search-index?q=%22"
                    + entity.replace(" ", "+")
                    + "%22&dateRange=custom&startdt=2020-01-01&forms=10-K",
                    headers={"User-Agent": settings.sec_user_agent},
                ),
                timeout=settings.timeout_seconds,
            )
            if resp.status_code == 200:
                hits = resp.json().get("hits", {}).get("hits", [])
                if hits:
                    return str(hits[0]["_source"].get("entity_id", "")).zfill(10)
    except Exception:
        pass
    return None


# ─── Filings fetch ────────────────────────────────────────────────────────────
@_edgar_retry()
async def _fetch_submissions(cik: str) -> dict:
    cached = await get_cached(NS_EDGAR, {"op": "submissions", "cik": cik})
    if cached:
        return cached
    async with _get_http_client() as client:
        url = f"{EDGAR_BASE}/submissions/CIK{cik}.json"
        resp = await asyncio.wait_for(client.get(url), timeout=settings.timeout_seconds)
        resp.raise_for_status()
        data = resp.json()
    await set_cached(NS_EDGAR, {"op": "submissions", "cik": cik}, data, ttl=settings.retrieval_cache_ttl)
    edgar_request_total.labels(status="success").inc()
    return data


async def fetch_recent_filings(
    cik: str,
    form_types: list[str] | None = None,
    count: int = 5,
) -> list[dict[str, Any]]:
    """Return recent filing metadata for a CIK, filtered by form type."""
    if form_types is None:
        form_types = ["10-K", "10-Q", "8-K", "DEF 14A", "S-1"]
    cache_key = {"op": "recent_filings", "cik": cik, "form_types": sorted(form_types), "count": count}
    cached = await get_cached(NS_EDGAR, cache_key)
    if cached:
        return cached

    try:
        submissions = await _fetch_submissions(cik)
    except Exception as exc:
        logger.error("Failed to fetch submissions for CIK %s: %s", cik, exc)
        edgar_request_total.labels(status="error").inc()
        return []

    filings_data = submissions.get("filings", {}).get("recent", {})
    forms = filings_data.get("form", [])
    accessions = filings_data.get("accessionNumber", [])
    dates = filings_data.get("filingDate", [])
    company_name = submissions.get("name", "Unknown")

    results = []
    for i, (form, acc, date) in enumerate(zip(forms, accessions, dates)):
        if form in form_types:
            results.append({
                "form_type": form,
                "accession_number": acc,
                "filing_date": date,
                "company_name": company_name,
                "cik": cik,
                "url": f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc.replace('-','')}/{acc}-index.htm",
            })
            if len(results) >= count:
                break

    await set_cached(NS_EDGAR, cache_key, results, ttl=settings.retrieval_cache_ttl)
    return results


# ─── Filing text fetch ────────────────────────────────────────────────────────
@_edgar_retry()
async def _fetch_filing_index(cik: str, accession_number: str) -> dict:
    acc_clean = accession_number.replace("-", "")
    url = f"{EDGAR_BASE}/Archives/edgar/data/{int(cik)}/{acc_clean}/{accession_number}-index.json"
    async with _get_http_client() as client:
        resp = await asyncio.wait_for(client.get(url), timeout=settings.timeout_seconds)
        resp.raise_for_status()
        edgar_request_total.labels(status="success").inc()
        return resp.json()


async def fetch_filing_text(cik: str, accession_number: str, max_chars: int = 8000) -> str:
    """Fetch and truncate the primary document text of a filing."""
    cache_key = {"op": "filing_text", "cik": cik, "acc": accession_number}
    cached = await get_cached(NS_EDGAR, cache_key)
    if cached:
        return cached

    try:
        index = await _fetch_filing_index(cik, accession_number)
        # Find primary HTM/TXT document
        docs = index.get("directory", {}).get("item", [])
        primary = next(
            (d for d in docs if d.get("type") in ("10-K", "10-Q", "8-K", "DEF 14A", "S-1")),
            docs[0] if docs else None,
        )
        if not primary:
            return ""
        acc_clean = accession_number.replace("-", "")
        doc_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc_clean}/{primary['name']}"
        async with _get_http_client() as client:
            resp = await asyncio.wait_for(client.get(doc_url), timeout=settings.timeout_seconds)
            resp.raise_for_status()
            text = resp.text[:max_chars]
        await set_cached(NS_EDGAR, cache_key, text, ttl=settings.retrieval_cache_ttl)
        edgar_request_total.labels(status="success").inc()
        return text
    except Exception as exc:
        logger.error("Failed to fetch filing text %s: %s", accession_number, exc)
        edgar_request_total.labels(status="error").inc()
        return ""


# ─── Parallel fetch ───────────────────────────────────────────────────────────
async def fetch_all_filings_parallel(
    cik: str,
    form_types: list[str] | None = None,
    count_per_type: int = 3,
) -> list[dict]:
    """Fetch filings and their full text concurrently."""
    filings_meta = await fetch_recent_filings(cik, form_types, count=count_per_type * len(form_types or []))

    async def enrich(meta: dict) -> dict:
        text = await fetch_filing_text(cik, meta["accession_number"])
        return {**meta, "text": text}

    # Parallel fetch with concurrency limit
    semaphore = asyncio.Semaphore(5)  # max 5 concurrent EDGAR requests

    async def bounded_enrich(meta: dict) -> dict:
        async with semaphore:
            return await enrich(meta)

    enriched = await asyncio.gather(*[bounded_enrich(m) for m in filings_meta], return_exceptions=True)
    return [e for e in enriched if isinstance(e, dict)]
