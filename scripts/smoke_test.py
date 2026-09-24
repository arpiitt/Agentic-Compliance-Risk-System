"""
Smoke test: probe all external API endpoints used by the system.
Usage: python3 scripts/smoke_test.py <GEMINI_API_KEY>
"""
from __future__ import annotations

import asyncio
import sys
import time

import httpx


def fmt(label: str, status: str, detail: str = "") -> None:
    color = {"PASS": "\033[92m", "FAIL": "\033[91m", "SKIP": "\033[93m"}.get(status, "")
    reset = "\033[0m"
    tag = f"[{status}]"
    print(f"  {color}{tag:<6}{reset} {label:<55} {detail}")


def test_gemini_llm(api_key: str) -> None:
    print("\nGemini LLM (generateContent)")
    import google.genai as genai
    from google.genai import types as genai_types
    client = genai.Client(api_key=api_key)
    candidates = [
        "gemini-3.5-flash-lite",
        "gemini-2.5-flash",
        "gemini-1.5-flash",
        "gemini-1.5-flash-002",
        "gemini-1.5-pro",
        "gemini-2.0-flash",
        "gemini-2.0-flash-lite",
    ]
    for model in candidates:
        t0 = time.perf_counter()
        try:
            resp = client.models.generate_content(
                model=model,
                contents="Reply with the single word: OK",
                config=genai_types.GenerateContentConfig(max_output_tokens=5),
            )
            ms = int((time.perf_counter() - t0) * 1000)
            text = resp.text.strip() if resp.text else ""
            fmt(model, "PASS", f"{ms}ms  reply={repr(text)}")
        except Exception as exc:
            ms = int((time.perf_counter() - t0) * 1000)
            fmt(model, "FAIL", f"{ms}ms  {str(exc)[:100]}")


def test_gemini_embedding(api_key: str) -> None:
    print("\nGemini Embedding (embedContent)")
    import google.genai as genai
    client = genai.Client(api_key=api_key)
    candidates = [
        "models/text-embedding-004",
        "text-embedding-004",
        "models/embedding-001",
        "embedding-001",
        "models/text-embedding-005",
        "text-multilingual-embedding-002",
    ]
    for model in candidates:
        t0 = time.perf_counter()
        try:
            result = client.models.embed_content(model=model, contents="compliance risk test")
            ms = int((time.perf_counter() - t0) * 1000)
            dim = len(result.embeddings[0].values)
            fmt(model, "PASS", f"{ms}ms  dim={dim}")
        except Exception as exc:
            ms = int((time.perf_counter() - t0) * 1000)
            fmt(model, "FAIL", f"{ms}ms  {str(exc)[:100]}")


async def test_edgar() -> None:
    print("\nSEC EDGAR (public, no auth)")
    async with httpx.AsyncClient(
        headers={"User-Agent": "SmokeTest smoketest@example.com"},
        timeout=15,
        follow_redirects=True,
    ) as client:
        t0 = time.perf_counter()
        try:
            resp = await client.get("https://www.sec.gov/files/company_tickers.json")
            ms = int((time.perf_counter() - t0) * 1000)
            fmt("Ticker map (sec.gov)", "PASS", f"{ms}ms  {len(resp.json())} tickers")
        except Exception as exc:
            ms = int((time.perf_counter() - t0) * 1000)
            fmt("Ticker map (sec.gov)", "FAIL", f"{ms}ms  {exc}")

        t0 = time.perf_counter()
        try:
            resp = await client.get("https://data.sec.gov/submissions/CIK0001318605.json")
            ms = int((time.perf_counter() - t0) * 1000)
            name = resp.json().get("name", "?")
            fmt("Submissions endpoint (TSLA)", "PASS", f"{ms}ms  name={name}")
        except Exception as exc:
            ms = int((time.perf_counter() - t0) * 1000)
            fmt("Submissions endpoint (TSLA)", "FAIL", f"{ms}ms  {exc}")

        # Use a known stable 2023 filing accession number
        t0 = time.perf_counter()
        try:
            acc = "0001628280-23-019617"
            acc_clean = acc.replace("-", "")
            url = f"https://data.sec.gov/Archives/edgar/data/1318605/{acc_clean}/{acc}-index.json"
            resp = await client.get(url)
            ms = int((time.perf_counter() - t0) * 1000)
            status = "PASS" if resp.status_code == 200 else "FAIL"
            fmt("Filing index (2023 stable)", status, f"{ms}ms  HTTP {resp.status_code}")
        except Exception as exc:
            ms = int((time.perf_counter() - t0) * 1000)
            fmt("Filing index (2023 stable)", "FAIL", f"{ms}ms  {exc}")


async def test_ddg_news() -> None:
    print("\nDuckDuckGo News (public, no auth)")
    async with httpx.AsyncClient(timeout=10, follow_redirects=True) as client:
        t0 = time.perf_counter()
        try:
            resp = await client.get("https://duckduckgo.com/?q=Tesla&ia=news&iar=news")
            ms = int((time.perf_counter() - t0) * 1000)
            fmt("DuckDuckGo news page", "PASS", f"{ms}ms  HTTP {resp.status_code}")
        except Exception as exc:
            ms = int((time.perf_counter() - t0) * 1000)
            fmt("DuckDuckGo news page", "FAIL", f"{ms}ms  {exc}")


async def main(gemini_key: str) -> None:
    print("=" * 70)
    print("  SMOKE TEST — Agentic Compliance Risk System")
    print(f"  Key prefix: {gemini_key[:16]}...")
    print("=" * 70)
    test_gemini_llm(gemini_key)
    test_gemini_embedding(gemini_key)
    await test_edgar()
    await test_ddg_news()
    print("\n" + "=" * 70)
    print("  Done.")
    print("=" * 70)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 scripts/smoke_test.py <API_KEY>")
        sys.exit(1)
    asyncio.run(main(sys.argv[1]))
