"""
Shared Gemini LLM client with automatic model fallback and in-process working model cache.

Both risk_scorer and verifier delegate to generate_content_with_fallback() so that:
  - The fallback candidate list is maintained in exactly one place.
  - The first model that succeeds is cached module-level; subsequent calls skip the probe loop.
  - Only 404 / NOT_FOUND errors trigger fallback; all other errors propagate immediately.
"""
from __future__ import annotations

import logging
from typing import Any

import google.genai as genai
from google.genai import types as genai_types

logger = logging.getLogger(__name__)

# Ordered candidate list — prioritised by generation (newest first).
# Centralising this list means model updates and deprecations are managed in one place.
_FALLBACK_CANDIDATES: list[str] = [
    "models/gemini-3.5-flash-lite",
    "gemini-3.5-flash-lite",
    "models/gemini-3.5-flash",
    "models/gemini-3.8-flash",
    "models/gemini-3.7-flash",
    "models/gemini-3.6-flash",
    "models/gemini-2.5-flash-lite",
    "models/gemini-2.5-flash",
    "models/gemini-flash-lite-latest",
    "models/gemini-flash-latest",
    "models/gemini-pro-latest",
]

# In-process cache: once a model responds successfully, all subsequent calls use it directly.
# Avoids re-probing the full candidate list on every single LLM call.
_working_model: str | None = None


def get_client(api_key: str) -> genai.Client:
    """Return a configured Gemini client."""
    return genai.Client(api_key=api_key)


def generate_content_with_fallback(
    client: genai.Client,
    prompt: str,
    config: genai_types.GenerateContentConfig,
    initial_model: str,
) -> Any:
    """
    Call Gemini with automatic model fallback.

    Tries initial_model first. On a 404 / NOT_FOUND response, falls through the
    _FALLBACK_CANDIDATES list in order. The first model that succeeds is cached
    module-level so all future calls skip the probe loop entirely.

    Any non-404 exception (auth failure, quota, network) is re-raised immediately
    without consuming the fallback list.

    Raises:
        The last exception encountered if all candidates are exhausted.
    """
    global _working_model

    # Fast path: use the cached working model directly.
    if _working_model and _working_model != initial_model:
        try:
            return client.models.generate_content(
                model=_working_model,
                contents=prompt,
                config=config,
            )
        except Exception as exc:
            exc_str = str(exc).lower()
            if "404" in exc_str or "not_found" in exc_str or "no longer available" in exc_str or "not found" in exc_str:
                logger.warning(
                    "Cached working model '%s' returned 404; re-probing candidates.",
                    _working_model,
                )
                _working_model = None
            else:
                raise exc

    # Build deduplicated candidate list: initial_model first, then fallbacks.
    seen: set[str] = set()
    candidates: list[str] = []
    for m in [initial_model] + _FALLBACK_CANDIDATES:
        if m not in seen:
            seen.add(m)
            candidates.append(m)

    last_error: Exception | None = None
    for m in candidates:
        try:
            result = client.models.generate_content(
                model=m,
                contents=prompt,
                config=config,
            )
            _working_model = m
            if m != initial_model:
                logger.info("LLM fallback resolved to model '%s'.", m)
            return result
        except Exception as exc:
            last_error = exc
            exc_str = str(exc).lower()
            if "404" in exc_str or "not_found" in exc_str or "no longer available" in exc_str or "not found" in exc_str:
                logger.warning("Model '%s' unavailable (404), trying next candidate.", m)
                continue
            raise exc

    if last_error:
        raise last_error
