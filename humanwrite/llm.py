"""Client for the local LLM endpoint.

Supports the two common OpenAI-compatible request shapes:

* OpenAI-style servers (vLLM, llama-swap, ...): ``POST {base}/chat/completions``
  with OpenAI parameter names (temperature, top_p, presence_penalty, ...).
* Raw llama-server: ``POST {base}/api/chat`` with llama parameter names
  (no presence_penalty; sampling knobs use llama's names).

``api_style`` in config.yaml selects the shape: ``auto`` (default) tries the
OpenAI path first and falls back to ``/api/chat`` if the server rejects it;
``openai`` and ``llama`` force one. The client is a plain class so tests can
substitute a stub with the same ``chat`` signature.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any

DEFAULT_TIMEOUT = 600
RETRY_BACKOFF = 5.0


class LLMError(RuntimeError):
    pass


class _StyleMismatch(LLMError):
    """The server rejected this request shape (HTTP 404/405); the caller may
    retry with the other API style."""


class LLMClient:
    """Minimal blocking client for a local OpenAI-compatible endpoint."""

    def __init__(
        self,
        endpoint: str,
        model: str,
        api_key: str | None = None,
        api_style: str = "auto",
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = 3,
        extra_body: dict[str, Any] | None = None,
        log=None,
    ) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.api_style = api_style
        self.timeout = timeout
        self.max_retries = max_retries
        self.extra_body = extra_body or {}
        self._log = log or (lambda msg: None)
        self._llama_fallback = api_style == "auto"

    # -- public API ---------------------------------------------------------

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        top_p: float | None = None,
        presence_penalty: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Send a chat completion and return the assistant text."""
        if self.api_style == "llama" and presence_penalty:
            self._log(f"note: presence_penalty={presence_penalty} ignored (llama endpoint)")

        last_err: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                return self._post_one(messages, temperature, top_p, presence_penalty, max_tokens)
            except _StyleMismatch as err:
                if self._llama_fallback and self.api_style != "llama":
                    self._log("server rejected OpenAI shape; falling back to /api/chat")
                    self.api_style = "llama"
                    continue
                last_err = err
            except (urllib.error.URLError, TimeoutError, OSError, LLMError) as err:
                last_err = err
                if attempt < self.max_retries:
                    # 429 = the server is busy (single concurrency slot);
                    # wait longer than the usual backoff before retrying
                    wait = 30.0 if "429" in str(err) else RETRY_BACKOFF * attempt
                    self._log(f"endpoint error ({err}); retry {attempt}/"
                              f"{self.max_retries} in {wait:.0f}s")
                    time.sleep(wait)
        raise LLMError(f"endpoint failed after {self.max_retries} attempts: {last_err}")

    def ping(self) -> bool:
        """Return True if the endpoint answers at all."""
        base = self.endpoint
        if self.api_style == "llama":
            paths = ("/api/version", "/api/show")
        else:
            paths = ("/models", "/v1/models", "/api/version")
        for path in paths:
            url = base + path
            if url.count("//") > 1 and path.startswith("/v1") and base.endswith("/v1"):
                continue
            try:
                req = urllib.request.Request(url)
                with urllib.request.urlopen(req, timeout=10) as resp:
                    return resp.status < 500
            except (urllib.error.HTTPError,):
                # a 4xx means the server is up even if the path is wrong
                return True
            except (urllib.error.URLError, TimeoutError, OSError, ValueError):
                continue
        return False

    # -- internals ----------------------------------------------------------

    def _post_one(self, messages, temperature, top_p, presence_penalty, max_tokens) -> str:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "temperature": temperature if temperature is not None else 1.0,
            "top_p": top_p if top_p is not None else 1.0,
            "max_tokens": max_tokens or 4096,
        }
        if presence_penalty:
            body["presence_penalty"] = presence_penalty
        if self.extra_body:
            body.update(self.extra_body)

        if self.api_style == "llama":
            url = self.endpoint + "/api/chat"
            # llama /api/chat ignores unknown keys; keep the same body.
        else:
            url = self.endpoint + "/chat/completions"

        return self._post(url, body)

    def _post(self, url: str, body: dict[str, Any]) -> str:
        data = json.dumps(body).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if self.api_key and self.api_key not in ("", "not-needed"):
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as err:
            detail = err.read().decode("utf-8", errors="replace")[:500]
            # 404/405 on /chat/completions means "not an OpenAI server" —
            # let the caller try /api/chat instead of retrying uselessly.
            if err.code in (404, 405) and self.api_style == "auto":
                raise self._StyleMismatch(f"HTTP {err.code} from {url}") from err
            raise LLMError(f"HTTP {err.code} from {url}: {detail}") from err

        try:
            if isinstance(payload.get("message"), dict):
                # llama-server /api/chat shape
                return str(payload["message"].get("content", "")).strip()
            if payload.get("choices"):
                # OpenAI shape
                return str(
                    payload["choices"][0].get("message", {}).get("content", "")
                ).strip()
        except (KeyError, TypeError) as err:
            raise LLMError(f"unexpected response shape from {url}: {payload!r:.300}") from err
        raise LLMError(f"no content in response from {url}: {payload!r:.300}")


def client_from_config(cfg: dict[str, Any], log=None) -> LLMClient:
    return LLMClient(
        endpoint=cfg.get("endpoint", "http://10.60.50.11:9292/v1"),
        model=cfg.get("model", "Qwen3.8-27B"),
        api_key=cfg.get("api_key"),
        api_style=cfg.get("api_style", "auto"),
        timeout=float(cfg.get("timeout_seconds", DEFAULT_TIMEOUT)),
        max_retries=int(cfg.get("max_retries", 3)),
        extra_body=cfg.get("extra_body"),
        log=log,
    )
