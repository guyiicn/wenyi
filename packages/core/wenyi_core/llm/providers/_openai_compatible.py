"""OpenAI-compatible wire protocol and single-attempt response handling."""

from __future__ import annotations

import json
import os
import socket
import struct
import threading
import time
from abc import abstractmethod
from typing import Any, Generic, TypeVar

from pydantic import BaseModel

from ..retrying import EmptyResponseError, TruncatedResponseError
from ..transport import Messages, ProviderAdapter, RequestContext, ResolvedModel
from ..usage import (
    UsageSample,
    make_usage_sample,
    read_usage_counter,
    read_usage_int,
    read_usage_value,
)

OptionsT = TypeVar("OptionsT", bound=BaseModel)
_JSON_MODE_INSTRUCTION = "Output must be valid json."
_APPLICATION_TIMEOUT_FACTOR = 1.1
_ABORT_JOIN_SECONDS = 2.0
_WATCHDOG_POLL_SECONDS = 0.05


def application_timeout_seconds(timeout: float) -> float:
    """Wall-clock bound used when httpx/SSL read timeouts do not fire."""
    return float(timeout) * _APPLICATION_TIMEOUT_FACTOR


def httpx_timeout_config(timeout: float):
    """Split connect/read/write/pool deadlines so a stalled peer cannot hang one phase."""
    import httpx

    bounded = max(0.001, float(timeout))
    return httpx.Timeout(
        connect=min(60.0, bounded),
        read=bounded,
        write=bounded,
        pool=min(60.0, bounded),
    )


def os_socket_timeout_options(timeout: float) -> list[tuple[int, int, bytes]]:
    """Kernel recv/send timeouts. Python SSLSocket.settimeout can miss SSL_read waits."""
    seconds = max(1.0, float(timeout))
    if os.name == "nt":
        packed = struct.pack("@I", int(seconds * 1000))
    else:
        usec = int(round((seconds % 1.0) * 1_000_000))
        packed = struct.pack("@ll", int(seconds), usec)
    return [
        (socket.SOL_SOCKET, socket.SO_RCVTIMEO, packed),
        (socket.SOL_SOCKET, socket.SO_SNDTIMEO, packed),
    ]


def _iter_network_sockets(root: Any) -> list[socket.socket]:
    """Collect live TCP/SSL sockets from an OpenAI/httpx/httpcore client tree."""
    found: list[socket.socket] = []
    seen: set[int] = set()
    stack: list[Any] = [root]
    while stack:
        current = stack.pop()
        ident = id(current)
        if ident in seen:
            continue
        seen.add(ident)
        if isinstance(current, socket.socket):
            found.append(current)
            continue
        if current is None or isinstance(current, (str, bytes, bytearray, memoryview)):
            continue
        module = getattr(type(current), "__module__", "") or ""
        if module.startswith("unittest.mock"):
            continue
        extra = getattr(current, "get_extra_info", None)
        if callable(extra):
            try:
                sock = extra("socket")
            except Exception:
                sock = None
            if isinstance(sock, socket.socket):
                found.append(sock)
        if isinstance(current, dict):
            stack.extend(current.values())
            continue
        if isinstance(current, (list, tuple, set, frozenset)):
            stack.extend(current)
            continue
        for attr in (
            "_client",
            "_transport",
            "_pool",
            "_connection",
            "_connections",
            "_network_stream",
            "_stream",
            "_sock",
            "connections",
        ):
            try:
                value = getattr(current, attr)
            except Exception:
                continue
            if callable(value) and attr == "connections":
                try:
                    value = value()
                except Exception:
                    continue
            stack.append(value)
    return found


def abort_openai_client(client: Any) -> None:
    """Unblock SSL_read by shutting the TCP fd down; do not wait for SSL_shutdown."""
    if client is None:
        return
    for sock in _iter_network_sockets(client):
        try:
            sock.shutdown(socket.SHUT_RDWR)
        except Exception:
            pass
        try:
            sock.close()
        except Exception:
            pass
    http_client = getattr(client, "_client", None)
    if http_client is not None:
        try:
            http_client.close()
        except Exception:
            pass
    try:
        client.close()
    except Exception:
        pass


def _wait_until(event: threading.Event, timeout: float) -> bool:
    """Wait with a monotonic deadline so a missed Event timeout cannot hang forever."""
    deadline = time.monotonic() + max(0.0, timeout)
    while not event.is_set():
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        event.wait(timeout=min(_WATCHDOG_POLL_SECONDS, remaining))
    return True


def base_request_kwargs(
    model: str,
    messages: Messages,
    *,
    json_mode: bool,
) -> dict[str, Any]:
    """Build base Chat Completions arguments and add explicit JSON instructions when requested."""
    request_messages = messages
    if json_mode:
        request_messages = [dict(message) for message in messages]
        for message in request_messages:
            if message.get("role") == "system":
                message["content"] = f"{message.get('content', '')}\n\n{_JSON_MODE_INSTRUCTION}"
                break
        else:
            request_messages.insert(
                0,
                {"role": "system", "content": _JSON_MODE_INSTRUCTION},
            )
        # Some gateways validate only user content, including when mapping to Responses input.
        # Mentioning JSON only in the system message may be insufficient,
        # so also append the instruction to the final user message.
        for message in reversed(request_messages):
            if message.get("role") == "user":
                content = str(message.get("content", ""))
                if "json" not in content.lower():
                    message["content"] = f"{content}\n\n{_JSON_MODE_INSTRUCTION}"
                break
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": request_messages,
        "stream": False,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    return kwargs


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge provider request bodies, preferring user values."""
    merged = dict(base)
    for key, value in override.items():
        current = merged.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            merged[key] = deep_merge(current, value)
        else:
            merged[key] = value
    return merged


def normalize_openai_usage(usage: Any, *, separate_cache_writes: bool = True) -> UsageSample | None:
    """Normalize inclusive Chat Completions totals, not native Anthropic input_tokens.

    Gateways may expose OpenRouter-style nested cache_write_tokens or top-level
    cache_creation_input_tokens/cache_creation. Absence is not evidence of zero
    writes on a generic endpoint. Native OpenAI has no separate write counter.
    """
    if usage is None or read_usage_counter(usage, "prompt_tokens") is None:
        return None
    prompt_tokens = read_usage_int(usage, "prompt_tokens")
    details = read_usage_value(usage, "prompt_tokens_details")
    cache_hit_tokens = read_usage_counter(details, "cached_tokens")
    if cache_hit_tokens is None:
        cache_hit_tokens = read_usage_counter(usage, "cache_read_input_tokens")
    cache_write_tokens = read_usage_counter(details, "cache_write_tokens")
    if cache_write_tokens is None:
        cache_write_tokens = read_usage_counter(usage, "cache_creation_input_tokens")
    creation = read_usage_value(usage, "cache_creation")
    write_5m = read_usage_counter(creation, "ephemeral_5m_input_tokens")
    write_1h = read_usage_counter(creation, "ephemeral_1h_input_tokens")
    writes_complete = cache_write_tokens is not None or (
        write_5m is not None and write_1h is not None
    )
    if cache_write_tokens is None and (write_5m is not None or write_1h is not None):
        # A partial TTL breakdown proves some writes, but not the ordinary remainder.
        cache_write_tokens = (write_5m or 0) + (write_1h or 0)
    if cache_write_tokens is None and not separate_cache_writes:
        cache_write_tokens = 0
        writes_complete = True
    ordinary = None
    if cache_hit_tokens is not None and cache_write_tokens is not None and writes_complete:
        ordinary = max(0, prompt_tokens - cache_hit_tokens - cache_write_tokens)
    return make_usage_sample(
        usage,
        cache_hit_tokens=cache_hit_tokens,
        cache_miss_tokens=max(0, prompt_tokens - cache_hit_tokens)
        if cache_hit_tokens is not None
        else 0,
        cache_write_tokens=cache_write_tokens,
        uncached_tokens=ordinary,
        cache_write_5m_tokens=write_5m,
        cache_write_1h_tokens=write_1h,
    )


class OpenAICompatibleBaseClient(ProviderAdapter, Generic[OptionsT]):
    """Reuse one SDK connection; route selection and usage belong to the caller."""

    def _openai_sdk_kwargs(self) -> dict[str, Any]:
        """Extra ``OpenAI()`` constructor kwargs owned by a concrete provider."""
        return {}

    def _create_openai_client(self) -> Any:
        """Build one isolated SDK client. Callers must close it."""
        import httpx
        from openai import OpenAI

        self.validate_credentials()
        timeout_config = httpx_timeout_config(self.cfg.timeout)
        http_client = httpx.Client(
            timeout=timeout_config,
            transport=httpx.HTTPTransport(
                retries=0,
                http1=True,
                http2=False,
                socket_options=os_socket_timeout_options(self.cfg.timeout),
                limits=httpx.Limits(max_keepalive_connections=0, max_connections=4),
            ),
            follow_redirects=True,
            trust_env=True,
        )
        try:
            client = OpenAI(
                api_key=self.api_key() or "no-key",
                base_url=self.base_url,
                timeout=timeout_config,
                max_retries=0,
                http_client=http_client,
                **self._openai_sdk_kwargs(),
            )
        except BaseException:
            http_client.close()
            raise
        # Test doubles often swallow http_client. Close the unused pool immediately.
        if getattr(client, "_client", None) is not http_client:
            http_client.close()
        return client

    def _ensure_client(self) -> Any:
        with self._client_lock:
            if self._client is None:
                self._client = self._create_openai_client()
        return self._client

    def list_models(self) -> list[str]:
        """Read ``GET /models``; the OpenAI SDK pages through gateways that paginate."""
        return [str(item.id) for item in self._ensure_client().models.list() if item.id]

    @classmethod
    def output_limit(cls, options: BaseModel, hint: int | None, explicit: int | None) -> int | None:
        thinking = bool(getattr(options, "thinking", False))
        if explicit is not None:
            if thinking and explicit < 4096:
                raise ValueError(
                    "Thinking mode requires max_output_tokens >= 4096; disable thinking or increase the explicit limit"
                )
            return explicit
        return max(hint, 4096) if thinking and hint is not None else hint

    def _normalize_usage(self, usage: Any) -> UsageSample | None:
        return normalize_openai_usage(usage)

    def _json_response_fallback(
        self, model_config: ResolvedModel[OptionsT], message: Any
    ) -> str | None:
        return None

    @abstractmethod
    def _build_request_kwargs(
        self,
        model_config: ResolvedModel[OptionsT],
        messages: Messages,
        *,
        json_mode: bool,
        max_tokens: int | None,
    ) -> dict[str, Any]:
        raise NotImplementedError

    def _request(
        self,
        messages: Messages,
        model: ResolvedModel[OptionsT],
        *,
        json_mode: bool,
        context: RequestContext,
        response_format: dict[str, Any] | None = None,
    ) -> str:
        model_config = model
        kwargs = self._build_request_kwargs(
            model, messages, json_mode=json_mode, max_tokens=context.max_tokens
        )
        if response_format is not None:
            kwargs["response_format"] = response_format

        # Isolated client per request. Join-then-close() can miss silent-after-body
        # TLS hangs: SSL_read does not return, join may not resume, and close() can
        # deadlock on SSL_shutdown. Abort via socket.shutdown from a watchdog, and
        # enforce SO_RCVTIMEO so the kernel unblocks SSL_read.
        result_box: list[Any] = []
        client_box: list[Any] = []
        done = threading.Event()
        aborted = threading.Event()
        timeout_seconds = application_timeout_seconds(self.cfg.timeout)

        def _do_request() -> None:
            client = None
            try:
                client = self._create_openai_client()
                client_box.append(client)
                if aborted.is_set():
                    abort_openai_client(client)
                    raise TimeoutError(
                        f"LLM request exceeded application timeout ({timeout_seconds:.1f}s)"
                    )
                response = client.chat.completions.create(**kwargs)
                result_box.append(("success", response))
            except BaseException as error:
                result_box.append(("error", error))
            finally:
                done.set()
                if client is not None:
                    try:
                        client.close()
                    except Exception:
                        pass

        def _watchdog() -> None:
            if _wait_until(done, timeout_seconds):
                return
            aborted.set()
            if client_box:
                abort_openai_client(client_box[0])

        thread = threading.Thread(target=_do_request, daemon=True, name="llm-request")
        watchdog = threading.Thread(target=_watchdog, daemon=True, name="llm-timeout-watchdog")
        thread.start()
        watchdog.start()
        _wait_until(done, timeout_seconds)

        if thread.is_alive() or aborted.is_set():
            aborted.set()
            if client_box:
                abort_openai_client(client_box[0])
            _wait_until(done, _ABORT_JOIN_SECONDS)
            thread.join(timeout=_ABORT_JOIN_SECONDS)
            raise TimeoutError(f"LLM request exceeded application timeout ({timeout_seconds:.1f}s)")

        if not result_box:
            raise RuntimeError("LLM request thread finished without result")

        status, payload = result_box[0]
        if status == "error":
            raise payload

        response = payload
        context.record_usage(self._normalize_usage(getattr(response, "usage", None)))
        choice = response.choices[0]
        message = choice.message
        raw_content = getattr(message, "content", None)
        content = raw_content if isinstance(raw_content, str) else ""
        finish_reason = str(getattr(choice, "finish_reason", "")).lower()
        if finish_reason == "length":
            raise TruncatedResponseError(
                "OpenAI-compatible response was truncated at the token limit"
            )
        if not content.strip():
            fallback = self._json_response_fallback(model_config, message) if json_mode else None
            if fallback is None or not fallback.strip():
                raise EmptyResponseError(f"{self.cfg.kind} response content is empty")
            try:
                json.loads(fallback)
            except json.JSONDecodeError as error:
                raise EmptyResponseError(
                    f"{self.cfg.kind} configured JSON fallback response is invalid JSON"
                ) from error
            content = fallback
        return content
