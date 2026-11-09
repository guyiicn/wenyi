"""OpenAI-compatible wire protocol and single-attempt response handling."""

from __future__ import annotations

import json
import logging
import multiprocessing
import os
import socket
import ssl
import struct
import threading
import time
from abc import abstractmethod
from contextlib import contextmanager
from typing import Any, Generic, Iterator, TypeVar

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
_WATCHDOG_POLL_SECONDS = 0.05
_IO_SLICE_SECONDS = 1.0
_DEFAULT_HARD_DEADLINE_S = 240.0
_HARD_DEADLINE_ENV = "WENYI_LLM_HARD_DEADLINE_S"
_ISOLATED_ATTEMPT_ENV = "WENYI_LLM_ISOLATED_ATTEMPT"
_SOCKET_WALK_LIMIT = 1024
_LOGGER = logging.getLogger(__name__)
_SSL_POLL_LOCK = threading.Lock()
_SSL_POLL_DEPTH = 0
_SSL_POLL_ORIG: tuple[Any, Any] | None = None


def hard_deadline_seconds(timeout: float) -> float:
    """Wall-clock cap for one attempt. Not part of ProviderConfig / digest identity.

    Production watchdog kills the process after 300s of silence, so a single
    attempt must finish well before that. Default ``min(timeout, 240)``.
    Override with ``WENYI_LLM_HARD_DEADLINE_S``.
    """
    raw = os.environ.get(_HARD_DEADLINE_ENV, "").strip()
    if raw:
        try:
            return max(0.001, float(raw))
        except ValueError:
            pass
    return min(max(0.001, float(timeout)), _DEFAULT_HARD_DEADLINE_S)


def application_timeout_seconds(timeout: float) -> float:
    """Wall-clock bound used when httpx/SSL read timeouts do not fire."""
    soft = float(timeout) * _APPLICATION_TIMEOUT_FACTOR
    return min(soft, hard_deadline_seconds(timeout))


def httpx_timeout_config(timeout: float):
    """Split connect/read/write/pool deadlines so a stalled peer cannot hang one phase."""
    import httpx

    bounded = max(0.001, min(float(timeout), hard_deadline_seconds(timeout)))
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


def _cap_io_timeout(timeout: float | None, remaining: float) -> float:
    """Never pass None/infinite into SSLSocket.settimeout: that poll()s forever."""
    bounded = max(0.001, remaining)
    if timeout is None:
        return bounded
    return max(0.001, min(float(timeout), bounded))


def _clamp_ssl_timeout(value: float | None, cap: float) -> float:
    """Keep SSL poll() slices short. 0 stays non-blocking for abort()."""
    if value is None:
        return cap
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return cap
    if numeric < 0:
        return cap
    if numeric == 0:
        return 0.0
    return min(numeric, cap)


@contextmanager
def bound_ssl_polls(max_seconds: float = _IO_SLICE_SECONDS) -> Iterator[None]:
    """Force SSLSocket.settimeout to ≤1s so CPython 3.13 _ssl cannot poll(330s).

    httpcore calls ``sock.settimeout(read_timeout)`` inside SyncStream.read.
    v1273 passed remaining≈330s there; production poll() in ``_ssl`` then
    outlived both ``_wait_until`` and the 300s process watchdog. Clamp even
    if ``_DeadlineStream`` wrapping is skipped. Refcounted so concurrent
    attempts cannot restore the uncapped method while another is in flight.
    """

    global _SSL_POLL_DEPTH, _SSL_POLL_ORIG
    cap = max(0.001, float(max_seconds))
    with _SSL_POLL_LOCK:
        if _SSL_POLL_DEPTH == 0:
            _SSL_POLL_ORIG = (ssl.SSLSocket.settimeout, socket.socket.settimeout)

            def _capped_ssl(self: ssl.SSLSocket, value: float | None) -> None:
                orig = _SSL_POLL_ORIG[0] if _SSL_POLL_ORIG is not None else ssl.SSLSocket.settimeout
                orig(self, _clamp_ssl_timeout(value, cap))

            def _capped_sock(self: socket.socket, value: float | None) -> None:
                orig = _SSL_POLL_ORIG[1] if _SSL_POLL_ORIG is not None else socket.socket.settimeout
                orig(self, _clamp_ssl_timeout(value, cap))

            ssl.SSLSocket.settimeout = _capped_ssl  # type: ignore[method-assign]
            socket.socket.settimeout = _capped_sock  # type: ignore[method-assign]
        _SSL_POLL_DEPTH += 1
    try:
        yield
    finally:
        with _SSL_POLL_LOCK:
            _SSL_POLL_DEPTH -= 1
            if _SSL_POLL_DEPTH == 0 and _SSL_POLL_ORIG is not None:
                ssl.SSLSocket.settimeout = _SSL_POLL_ORIG[0]
                socket.socket.settimeout = _SSL_POLL_ORIG[1]
                _SSL_POLL_ORIG = None


def _is_timeout_error(error: BaseException) -> bool:
    if isinstance(error, (TimeoutError, socket.timeout)):
        return True
    module = getattr(type(error), "__module__", "") or ""
    name = type(error).__name__.lower()
    return "timeout" in name and ("httpcore" in module or "httpx" in module)


class _DeadlineStream:
    """Cap every TCP/TLS I/O slice so a silent peer cannot pin the interpreter.

    httpcore calls ``SSLSocket.settimeout(read_timeout)`` per recv, which becomes
    a C-level ``poll()`` in CPython 3.13 ``_ssl``. That poll does release the
    GIL (``Py_BEGIN_ALLOW_THREADS``), but v1273 passed the entire remaining
    window (~330s) as the poll timeout. MainThread/watchdog then sat in
    ``clock_nanosleep`` inside ``_wait_until`` while the worker stayed in
    ``SSLSocket.read``; at 354s the 330s bound had still not raised. Slice
    each SSL poll to 1s so the deadline check can run, and re-apply
    ``SO_RCVTIMEO`` after wrap/settimeout.
    """

    def __init__(self, inner: Any, deadline: float, socket_options: list[Any]) -> None:
        self._inner = inner
        self._deadline = deadline
        self._socket_options = socket_options
        self._apply_kernel_timeouts()

    def _remaining(self) -> float:
        return self._deadline - time.monotonic()

    def _apply_kernel_timeouts(self, seconds: float | None = None) -> None:
        sock = None
        extra = getattr(self._inner, "get_extra_info", None)
        if callable(extra):
            try:
                sock = extra("socket")
            except Exception:
                sock = None
        if not isinstance(sock, socket.socket):
            sock = getattr(self._inner, "_sock", None)
        if not isinstance(sock, socket.socket):
            return
        options = (
            os_socket_timeout_options(seconds) if seconds is not None else self._socket_options
        )
        for option in options:
            try:
                sock.setsockopt(*option)
            except Exception:
                pass

    def _raise_if_expired(self) -> float:
        remaining = self._remaining()
        if remaining <= 0:
            raise socket.timeout("LLM application deadline exceeded")
        return remaining

    def _slice_timeout(self, timeout: float | None) -> float:
        remaining = self._raise_if_expired()
        return _cap_io_timeout(timeout, min(remaining, _IO_SLICE_SECONDS))

    def read(self, max_bytes: int, timeout: float | None = None) -> bytes:
        # v1273: one poll(remaining) of ~330s held the GIL in CPython 3.13.5
        # SSL_read, so MainThread/watchdog time.sleep could not resume past
        # the deadline. Slice each SSL poll to 1s and retry until the wall cap.
        while True:
            cap = self._slice_timeout(timeout)
            self._apply_kernel_timeouts(cap)
            try:
                return self._inner.read(max_bytes, timeout=cap)
            except BaseException as error:
                if self._remaining() <= 0 or not _is_timeout_error(error):
                    raise

    def write(self, buffer: bytes, timeout: float | None = None) -> None:
        # Do not retry a timed-out write: sendall may already have pushed a prefix.
        remaining = self._raise_if_expired()
        cap = _cap_io_timeout(timeout, min(remaining, 60.0))
        self._apply_kernel_timeouts(cap)
        return self._inner.write(buffer, timeout=cap)

    def close(self) -> None:
        self._inner.close()

    def start_tls(
        self,
        ssl_context: Any,
        server_hostname: str | None = None,
        timeout: float | None = None,
    ) -> _DeadlineStream:
        remaining = self._raise_if_expired()
        inner = self._inner.start_tls(
            ssl_context,
            server_hostname=server_hostname,
            timeout=_cap_io_timeout(timeout, min(remaining, 60.0)),
        )
        wrapped = _DeadlineStream(inner, self._deadline, self._socket_options)
        wrapped._apply_kernel_timeouts()
        return wrapped

    def get_extra_info(self, info: str) -> Any:
        extra = getattr(self._inner, "get_extra_info", None)
        if callable(extra):
            return extra(info)
        return None


class _DeadlineBackend:
    def __init__(self, inner: Any, deadline: float, socket_options: list[Any]) -> None:
        self._inner = inner
        self._deadline = deadline
        self._socket_options = socket_options

    def _backend(self) -> Any:
        if self._inner is None:
            import httpcore

            self._inner = httpcore.SyncBackend()
        return self._inner

    def connect_tcp(self, *args: Any, **kwargs: Any) -> _DeadlineStream:
        return _DeadlineStream(
            self._backend().connect_tcp(*args, **kwargs),
            self._deadline,
            self._socket_options,
        )

    def connect_unix_socket(self, *args: Any, **kwargs: Any) -> _DeadlineStream:
        return _DeadlineStream(
            self._backend().connect_unix_socket(*args, **kwargs),
            self._deadline,
            self._socket_options,
        )

    def sleep(self, seconds: float) -> None:
        remaining = self._deadline - time.monotonic()
        if remaining <= 0:
            return
        time.sleep(min(max(0.0, seconds), remaining))


def attach_deadline_backend(transport: Any, deadline: float, socket_options: list[Any]) -> None:
    """Wrap httpx/httpcore TCP+TLS streams with a remaining-time deadline."""
    pool = getattr(transport, "_pool", None)
    if pool is None:
        return
    inner = getattr(pool, "_network_backend", None)
    pool._network_backend = _DeadlineBackend(inner, deadline, socket_options)


def _force_close_socket(sock: socket.socket) -> None:
    """TCP-close without SSL_shutdown so a silent peer cannot block this thread."""
    try:
        sock._sslobj = None  # type: ignore[attr-defined]
    except Exception:
        pass
    try:
        sock.settimeout(0)
    except Exception:
        pass
    try:
        sock.shutdown(socket.SHUT_RDWR)
    except Exception:
        pass
    fd = -1
    try:
        fd = sock.fileno()
    except Exception:
        pass
    if fd >= 0:
        try:
            os.close(fd)
        except Exception:
            pass
    try:
        socket.socket.close(sock)
    except Exception:
        pass


def _sockets_from_httpx_client(http_client: Any) -> list[socket.socket]:
    """Read in-flight httpcore sockets without traversing the OpenAI object graph."""
    found: list[socket.socket] = []
    try:
        transport = getattr(http_client, "_transport", None)
        pool = getattr(transport, "_pool", None)
        connections = list(getattr(pool, "_connections", None) or [])
    except Exception:
        return found
    for connection in connections:
        inner = getattr(connection, "_connection", connection)
        stream = getattr(inner, "_network_stream", None)
        sock = None
        extra = getattr(stream, "get_extra_info", None)
        if callable(extra):
            try:
                sock = extra("socket")
            except Exception:
                sock = None
        if sock is None:
            sock = getattr(stream, "_sock", None)
        if isinstance(sock, socket.socket):
            found.append(sock)
    return found


def _iter_network_sockets(root: Any) -> list[socket.socket]:
    """Collect live TCP/SSL sockets from an OpenAI/httpx/httpcore client tree."""
    found: list[socket.socket] = []
    seen: set[int] = set()
    stack: list[Any] = list(root) if isinstance(root, (list, tuple)) else [root]
    while stack and len(seen) < _SOCKET_WALK_LIMIT:
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
        if module.startswith(("httpx", "httpcore")):
            try:
                stack.extend(vars(current).values())
            except Exception:
                pass
        for attr in (
            "_wenyi_http_client",
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
    """Unblock SSL_read/handshake. Never SSL_shutdown on this thread.

    ``httpx.Client.close()`` / ``OpenAI.close()`` issue SSL_shutdown and wait for
    the peer's close_notify. A silent Gemini socket never replies, so that wait
    can last until TCP gives up (~20 minutes) and blocks the timeout path.
    """
    if client is None:
        return
    http_client = getattr(client, "_wenyi_http_client", None) or getattr(client, "_client", None)
    sockets: list[socket.socket] = []
    if http_client is not None:
        sockets.extend(_sockets_from_httpx_client(http_client))
    sockets.extend(_iter_network_sockets([client, http_client]))
    seen_socks: set[int] = set()
    for sock in sockets:
        ident = id(sock)
        if ident in seen_socks:
            continue
        seen_socks.add(ident)
        _force_close_socket(sock)

    def _close_later() -> None:
        if http_client is not None:
            try:
                http_client.close()
            except Exception:
                pass
        try:
            client.close()
        except Exception:
            pass

    threading.Thread(target=_close_later, daemon=True, name="llm-client-close").start()


def isolated_attempts_enabled() -> bool:
    """Run one SDK attempt in a child process so a GIL-holding SSL poll cannot
    pin the caller. Not part of ProviderConfig / digest identity.

    v1274 hung three times (15:13 PID 363675 result-10, 15:21 PID 368348
    result-11, 15:27 PID 370971 result-12): MainThread and
    llm-timeout-watchdog stayed in ``_wait_until`` while the worker was in
    SSL_read. Sleep/join in the parent interpreter cannot preempt a CPython
    3.13 ``_ssl`` poll that holds the GIL.
    A spawned child has its own interpreter; the parent ``Pipe.poll`` then
    ``SIGKILL`` does not need that GIL. Test doubles that replace ``openai.OpenAI``
    stay in-process. Override with ``WENYI_LLM_ISOLATED_ATTEMPT=0``.
    """
    raw = os.environ.get(_ISOLATED_ATTEMPT_ENV, "1").strip().lower()
    if raw in {"0", "false", "no", "off"}:
        return False
    try:
        import openai
    except Exception:
        return False
    ctor = getattr(openai, "OpenAI", None)
    if not isinstance(ctor, type):
        return False
    module = getattr(ctor, "__module__", "") or ""
    if module != "openai" and not module.startswith("openai."):
        return False
    if getattr(ctor, "_mock_name", None) is not None:
        return False
    return True


def build_openai_client(
    *,
    api_key: str,
    base_url: str,
    timeout: float,
    deadline: float | None,
    sdk_extra: dict[str, Any] | None = None,
) -> Any:
    """Construct one isolated OpenAI+httpx client. Callers must close it."""
    import httpx
    from openai import OpenAI

    timeout_config = httpx_timeout_config(timeout)
    socket_options = os_socket_timeout_options(hard_deadline_seconds(timeout))
    transport = httpx.HTTPTransport(
        retries=0,
        http1=True,
        http2=False,
        socket_options=socket_options,
        limits=httpx.Limits(max_keepalive_connections=0, max_connections=4),
    )
    if deadline is not None:
        attach_deadline_backend(transport, deadline, socket_options)
    http_client = httpx.Client(
        timeout=timeout_config,
        transport=transport,
        follow_redirects=True,
        trust_env=True,
    )
    try:
        client = OpenAI(
            api_key=api_key or "no-key",
            base_url=base_url,
            timeout=timeout_config,
            max_retries=0,
            http_client=http_client,
            **(sdk_extra or {}),
        )
    except BaseException:
        http_client.close()
        raise
    if getattr(client, "_client", None) is not http_client:
        http_client.close()
    else:
        try:
            client._wenyi_http_client = http_client
        except Exception:
            pass
    return client


def isolated_openai_chat_completion(payload: dict[str, Any]) -> dict[str, Any]:
    """Child-process body: one Chat Completions call, then exit."""
    remaining = max(0.001, float(payload["remaining_s"]))
    deadline = time.monotonic() + remaining
    client = build_openai_client(
        api_key=str(payload.get("api_key") or "no-key"),
        base_url=str(payload["base_url"]),
        timeout=float(payload["timeout"]),
        deadline=deadline,
        sdk_extra=dict(payload.get("sdk_extra") or {}),
    )
    try:
        with bound_ssl_polls(_IO_SLICE_SECONDS):
            response = client.chat.completions.create(**payload["request_kwargs"])
        choice = response.choices[0]
        message = choice.message
        raw_content = getattr(message, "content", None)
        usage = getattr(response, "usage", None)
        if usage is None:
            usage_data = None
        elif hasattr(usage, "model_dump"):
            usage_data = usage.model_dump()
        elif isinstance(usage, dict):
            usage_data = usage
        else:
            usage_data = {
                key: getattr(usage, key, None)
                for key in (
                    "prompt_tokens",
                    "completion_tokens",
                    "total_tokens",
                    "prompt_tokens_details",
                    "cache_read_input_tokens",
                    "cache_creation_input_tokens",
                    "cache_creation",
                )
            }
        return {
            "ok": True,
            "content": raw_content if isinstance(raw_content, str) else "",
            "finish_reason": str(getattr(choice, "finish_reason", "") or ""),
            "usage": usage_data,
        }
    finally:
        abort_openai_client(client)


def _isolated_attempt_main(payload: dict[str, Any], conn: Any) -> None:
    try:
        conn.send(isolated_openai_chat_completion(payload))
    except BaseException as error:
        conn.send(
            {
                "ok": False,
                "error_type": type(error).__name__,
                "message": str(error),
                "timeout": isinstance(error, (TimeoutError, socket.timeout))
                or "timeout" in type(error).__name__.lower(),
                "truncated": isinstance(error, TruncatedResponseError),
                "empty": isinstance(error, EmptyResponseError),
            }
        )
    finally:
        try:
            conn.close()
        except Exception:
            pass


def _wait_until(event: threading.Event, timeout: float) -> bool:
    """Sleep to a monotonic deadline. Do not use Event.wait(timeout).

    v1257 production (CPython 3.13.5): py-spy showed MainThread and
    llm-timeout-watchdog parked in Event.wait while llm-request sat in
    SSLSocket.read → poll() waiting for response headers. Event.wait's
    timeout did not recover, so the 330s bound never fired. time.sleep
    releases the GIL independently of Condition locks.
    """
    deadline = time.monotonic() + max(0.0, timeout)
    while not event.is_set():
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        time.sleep(min(_WATCHDOG_POLL_SECONDS, remaining))
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

    def _create_openai_client(self, *, deadline: float | None = None) -> Any:
        """Build one isolated SDK client. Callers must close it."""
        self.validate_credentials()
        return build_openai_client(
            api_key=self.api_key() or "no-key",
            base_url=self.base_url,
            timeout=self.cfg.timeout,
            deadline=deadline,
            sdk_extra=self._openai_sdk_kwargs(),
        )

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

    def _request_isolated(
        self,
        request_kwargs: dict[str, Any],
        *,
        timeout_seconds: float,
    ) -> dict[str, Any]:
        """Parent wait that does not share a GIL with SSL_read."""
        import wenyi_core

        core_root = os.path.dirname(os.path.dirname(os.path.abspath(wenyi_core.__file__)))
        previous_pythonpath = os.environ.get("PYTHONPATH")
        pythonpath = previous_pythonpath or ""
        mutated_pythonpath = False
        if core_root not in pythonpath.split(os.pathsep):
            os.environ["PYTHONPATH"] = (
                core_root + os.pathsep + pythonpath if pythonpath else core_root
            )
            mutated_pythonpath = True
        remaining = max(0.001, timeout_seconds)
        payload = {
            "api_key": self.api_key() or "no-key",
            "base_url": self.base_url,
            "timeout": self.cfg.timeout,
            "remaining_s": remaining,
            "request_kwargs": request_kwargs,
            "sdk_extra": self._openai_sdk_kwargs(),
        }
        ctx = multiprocessing.get_context("spawn")
        parent_conn, child_conn = ctx.Pipe(duplex=False)
        proc = ctx.Process(
            target=_isolated_attempt_main,
            args=(payload, child_conn),
            daemon=True,
            name="llm-isolated-attempt",
        )
        try:
            proc.start()
            child_conn.close()
            child_conn = None
            if not parent_conn.poll(remaining):
                raise TimeoutError(
                    f"LLM request exceeded application timeout ({timeout_seconds:.1f}s)"
                )
            try:
                result = parent_conn.recv()
            except EOFError as error:
                raise TimeoutError(
                    f"LLM request exceeded application timeout ({timeout_seconds:.1f}s)"
                ) from error
        finally:
            if mutated_pythonpath:
                if previous_pythonpath is None:
                    os.environ.pop("PYTHONPATH", None)
                else:
                    os.environ["PYTHONPATH"] = previous_pythonpath
            if child_conn is not None:
                try:
                    child_conn.close()
                except Exception:
                    pass
            if proc.pid is not None:
                if proc.is_alive():
                    proc.kill()
                proc.join(5)
            try:
                parent_conn.close()
            except Exception:
                pass
            closer = getattr(proc, "close", None)
            if callable(closer):
                try:
                    closer()
                except Exception:
                    pass
        if not isinstance(result, dict):
            raise RuntimeError("LLM isolated attempt returned an invalid payload")
        if result.get("ok"):
            return result
        message = str(result.get("message") or "LLM isolated attempt failed")
        if result.get("timeout"):
            raise TimeoutError(message)
        if result.get("truncated"):
            raise TruncatedResponseError(message)
        if result.get("empty"):
            raise EmptyResponseError(message)
        raise RuntimeError(f"{result.get('error_type') or 'LLMError'}: {message}")

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
        timeout_seconds = application_timeout_seconds(self.cfg.timeout)
        deadline = time.monotonic() + timeout_seconds

        if isolated_attempts_enabled():
            request_kwargs = self._build_request_kwargs(
                model, messages, json_mode=json_mode, max_tokens=context.max_tokens
            )
            if response_format is not None:
                request_kwargs["response_format"] = response_format
            isolated = self._request_isolated(request_kwargs, timeout_seconds=timeout_seconds)
            usage = isolated.get("usage")
            context.record_usage(self._normalize_usage(usage))
            content = isolated.get("content") if isinstance(isolated.get("content"), str) else ""
            finish_reason = str(isolated.get("finish_reason") or "").lower()
            if finish_reason == "length":
                raise TruncatedResponseError(
                    "OpenAI-compatible response was truncated at the token limit"
                )
            if not content.strip():
                fallback = self._json_response_fallback(model_config, None) if json_mode else None
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

        # Watchdog starts before client construction, TLS, and the HTTP write.
        # Used when OpenAI is a test double; production uses an isolated child.
        result_box: list[Any] = []
        client_box: list[Any] = []
        done = threading.Event()
        aborted = threading.Event()

        def _do_request() -> None:
            client = None
            try:
                if aborted.is_set():
                    raise TimeoutError(
                        f"LLM request exceeded application timeout ({timeout_seconds:.1f}s)"
                    )
                request_kwargs = self._build_request_kwargs(
                    model, messages, json_mode=json_mode, max_tokens=context.max_tokens
                )
                if response_format is not None:
                    request_kwargs["response_format"] = response_format
                client = self._create_openai_client(deadline=deadline)
                client_box.append(client)
                if aborted.is_set():
                    abort_openai_client(client)
                    raise TimeoutError(
                        f"LLM request exceeded application timeout ({timeout_seconds:.1f}s)"
                    )
                with bound_ssl_polls(_IO_SLICE_SECONDS):
                    response = client.chat.completions.create(**request_kwargs)
                result_box.append(("success", response))
            except BaseException as error:
                result_box.append(("error", error))
            finally:
                if client is not None:
                    if aborted.is_set():
                        abort_openai_client(client)
                    else:
                        try:
                            client.close()
                        except Exception:
                            pass
                done.set()

        def _watchdog() -> None:
            if _wait_until(done, timeout_seconds):
                return
            aborted.set()
            _LOGGER.warning(
                "LLM request watchdog fired after %.1fs (operation=%s json_mode=%s)",
                timeout_seconds,
                context.operation,
                json_mode,
            )
            if client_box:
                abort_openai_client(client_box[0])

        thread = threading.Thread(target=_do_request, daemon=True, name="llm-request")
        watchdog = threading.Thread(target=_watchdog, daemon=True, name="llm-timeout-watchdog")
        watchdog.start()
        thread.start()
        _wait_until(done, timeout_seconds)

        finished_ok = bool(result_box) and result_box[0][0] == "success" and not aborted.is_set()
        if (thread.is_alive() or aborted.is_set()) and not finished_ok:
            aborted.set()
            if client_box:
                abort_openai_client(client_box[0])
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
