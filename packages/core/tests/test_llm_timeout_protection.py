"""Test application-layer timeout protection against stalled/dead HTTPS connections."""

import threading
import time
from unittest.mock import Mock, patch

import pytest
from pydantic import BaseModel, ConfigDict
from wenyi_core.llm.configuration import ProviderConfig
from wenyi_core.llm.providers._openai_compatible import OpenAICompatibleBaseClient
from wenyi_core.llm.transport import RequestContext, ResolvedModel


class FakeOptions(BaseModel):
    """Minimal options for testing."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class FakeClient(OpenAICompatibleBaseClient[FakeOptions]):
    """Minimal client for testing timeout protection."""

    def _build_request_kwargs(self, model_config, messages, *, json_mode, max_tokens):
        from wenyi_core.llm.providers._openai_compatible import base_request_kwargs

        return base_request_kwargs(model_config.model, messages, json_mode=json_mode)


def _wait_until_close(mock_sdk_client, timeout: float = 1.0) -> None:
    deadline = time.monotonic() + timeout
    while not mock_sdk_client.close.called and time.monotonic() < deadline:
        time.sleep(0.01)
    assert mock_sdk_client.close.called


def test_application_timeout_fires_on_stalled_connection():
    """Application timeout should fire when SDK call hangs beyond configured timeout.

    Regression test for stalled synopsis.chapter calls: ESTABLISHED HTTPS connection
    with no response body should become a retryable timeout, not hang indefinitely.
    """
    import threading

    # Use very short timeout for test speed
    cfg = ProviderConfig(kind="test-provider", base_url="https://example.com", timeout=2.0)
    client = FakeClient(cfg)
    client._credentials = ("test-key",)  # Tuple: (api_key,)

    # Mock OpenAI constructor to return mock SDK client (called per-request in v1246)
    mock_sdk_client = Mock()
    mock_completions = Mock()
    mock_sdk_client.chat.completions = mock_completions
    mock_sdk_client.close = Mock()  # Add close method

    # Simulate stalled connection: create() hangs for longer than timeout
    stop = threading.Event()

    def slow_create(**kwargs):
        stop.wait(10.0)
        if not mock_sdk_client.close.called:
            raise AssertionError("Should not reach here - timeout should fire first")

    mock_completions.create = slow_create

    threads_before = threading.active_count()

    with patch("openai.OpenAI", return_value=mock_sdk_client):
        messages = [{"role": "user", "content": "test"}]
        model = ResolvedModel(model="test-model", options=FakeOptions())
        context = RequestContext(
            operation="synopsis.chapter",
            tier="fast",
            max_tokens=1000,
            emit=Mock(),
            record_usage=Mock(),
            attempt_scope=Mock(),
        )

        # Should raise TimeoutError, not hang
        with pytest.raises(TimeoutError, match="exceeded application timeout"):
            client._request(messages, model, json_mode=False, context=context)
    stop.set()

    # Verify daemon thread doesn't block (test should complete quickly, not wait 10s)
    # Allow small grace period for thread cleanup
    time.sleep(0.5)
    threads_after = threading.active_count()
    # May have one daemon thread still running, that's OK (it won't block exit)
    assert threads_after <= threads_before + 1, f"Thread leak: {threads_before} -> {threads_after}"

    # Verify close was called on timeout
    _wait_until_close(mock_sdk_client)


def test_application_timeout_allows_slow_but_progressing_requests():
    """Application timeout should NOT fire for slow requests that complete within timeout.

    Ensures we don't kill legitimate long-running requests that are making progress.
    """
    cfg = ProviderConfig(kind="test-provider", base_url="https://example.com", timeout=3.0)
    client = FakeClient(cfg)
    client._credentials = ("test-key",)  # Tuple: (api_key,)

    # Mock OpenAI constructor to return mock SDK client
    mock_sdk_client = Mock()
    mock_completions = Mock()
    mock_sdk_client.chat.completions = mock_completions
    mock_sdk_client.close = Mock()

    # Simulate slow but successful request
    mock_response = Mock()
    mock_response.choices = [Mock(message=Mock(content="test response"), finish_reason="stop")]
    mock_response.usage = None

    def slow_create(**kwargs):
        time.sleep(1.5)  # Slow but within timeout
        return mock_response

    mock_completions.create = slow_create

    with patch("openai.OpenAI", return_value=mock_sdk_client):
        messages = [{"role": "user", "content": "test"}]
        model = ResolvedModel(model="test-model", options=FakeOptions())
        context = RequestContext(
            operation="synopsis.chapter",
            tier="fast",
            max_tokens=1000,
            emit=Mock(),
            record_usage=Mock(),
            attempt_scope=Mock(),
        )

        # Should complete successfully
        result = client._request(messages, model, json_mode=False, context=context)
        assert result == "test response"

        # Verify close was still called (in finally block)
        assert mock_sdk_client.close.called


def test_application_timeout_fires_on_trickle_response():
    """Application timeout should fire when server sends slow trickle response.

    Critical regression: server returns 200 headers then 1 byte/second trickle.
    Each byte arrives within httpx read timeout, so httpx doesn't fire.
    Application timeout must enforce total wall-clock deadline.

    Note: In mock environment, time.sleep() cannot be interrupted by client.close().
    Real socket read IS interrupted (verified by convergence_test_v1246.py).
    Test verifies timeout detection and close() call, not mock sleep interruption.
    """

    # Use very short timeout for test speed
    cfg = ProviderConfig(kind="test-provider", base_url="https://example.com", timeout=1.0)
    client = FakeClient(cfg)
    client._credentials = ("test-key",)  # Tuple: (api_key,)

    mock_sdk_client = Mock()
    mock_completions = Mock()
    mock_sdk_client.chat.completions = mock_completions
    close_called = []  # Track close() calls

    def mock_close():
        close_called.append(True)

    mock_sdk_client.close = mock_close

    # Simulate trickle response: send bytes slowly but continuously
    def trickle_create(**kwargs):
        # Sleep in small increments, simulating trickle (1 byte every 0.3s)
        for i in range(10):  # Would take 3 seconds total
            time.sleep(0.3)
            # In mock: sleep can't be interrupted. In real: socket read terminates.
            # Check if close was called (would interrupt real socket)
            if close_called:
                # Real behavior: socket closed, exit promptly
                return
        raise AssertionError("Should not reach here - timeout should fire first")

    mock_completions.create = trickle_create

    with patch("openai.OpenAI", return_value=mock_sdk_client):
        messages = [{"role": "user", "content": "test"}]
        model = ResolvedModel(model="test-model", options=FakeOptions())
        context = RequestContext(
            operation="synopsis.chapter",
            tier="fast",
            max_tokens=1000,
            emit=Mock(),
            record_usage=Mock(),
            attempt_scope=Mock(),
        )

        start = time.time()
        with pytest.raises(TimeoutError, match="exceeded application timeout"):
            client._request(messages, model, json_mode=False, context=context)
        elapsed = time.time() - start

        # Verify timeout fired around configured timeout (1.0s * 1.1 = 1.1s)
        # With mock checking close_called, should exit after ~1.2s (1.0s + 0.3s*1 loop)
        assert elapsed < 2.0, f"Timeout took too long: {elapsed:.2f}s (expected ~1.2s)"
        assert elapsed > 1.0, f"Timeout fired too early: {elapsed:.2f}s (expected ~1.2s)"

        # Verify close was called on timeout
        assert len(close_called) > 0, "close() should have been called on timeout"


def test_timeout_error_is_retryable():
    """TimeoutError from application timeout should be classified as retryable.

    Ensures the timeout becomes a retryable error that triggers llm_retry_wait.
    """
    from wenyi_core.llm.retrying import retry_reason

    error = TimeoutError("LLM request exceeded application timeout")
    reason = retry_reason(error)

    assert reason == "timeout", f"Expected 'timeout', got {reason!r}"


def test_daemon_thread_allows_process_exit():
    """Daemon threads from stalled requests must not block process exit.

    Critical: after timeout, daemon thread continues in background but doesn't
    prevent main thread from continuing or process from exiting cleanly.

    v1246 improves on this: close() should interrupt socket, causing thread exit.
    """

    cfg = ProviderConfig(kind="test-provider", base_url="https://example.com", timeout=0.5)
    client = FakeClient(cfg)
    client._credentials = ("test-key",)  # Tuple: (api_key,)

    mock_sdk_client = Mock()
    mock_completions = Mock()
    mock_sdk_client.chat.completions = mock_completions
    mock_sdk_client.close = Mock()

    # Simulate long-running request that never completes
    stop = threading.Event()

    def infinite_request(**kwargs):
        stop.wait(3600)

    mock_completions.create = infinite_request

    with patch("openai.OpenAI", return_value=mock_sdk_client):
        messages = [{"role": "user", "content": "test"}]
        model = ResolvedModel(model="test-model", options=FakeOptions())
        context = RequestContext(
            operation="synopsis.chapter",
            tier="fast",
            max_tokens=1000,
            emit=Mock(),
            record_usage=Mock(),
            attempt_scope=Mock(),
        )

        # Timeout should fire quickly
        with pytest.raises(TimeoutError):
            client._request(messages, model, json_mode=False, context=context)

        # Verify close was called to interrupt socket
        _wait_until_close(mock_sdk_client)
    stop.set()

    # Main thread continues immediately (daemon thread still running in background)
    # Test completes without waiting for daemon thread


def test_httpx_timeout_configuration():
    """Verify httpx.Timeout is properly configured with all timeout components."""
    cfg = ProviderConfig(kind="test-provider", base_url="https://example.com", timeout=900.0)
    client = FakeClient(cfg)

    # Trigger client creation with patched OpenAI (imported inside _ensure_client)
    with patch("openai.OpenAI") as mock_openai:
        mock_instance = Mock()
        mock_openai.return_value = mock_instance

        # Ensure client creation
        client._ensure_client()

        # Verify OpenAI was called with proper timeout config
        assert mock_openai.called
        call_kwargs = mock_openai.call_args[1]

        # Check timeout is httpx.Timeout object, not just float
        timeout_arg = call_kwargs.get("timeout")
        assert timeout_arg is not None

        # Verify it's an httpx.Timeout object with proper read timeout
        import httpx

        assert isinstance(timeout_arg, httpx.Timeout)
        assert timeout_arg.read == 900.0, f"Expected read timeout 900.0, got {timeout_arg.read}"
        assert timeout_arg.write == 900.0
        assert timeout_arg.connect == 60.0
        assert timeout_arg.pool == 60.0
        http_client = call_kwargs.get("http_client")
        assert http_client is not None
        options = list(
            getattr(getattr(http_client, "_transport", None), "_pool", None)._socket_options or []
        )
        import socket as socket_mod

        assert any(option[1] == socket_mod.SO_RCVTIMEO for option in options)
        assert any(option[1] == socket_mod.SO_SNDTIMEO for option in options)


def _timeout_context():
    return RequestContext(
        operation="review.scan",
        tier="fast",
        max_tokens=1000,
        emit=Mock(),
        record_usage=Mock(),
        attempt_scope=Mock(),
    )


def test_timeout_does_not_wait_for_blocking_close():
    """SSL_shutdown/close() must not delay TimeoutError on the calling thread.

    Production v1257: wait returned, then client.close() blocked ~20 minutes on a
    silent Gemini socket, so attempt 2 never started after json_mode_selected.
    """
    cfg = ProviderConfig(kind="test-provider", base_url="https://example.com", timeout=1.0)
    client = FakeClient(cfg)
    client._credentials = ("test-key",)

    mock_sdk_client = Mock()
    stop = threading.Event()
    mock_sdk_client.chat.completions.create = lambda **_kwargs: stop.wait(30)

    def hanging_close():
        stop.wait(30)

    mock_sdk_client.close = hanging_close

    started = time.monotonic()
    with patch("openai.OpenAI", return_value=mock_sdk_client):
        with pytest.raises(TimeoutError, match="exceeded application timeout"):
            client._request(
                [{"role": "user", "content": "scan"}],
                ResolvedModel(model="test-model", options=FakeOptions()),
                json_mode=True,
                context=_timeout_context(),
            )
    elapsed = time.monotonic() - started
    stop.set()
    assert elapsed < 4.0, f"TimeoutError waited for close(): {elapsed:.2f}s"


def test_timeout_covers_client_construction_before_http():
    """Hang in OpenAI()/TLS setup after json_mode_selected must still time out."""
    cfg = ProviderConfig(kind="test-provider", base_url="https://example.com", timeout=1.0)
    client = FakeClient(cfg)
    client._credentials = ("test-key",)

    stop = threading.Event()

    def hanging_openai(**_kwargs):
        stop.wait(30)
        raise AssertionError("constructor should have been aborted")

    started = time.monotonic()
    with patch("openai.OpenAI", hanging_openai):
        with pytest.raises(TimeoutError, match="exceeded application timeout"):
            client._request(
                [{"role": "user", "content": "scan"}],
                ResolvedModel(model="test-model", options=FakeOptions()),
                json_mode=True,
                context=_timeout_context(),
            )
    elapsed = time.monotonic() - started
    stop.set()
    assert elapsed < 4.0, f"constructor hang was not bounded: {elapsed:.2f}s"


def test_timeout_covers_json_request_kwargs_build():
    """Building json_object kwargs is inside the watchdog, not before it."""
    cfg = ProviderConfig(kind="test-provider", base_url="https://example.com", timeout=1.0)
    client = FakeClient(cfg)
    client._credentials = ("test-key",)

    stop = threading.Event()

    def hanging_kwargs(*_args, **_kwargs):
        stop.wait(30)
        return {}

    started = time.monotonic()
    with (
        patch.object(client, "_build_request_kwargs", hanging_kwargs),
        patch("openai.OpenAI", return_value=Mock()),
    ):
        with pytest.raises(TimeoutError, match="exceeded application timeout"):
            client._request(
                [{"role": "user", "content": "x" * 8000}],
                ResolvedModel(model="test-model", options=FakeOptions()),
                json_mode=True,
                context=_timeout_context(),
            )
    elapsed = time.monotonic() - started
    stop.set()
    assert elapsed < 4.0, f"kwargs build hang was not bounded: {elapsed:.2f}s"
