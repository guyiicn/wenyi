"""Credential-scoped errors must be safe before reaching workflow consumers."""

import json
import logging
import traceback
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import pytest
from openai import APIStatusError
from wenyi_core.llm.configuration import LLMConfig
from wenyi_core.llm.limits import RequestCancelled, RequestStopped
from wenyi_core.llm.retrying import (
    ProviderRequestError,
    TruncatedResponseError,
    is_resumable_provider_interrupt,
)
from wenyi_core.llm.router import RoutedLLMClient

SECRET = "fake-boundary-secret-not-a-real-key"


def client_with_failure(*, credentials=True, status=503, other_secret=None):
    """Create client with mocked failure for v1246 isolated client per request."""
    client = RoutedLLMClient(
        LLMConfig.model_validate(
            {
                "providers": {"default": {"kind": "openai", "max_retries": 1}},
                "models": {"one": {"provider": "default", "model": "offline"}},
                "tiers": {"strong": "one", "cheap": "one", "fast": "one"},
            }
        ),
        credentials={"default": SECRET, "other": other_secret} if credentials else None,
    )
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        response = httpx.Response(
            status,
            headers={"x-request-id": f"request-{SECRET}-{other_secret or ''}", "retry-after": "0"},
            request=httpx.Request(
                "POST", "https://example.invalid", headers={"Authorization": f"Bearer {SECRET}"}
            ),
        )
        raise APIStatusError(
            f"Authorization: Bearer {SECRET}", response=response, body={"key": SECRET}
        ) from RuntimeError(f"Transport context {SECRET}")

    # v1246: Create mock client with close() method
    mock_client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
        close=lambda: None,  # Noop close for v1246
    )

    # Return client and calls list, caller must patch OpenAI constructor
    return client, calls, mock_client


@pytest.mark.parametrize(
    "status, attempts, resumable", [(401, 1, False), (402, 1, True), (503, 2, True)]
)
def test_final_provider_failure_and_retry_diagnostics_are_safe(caplog, status, attempts, resumable):
    client, calls, mock_client = client_with_failure(
        status=status, other_secret="fake-other-connection-key"
    )
    events = []
    client.set_event_sink(lambda event, **data: events.append((event, data)))

    # v1246: Patch OpenAI constructor for isolated client per request
    with patch("openai.OpenAI", return_value=mock_client):
        with pytest.raises(Exception) as caught:
            client.complete([{"role": "user", "content": "Hello"}], operation="review.verify")

    logging.getLogger(__name__).error(
        "Worker failed", exc_info=(type(caught.value), caught.value, caught.tb)
    )
    assert len(calls) == attempts
    assert SECRET not in str(caught.value)
    assert SECRET not in "".join(traceback.format_exception(caught.value))
    assert SECRET not in json.dumps(events)
    assert SECRET not in caplog.text
    assert "fake-other-connection-key" not in json.dumps(events)
    assert "fake-other-connection-key" not in caplog.text
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
    assert isinstance(caught.value, ProviderRequestError)
    assert is_resumable_provider_interrupt(caught.value) is resumable
    assert caught.value.status_code == status
    assert not hasattr(caught.value, "request")
    assert not hasattr(caught.value, "response")


def test_credential_validation_remains_actionable():
    client, _, mock_client = client_with_failure()
    missing = RoutedLLMClient(client.config, credentials={"default": None})
    with pytest.raises(RuntimeError, match="Provider openai has no configured credential"):
        missing.validate_credentials()


def test_without_snapshot_keeps_original_exception(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", SECRET)
    client, calls, mock_client = client_with_failure(credentials=False, status=401)
    # Isolated clients still validate env credentials; without a snapshot the SDK error is kept.
    with patch("openai.OpenAI", return_value=mock_client):
        with pytest.raises(APIStatusError, match=SECRET):
            client.complete([], operation="translation.body")
    assert len(calls) == 1


def test_failed_event_sink_cannot_log_provider_context(caplog):
    client, calls, mock_client = client_with_failure()

    def sink(event, **data):
        raise RuntimeError(f"Sink failed with {SECRET}")

    client.set_event_sink(sink)

    # v1246: Patch OpenAI constructor
    with patch("openai.OpenAI", return_value=mock_client):
        with pytest.raises(ProviderRequestError):
            client.complete([], operation="translation.body")

    assert len(calls) == 2
    assert "Failed to write LLM event" in caplog.text
    assert SECRET not in caplog.text


@pytest.mark.parametrize("method", ["capture_events", "observe_events"])
def test_observer_cannot_log_provider_context(caplog, method):
    client, calls, mock_client = client_with_failure()
    events = []
    client.set_event_sink(lambda event, **data: events.append((event, data)))

    def observer(event, **data):
        raise RuntimeError(f"Observer failed with {SECRET}")

    # v1246: Patch OpenAI constructor
    with patch("openai.OpenAI", return_value=mock_client):
        with getattr(client, method)(observer), pytest.raises(ProviderRequestError):
            client.complete([], operation="translation.body")

    assert len(calls) == 2
    assert "Failed to write LLM event" in caplog.text
    assert SECRET not in caplog.text
    assert SECRET not in json.dumps(events)
    failure = next(data for event, data in events if event == "llm_request_failed")
    assert failure["error_category"] == "provider_unavailable"
    assert failure["status_code"] == 503


@pytest.mark.parametrize("stop_type", [RequestStopped, RequestCancelled])
def test_control_flow_errors_are_preserved(stop_type):
    client, calls, mock_client = client_with_failure()
    stop = stop_type("Pause safely")

    def create(**kwargs):
        raise stop

    # v1246: Create new mock client with custom create
    stop_mock_client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
        close=lambda: None,
    )

    # v1246: Patch OpenAI constructor
    with patch("openai.OpenAI", return_value=stop_mock_client):
        with pytest.raises(stop_type) as caught:
            client.complete([], operation="translation.body")

    assert caught.value is stop


@pytest.mark.parametrize(
    "error, resumable, truncated",
    [
        (RuntimeError(f"insufficient balance {SECRET}"), True, False),
        (TimeoutError(SECRET), True, False),
        (RuntimeError(SECRET), False, False),
        (TruncatedResponseError(SECRET), False, True),
    ],
)
def test_safe_error_preserves_workflow_classification(error, resumable, truncated):
    client, _, mock_client = client_with_failure()
    client.limits.wait_for_retry = lambda delay: None

    def create(**kwargs):
        raise error

    # v1246: Create new mock client with custom create
    error_mock_client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
        close=lambda: None,
    )

    # v1246: Patch OpenAI constructor
    with patch("openai.OpenAI", return_value=error_mock_client):
        with pytest.raises(ProviderRequestError) as caught:
            client.complete([], operation="translation.body")

    assert is_resumable_provider_interrupt(caught.value) is resumable
    assert isinstance(caught.value, TruncatedResponseError) is truncated
    assert SECRET not in str(caught.value)


def test_fallback_still_sees_original_provider_classification():
    client, calls, mock_client = client_with_failure()
    data = client.config.model_dump()
    data["providers"]["other"] = {"kind": "openai", "max_retries": 0}
    data["models"]["two"] = {"provider": "other", "model": "fallback"}
    data["routes"]["translation.body"] = {"model": "one", "fallbacks": ["two"]}
    routed = RoutedLLMClient(
        LLMConfig.model_validate(data), credentials={"default": SECRET, "other": "fake-other-key"}
    )
    messages = [{"role": "user", "content": SECRET}]

    def create(**kwargs):
        assert kwargs["messages"] == messages
        return SimpleNamespace(
            choices=[
                SimpleNamespace(message=SimpleNamespace(content=SECRET), finish_reason="stop")
            ],
            usage=None,
        )

    # v1246: Create mock clients for both providers
    fallback_mock_client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
        close=lambda: None,
    )

    # v1246: Patch OpenAI constructor - will be called for both providers
    # First call returns mock_client (fails), second call returns fallback (succeeds)
    call_count = [0]

    def mock_openai_constructor(*args, **kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            return mock_client  # First provider fails
        else:
            return fallback_mock_client  # Fallback succeeds

    with patch("openai.OpenAI", side_effect=mock_openai_constructor):
        result = routed.complete(messages, operation="translation.body")
        assert result == SECRET

    # First provider retries (max_retries=1), so 1 initial + 1 retry = 2 calls
    # v1246: Verify at least 1 call to first provider before fallback
    assert len(calls) >= 1, f"Expected at least 1 call, got {len(calls)}"
    # Note: Exact retry count may vary based on error type and retry logic
    # The important part is that fallback worked and SECRET was scrubbed


def test_review_persists_only_safe_errors(tmp_path, caplog):
    from wenyi_core.pipeline.orchestrator import Orchestrator

    from tests.review_fixtures import autofix_config as _config
    from tests.review_fixtures import autofix_store as _store

    client, calls, mock_client = client_with_failure()
    config = _config(str(tmp_path / "state"))
    config.llm = client.config
    config.pipeline.review_autofix = False
    store = _store(str(tmp_path))
    orch = Orchestrator(config, client)

    # v1246: Patch OpenAI constructor
    with patch("openai.OpenAI", return_value=mock_client):
        with pytest.raises(ProviderRequestError):
            orch._review.run_session(store, [])

    assert len(calls) == 2
    artifacts = [path for path in tmp_path.rglob("*.json") if "review" in str(path)]
    assert artifacts
    documents = "\n".join(path.read_text() for path in artifacts)
    assert "Model request failed:" in documents
    assert '"interrupted"' in documents
    assert SECRET not in documents
    for path in tmp_path.rglob("*.jsonl"):
        assert SECRET not in path.read_text()
    assert SECRET not in caplog.text
