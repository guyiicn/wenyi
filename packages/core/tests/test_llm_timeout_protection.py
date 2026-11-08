"""Test application-layer timeout protection against stalled/dead HTTPS connections."""

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


def test_application_timeout_fires_on_stalled_connection():
    """Application timeout should fire when SDK call hangs beyond configured timeout.
    
    Regression test for stalled synopsis.chapter calls: ESTABLISHED HTTPS connection
    with no response body should become a retryable timeout, not hang indefinitely.
    """
    # Use very short timeout for test speed
    cfg = ProviderConfig(kind="test-provider", base_url="https://example.com", timeout=2.0)
    client = FakeClient(cfg)
    
    # Mock _ensure_client to return a mock SDK client
    mock_sdk_client = Mock()
    mock_completions = Mock()
    mock_sdk_client.chat.completions = mock_completions
    
    # Simulate stalled connection: create() hangs for longer than timeout
    def slow_create(**kwargs):
        time.sleep(5.0)  # Hang longer than timeout
        raise AssertionError("Should not reach here - timeout should fire first")
    
    mock_completions.create = slow_create
    
    with patch.object(client, '_ensure_client', return_value=mock_sdk_client):
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


def test_application_timeout_allows_slow_but_progressing_requests():
    """Application timeout should NOT fire for slow requests that complete within timeout.
    
    Ensures we don't kill legitimate long-running requests that are making progress.
    """
    cfg = ProviderConfig(kind="test-provider", base_url="https://example.com", timeout=3.0)
    client = FakeClient(cfg)
    
    # Mock _ensure_client to return a mock SDK client
    mock_sdk_client = Mock()
    mock_completions = Mock()
    mock_sdk_client.chat.completions = mock_completions
    
    # Simulate slow but successful request
    mock_response = Mock()
    mock_response.choices = [Mock(message=Mock(content="test response"), finish_reason="stop")]
    mock_response.usage = None
    
    def slow_create(**kwargs):
        time.sleep(1.5)  # Slow but within timeout
        return mock_response
    
    mock_completions.create = slow_create
    
    with patch.object(client, '_ensure_client', return_value=mock_sdk_client):
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


def test_timeout_error_is_retryable():
    """TimeoutError from application timeout should be classified as retryable.
    
    Ensures the timeout becomes a retryable error that triggers llm_retry_wait.
    """
    from wenyi_core.llm.retrying import retry_reason
    
    error = TimeoutError("LLM request exceeded application timeout")
    reason = retry_reason(error)
    
    assert reason == "timeout", f"Expected 'timeout', got {reason!r}"


def test_httpx_timeout_configuration():
    """Verify httpx.Timeout is properly configured with all timeout components."""
    cfg = ProviderConfig(kind="test-provider", base_url="https://example.com", timeout=900.0)
    client = FakeClient(cfg)
    
    # Trigger client creation with patched OpenAI (imported inside _ensure_client)
    with patch('openai.OpenAI') as mock_openai:
        mock_instance = Mock()
        mock_openai.return_value = mock_instance
        
        # Ensure client creation
        client._ensure_client()
        
        # Verify OpenAI was called with proper timeout config
        assert mock_openai.called
        call_kwargs = mock_openai.call_args[1]
        
        # Check timeout is httpx.Timeout object, not just float
        timeout_arg = call_kwargs.get('timeout')
        assert timeout_arg is not None
        
        # Verify it's an httpx.Timeout object with proper read timeout
        import httpx
        assert isinstance(timeout_arg, httpx.Timeout)
        assert timeout_arg.read == 900.0, f"Expected read timeout 900.0, got {timeout_arg.read}"
