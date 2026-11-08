"""OpenCode Go provider identity and session header tests."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from wenyi_core.config import Config
from wenyi_core.llm.configuration import ProviderConfig
from wenyi_core.llm.providers.opencode_go import USER_AGENT, OpenCodeGoClient
from wenyi_core.llm.registry import provider_spec


class TestOpenCodeGoProvider(unittest.TestCase):
    def test_provider_is_registered(self):
        self.assertEqual(provider_spec("opencode-go").kind, "opencode-go")
        self.assertEqual(provider_spec("opencode-go").client_class, "OpenCodeGoClient")

    def test_opencode_go_has_no_builtin_preset(self):
        with self.assertRaisesRegex(ValueError, "has no preset"):
            provider_spec("opencode-go").preset()

    def test_config_accepts_opencode_go_connection(self):
        cfg = Config.from_dict(
            {
                "llm": {
                    "providers": {"go": {"kind": "opencode-go"}},
                    "models": {
                        "writer": {
                            "provider": "go",
                            "model": "deepseek-v4-flash",
                        }
                    },
                    "tiers": {"strong": "writer", "cheap": "writer", "fast": "writer"},
                }
            }
        )
        self.assertEqual(cfg.llm.providers["go"].kind, "opencode-go")

    def test_ensure_client_passes_opencode_go_headers_only(self):
        client = OpenCodeGoClient(ProviderConfig(kind="opencode-go"))
        with (
            patch.dict("os.environ", {"OPENCODE_API_KEY": "test-key"}),
            patch("openai.OpenAI") as openai_ctor,
        ):
            client._ensure_client()
            client._ensure_client()  # reuse the same session id

        self.assertEqual(openai_ctor.call_count, 1)
        kwargs = openai_ctor.call_args.kwargs
        self.assertEqual(kwargs["base_url"], "https://opencode.ai/zen/go/v1")
        self.assertEqual(kwargs["api_key"], "test-key")
        headers = kwargs["default_headers"]
        self.assertEqual(headers["User-Agent"], USER_AGENT)
        self.assertEqual(headers["User-Agent"], "wenyi")
        self.assertRegex(
            headers["x-opencode-session"],
            r"^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$",
        )
        self.assertEqual(headers["x-opencode-session"], client._session_id)
        self.assertEqual(kwargs["max_retries"], 0)

    def test_isolated_complete_client_sends_the_same_session_headers(self):
        from types import SimpleNamespace

        import httpx
        from wenyi_core.llm.providers.opencode_go import OpenCodeGoOptions
        from wenyi_core.llm.transport import RequestContext, ResolvedModel

        client = OpenCodeGoClient(ProviderConfig(kind="opencode-go"))
        reply = SimpleNamespace(
            usage=None,
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(content="ok"),
                )
            ],
        )
        constructed: list[dict] = []

        class FakeOpenAI:
            def __init__(self, **kwargs):
                constructed.append(kwargs)
                self.chat = SimpleNamespace(
                    completions=SimpleNamespace(create=lambda **_kwargs: reply)
                )

            def close(self):
                return None

        with (
            patch.dict("os.environ", {"OPENCODE_API_KEY": "test-key"}),
            patch("openai.OpenAI", FakeOpenAI),
        ):
            text = client._request(
                [{"role": "user", "content": "hi"}],
                ResolvedModel(model="deepseek-v4-flash", options=OpenCodeGoOptions()),
                json_mode=False,
                context=RequestContext(
                    operation="translation.body",
                    tier="strong",
                    max_tokens=16,
                    emit=lambda *_args, **_kwargs: None,
                    record_usage=lambda *_args: None,
                    attempt_scope=lambda: None,
                ),
            )

        self.assertEqual(text, "ok")
        self.assertEqual(len(constructed), 1)
        kwargs = constructed[0]
        self.assertEqual(kwargs["base_url"], "https://opencode.ai/zen/go/v1")
        self.assertEqual(kwargs["api_key"], "test-key")
        self.assertEqual(kwargs["max_retries"], 0)
        self.assertIsInstance(kwargs["timeout"], httpx.Timeout)
        headers = kwargs["default_headers"]
        self.assertEqual(headers["User-Agent"], USER_AGENT)
        self.assertEqual(headers["x-opencode-session"], client._session_id)


if __name__ == "__main__":
    unittest.main()
