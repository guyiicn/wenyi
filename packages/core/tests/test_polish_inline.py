"""Test Polish/Review inline marker preservation with text length and order changes."""

from __future__ import annotations

import json

from wenyi_core.agents.polisher import Polisher
from wenyi_core.agents.translator import Translator
from wenyi_core.llm.providers.fake import FakeClient
from wenyi_core.pipeline.translate.translation_batch import TranslationBatchExecutor

from .pipeline_fixtures import batch_plan, fake_pipeline_config


def test_polish_with_length_change_preserves_inline_markers():
    """Polish rewrites Chinese text with significant length changes, inline ranges updated."""
    config = fake_pipeline_config(state_dir="/tmp/test")

    def handler(messages, tier, json_mode):
        # Only Polish call
        return json.dumps(
            {
                "polished": [
                    "这是一个完整[[I_START]]示例性[[I_END]]的句子描述内容。",
                    "测试的[[STRONG_START]]段落[[STRONG_END]]内容。",
                ]
            }
        )

    client = FakeClient(handler=handler, config=config.llm)
    translator = Translator(client, config)
    polisher = Polisher(client, config)
    executor = TranslationBatchExecutor(translator, polisher)

    plan = batch_plan(["This is an example sentence.", "Another test paragraph."])

    # Manually inject clean targets and inline_metas (simulating translation result)
    executor._translator.last_batch_turn = [{"role": "user", "content": "test"}]
    executor._translator.last_batch_indices = [0, 1]
    executor._translator.last_inline_metas = [
        {
            "version": 1,
            "target_length": len("这是一个示例句子。"),
            "ranges": [{"start": 4, "end": 6, "tag": "i"}],
        },
        {
            "version": 1,
            "target_length": len("另一个测试段落。"),
            "ranges": [{"start": 3, "end": 5, "tag": "strong"}],
        },
    ]

    # Mock translator.translate_batch to return clean text
    def mock_translate(*args, **kwargs):
        translator.last_inline_metas = executor._translator.last_inline_metas
        translator.last_batch_turn = executor._translator.last_batch_turn
        translator.last_batch_indices = executor._translator.last_batch_indices
        return ["这是一个示例句子。", "另一个测试段落。"]

    translator.translate_batch = mock_translate

    # Execute with polish=True
    result = executor.execute(plan, polish=True)

    assert len(result.targets) == 2
    assert len(result.inline_metas) == 2

    # First target should have clean text
    target0 = result.targets[0]
    meta0 = result.inline_metas[0]

    assert "[[I_START]]" not in target0
    assert "[[I_END]]" not in target0
    assert "示例性" in target0

    assert meta0["version"] == 1
    assert meta0["target_length"] == len(target0)
    ranges0 = meta0["ranges"]
    assert len(ranges0) == 1
    assert ranges0[0]["tag"] == "i"

    # Extract emphasized text using ranges
    r = ranges0[0]
    emphasized_text = target0[r["start"] : r["end"]]
    assert emphasized_text == "示例性"

    # Second target
    target1 = result.targets[1]
    meta1 = result.inline_metas[1]

    assert "[[STRONG_START]]" not in target1
    assert "[[STRONG_END]]" not in target1
    assert "段落" in target1

    assert meta1["version"] == 1
    ranges1 = meta1["ranges"]
    assert len(ranges1) == 1
    assert ranges1[0]["tag"] == "strong"

    r = ranges1[0]
    emphasized_text = target1[r["start"] : r["end"]]
    assert emphasized_text == "段落"


def test_polish_without_inline_passes_through():
    """Polish without inline markers works normally."""
    config = fake_pipeline_config(state_dir="/tmp/test")

    def handler(messages, tier, json_mode):
        return json.dumps({"polished": ["精炼的句子。", "优化的句子。"]})

    client = FakeClient(handler=handler, config=config.llm)
    translator = Translator(client, config)
    polisher = Polisher(client, config)
    executor = TranslationBatchExecutor(translator, polisher)

    plan = batch_plan(["Normal sentence.", "Another sentence."])

    # Mock translator to return clean text with no inline_metas
    def mock_translate(*args, **kwargs):
        translator.last_inline_metas = [{}, {}]
        translator.last_batch_turn = [{"role": "user", "content": "test"}]
        translator.last_batch_indices = [0, 1]
        return ["普通句子。", "另一个句子。"]

    translator.translate_batch = mock_translate

    result = executor.execute(plan, polish=True)

    assert len(result.targets) == 2
    assert result.targets[0] == "精炼的句子。"
    assert result.targets[1] == "优化的句子。"

    # No inline metas or empty
    assert all(not meta or not meta.get("ranges") for meta in result.inline_metas)


def test_polish_with_corrupted_markers_preserves_clean_text():
    """When polish returns corrupted markers, decode warnings raised but text cleaned."""
    config = fake_pipeline_config(state_dir="/tmp/test")

    def handler(messages, tier, json_mode):
        # Polish returns with missing END marker
        return json.dumps({"polished": ["这是[[I_START]]示例的句子。"]})

    client = FakeClient(handler=handler, config=config.llm)
    translator = Translator(client, config)
    polisher = Polisher(client, config)
    executor = TranslationBatchExecutor(translator, polisher)

    plan = batch_plan(["This is an example."])

    # Mock translator
    def mock_translate(*args, **kwargs):
        translator.last_inline_metas = [
            {
                "version": 1,
                "target_length": 8,
                "ranges": [{"start": 2, "end": 4, "tag": "i"}],
            }
        ]
        translator.last_batch_turn = [{"role": "user", "content": "test"}]
        translator.last_batch_indices = [0]
        return ["这是示例。"]

    translator.translate_batch = mock_translate

    result = executor.execute(plan, polish=True)

    # Should still produce clean text (markers removed)
    assert "[[I_START]]" not in result.targets[0]
    # Inline meta should be empty or warning recorded (decode will clear corrupted ranges)
    assert not result.inline_metas[0].get("ranges")


def test_polish_render_not_polluted_across_batches():
    """Polisher.render is restored after inline batch, doesn't affect non-inline batches."""
    config = fake_pipeline_config(state_dir="/tmp/test")

    call_count = 0
    prompts_seen = []

    def handler(messages, tier, json_mode):
        nonlocal call_count
        call_count += 1
        # Record user prompts to check for pollution
        for msg in messages:
            if msg["role"] == "user":
                prompts_seen.append(msg["content"])
        return json.dumps({"polished": ["润色结果。"]})

    client = FakeClient(handler=handler, config=config.llm)
    translator = Translator(client, config)
    polisher = Polisher(client, config)
    executor = TranslationBatchExecutor(translator, polisher)

    # Batch 1: with inline
    def mock_translate_inline(*args, **kwargs):
        translator.last_inline_metas = [
            {"version": 1, "target_length": 4, "ranges": [{"start": 0, "end": 2, "tag": "i"}]}
        ]
        translator.last_batch_turn = [{"role": "user", "content": "test"}]
        translator.last_batch_indices = [0]
        return ["示例。"]

    translator.translate_batch = mock_translate_inline
    plan1 = batch_plan(["Example."])
    executor.execute(plan1, polish=True)

    inline_prompt = prompts_seen[-1]
    assert "CRITICAL" in inline_prompt or "protected inline markers" in inline_prompt

    # Batch 2: without inline
    prompts_seen.clear()

    def mock_translate_plain(*args, **kwargs):
        translator.last_inline_metas = [{}]
        translator.last_batch_turn = [{"role": "user", "content": "test"}]
        translator.last_batch_indices = [0]
        return ["普通文本。"]

    translator.translate_batch = mock_translate_plain
    plan2 = batch_plan(["Normal text."])
    executor.execute(plan2, polish=True)

    plain_prompt = prompts_seen[-1]
    # Should NOT have inline preservation instructions
    assert "CRITICAL" not in plain_prompt and "protected inline markers" not in plain_prompt
