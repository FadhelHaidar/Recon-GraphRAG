"""Tests for token counting and packing utilities."""

from __future__ import annotations

import pytest

from recon_graphrag.config.settings import BudgetConfig, PipelineConfig
from recon_graphrag.utils.tokens import (
    ApproximateTokenCounter,
    PackItem,
    TokenCounter,
    TokenPacker,
    count_tokens,
    create_token_counter,
    truncate_text,
)


class ConstantTokenCounter(TokenCounter):
    """Test counter that counts one token per character."""

    def count(self, text: str) -> int:
        return len(text)

    def truncate(self, text: str, max_tokens: int) -> str:
        return text[:max_tokens]


class TestApproximateTokenCounter:
    def test_empty_text_returns_zero(self):
        counter = ApproximateTokenCounter()
        assert counter.count("") == 0

    def test_ascii_text_estimate(self):
        counter = ApproximateTokenCounter(ratio=4.0)
        # 8 chars / 4 = 2 tokens
        assert counter.count("abcdefgh") == 2

    def test_unicode_text_counts_characters(self):
        counter = ApproximateTokenCounter(ratio=2.0)
        # Each emoji is one character
        assert counter.count("🙂🙂🙂🙂") == 2

    def test_custom_ratio(self):
        counter = ApproximateTokenCounter(ratio=2.0)
        assert counter.count("abcd") == 2

    def test_invalid_ratio_raises(self):
        with pytest.raises(ValueError):
            ApproximateTokenCounter(ratio=0)

    def test_truncate_empty_text(self):
        counter = ApproximateTokenCounter()
        assert counter.truncate("", 10) == ""

    def test_truncate_zero_tokens(self):
        counter = ApproximateTokenCounter()
        assert counter.truncate("hello", 0) == ""

    def test_truncate_negative_raises(self):
        counter = ApproximateTokenCounter()
        with pytest.raises(ValueError):
            counter.truncate("hello", -1)

    def test_truncate_respects_ratio(self):
        counter = ApproximateTokenCounter(ratio=4.0)
        # 8 chars max for 2 tokens
        assert counter.truncate("abcdefghij", 2) == "abcdefgh"

    def test_multiline_text(self):
        counter = ApproximateTokenCounter(ratio=4.0)
        text = "line one\nline two\nline three"
        assert counter.count(text) == math_ceil(len(text) / 4.0)


def math_ceil(value: float) -> int:
    import math

    return math.ceil(value)


class TestCreateTokenCounter:
    def test_create_approximate(self):
        counter = create_token_counter("approximate")
        assert isinstance(counter, ApproximateTokenCounter)

    def test_create_approximate_with_ratio(self):
        counter = create_token_counter("approximate", ratio=3.0)
        assert counter.count("abc") == 1

    def test_create_unknown_raises(self):
        with pytest.raises(ValueError):
            create_token_counter("unknown")

    def test_create_tiktoken_without_dependency_raises(self):
        # tiktoken is not a mandatory dependency; this test documents the error.
        with pytest.raises(ImportError):
            create_token_counter("tiktoken")


class TestTokenPacker:
    def test_empty_items(self):
        packer = TokenPacker()
        result = packer.pack([], max_tokens=10)
        assert result.included == []
        assert result.excluded == []
        assert result.used_tokens == 0

    def test_all_items_fit(self):
        counter = ConstantTokenCounter()
        packer = TokenPacker(counter)
        items = [PackItem(id="a", text="ab"), PackItem(id="b", text="cd")]
        result = packer.pack(items, max_tokens=4)
        assert [i.id for i in result.included] == ["a", "b"]
        assert result.excluded == []
        assert result.used_tokens == 4

    def test_stops_when_budget_exhausted(self):
        counter = ConstantTokenCounter()
        packer = TokenPacker(counter, truncate=False)
        items = [
            PackItem(id="a", text="ab"),
            PackItem(id="b", text="cd"),
            PackItem(id="c", text="ef"),
        ]
        result = packer.pack(items, max_tokens=3)
        assert [i.id for i in result.included] == ["a"]
        assert [i.id for i in result.excluded] == ["b", "c"]
        assert result.used_tokens == 2

    def test_truncates_remaining_item_when_enabled(self):
        counter = ConstantTokenCounter()
        packer = TokenPacker(counter, truncate=True)
        items = [
            PackItem(id="a", text="ab"),
            PackItem(id="b", text="cd"),
        ]
        result = packer.pack(items, max_tokens=3)
        assert [i.id for i in result.included] == ["a", "b"]
        assert result.included[1].text == "c"
        assert result.truncated_item_ids == ["b"]
        assert result.used_tokens == 3

    def test_truncates_last_fitting_item(self):
        counter = ConstantTokenCounter()
        packer = TokenPacker(counter, truncate=True)
        items = [PackItem(id="a", text="ab"), PackItem(id="b", text="cdef")]
        result = packer.pack(items, max_tokens=3)
        assert [i.id for i in result.included] == ["a", "b"]
        assert result.included[1].text == "c"
        assert result.truncated_item_ids == ["b"]
        assert result.used_tokens == 3

    def test_no_truncate_policy_excludes_item(self):
        counter = ConstantTokenCounter()
        packer = TokenPacker(counter, truncate=False)
        items = [PackItem(id="a", text="ab"), PackItem(id="b", text="cdef")]
        result = packer.pack(items, max_tokens=3)
        assert [i.id for i in result.included] == ["a"]
        assert [i.id for i in result.excluded] == ["b"]
        assert result.truncated_item_ids == []

    def test_single_oversized_item_truncates(self):
        counter = ConstantTokenCounter()
        packer = TokenPacker(counter, truncate=True)
        items = [PackItem(id="big", text="abcdefghij")]
        result = packer.pack(items, max_tokens=4)
        assert len(result.included) == 1
        assert result.included[0].text == "abcd"
        assert result.truncated_item_ids == ["big"]

    def test_single_oversized_item_excluded_when_no_truncate(self):
        counter = ConstantTokenCounter()
        packer = TokenPacker(counter, truncate=False)
        items = [PackItem(id="big", text="abcdefghij")]
        result = packer.pack(items, max_tokens=4)
        assert result.included == []
        assert [i.id for i in result.excluded] == ["big"]

    def test_max_tokens_must_be_positive(self):
        packer = TokenPacker()
        with pytest.raises(ValueError):
            packer.pack([], max_tokens=0)
        with pytest.raises(ValueError):
            packer.pack([], max_tokens=-1)


class TestConvenienceFunctions:
    def test_count_tokens_default_counter(self):
        assert count_tokens("abcdefgh") == 2  # default ratio 4

    def test_truncate_text_default_counter(self):
        assert truncate_text("abcdefghij", 2) == "abcdefgh"


class TestBudgetConfig:
    def test_default_budgets_are_none(self):
        cfg = BudgetConfig()
        assert cfg.extraction_chunk_tokens is None
        assert cfg.global_reduce_input_tokens is None

    def test_positive_budgets_allowed(self):
        cfg = BudgetConfig(extraction_chunk_tokens=100)
        assert cfg.extraction_chunk_tokens == 100

    def test_zero_budget_raises(self):
        with pytest.raises(ValueError):
            BudgetConfig(community_input_tokens=0)

    def test_negative_budget_raises(self):
        with pytest.raises(ValueError):
            BudgetConfig(global_map_input_tokens=-10)

    def test_non_integer_budget_raises(self):
        with pytest.raises(ValueError):
            BudgetConfig(global_reduce_output_tokens="large")


class TestPipelineConfig:
    def test_default_config(self):
        cfg = PipelineConfig()
        assert cfg.chunk_size == 1000
        assert cfg.chunk_overlap == 200
        assert cfg.budget is None
        assert cfg.token_counter is None

    def test_config_with_budget(self):
        budget = BudgetConfig(extraction_chunk_tokens=512)
        cfg = PipelineConfig(budget=budget)
        assert cfg.budget is budget

    def test_invalid_chunk_size_raises(self):
        with pytest.raises(ValueError):
            PipelineConfig(chunk_size=0)

    def test_invalid_overlap_raises(self):
        with pytest.raises(ValueError):
            PipelineConfig(chunk_overlap=1000)

    def test_negative_overlap_raises(self):
        with pytest.raises(ValueError):
            PipelineConfig(chunk_overlap=-1)
