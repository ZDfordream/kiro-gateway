"""Unit tests for kiro.identity_scrub (runtime-identity scrubbing on output).

Two layers, both replacing hits with the neutral token ``the tool``:

* **Brand layer** — unconditional: every spelling of the gateway/CLI product
  name goes, no matter the surrounding prose.
* **Generic layer** — conditional: bare ``gateway`` / ``网关`` is a legitimate
  English/Chinese word (API Gateway, Kong, ``api_gateway``), so it is only
  replaced when the same sentence carries runtime context.
"""
from __future__ import annotations

import pytest

from kiro.identity_scrub import (
    NEUTRAL_TOKEN,
    IdentityScrubber,
    StreamIdentityScrubber,
    scrub_identity,
)


# ---------------------------------------------------------------------------
# Brand layer — unconditional
# ---------------------------------------------------------------------------

class TestBrandLayer:
    """Product-name spellings are always replaced."""

    @pytest.mark.parametrize(
        "raw",
        [
            "backed by kiro-cli",
            "backed by Kiro CLI",
            "backed by KIRO-CLI",
            "backed by Kiro-CLI",
        ],
    )
    def test_cli_spellings_become_neutral(self, raw: str):
        assert scrub_identity(raw) == f"backed by {NEUTRAL_TOKEN}"

    @pytest.mark.parametrize(
        "raw",
        [
            "runs in kiro-gateway",
            "runs in kiro gateway",
            "runs in Kiro Gateway",
            "runs in KIRO-GATEWAY",
        ],
    )
    def test_gateway_product_spellings_become_neutral(self, raw: str):
        assert scrub_identity(raw) == f"runs in {NEUTRAL_TOKEN}"

    def test_agent_banner_becomes_neutral(self):
        assert (
            scrub_identity("Kiro CLI Agent v2.26.1 here")
            == f"{NEUTRAL_TOKEN} here"
        )

    def test_standalone_product_name_becomes_neutral(self):
        assert scrub_identity("Hi, I am Kiro.") == f"Hi, I am {NEUTRAL_TOKEN}."

    def test_adjacent_punctuation_is_kept(self):
        assert scrub_identity("(kiro-cli)") == f"({NEUTRAL_TOKEN})"

    def test_internal_identifiers_are_not_split(self):
        # ``KiroGatewayError`` has no word boundary after "Kiro"; it is source
        # code the user may be discussing and must survive verbatim.
        assert scrub_identity("raise KiroGatewayError") == "raise KiroGatewayError"


# ---------------------------------------------------------------------------
# Generic layer — conditional on same-sentence runtime context
# ---------------------------------------------------------------------------

class TestGenericLayer:
    """Bare gateway/网关 is scrubbed only with runtime context nearby."""

    def test_scrubs_gateway_when_sentence_has_brand_context(self):
        assert (
            scrub_identity("the request goes through the gateway to kiro-cli")
            == f"the request goes through {NEUTRAL_TOKEN} to {NEUTRAL_TOKEN}"
        )

    def test_scrubs_gateway_when_sentence_has_runtime_context(self):
        assert (
            scrub_identity("the transport is a gateway over stdio")
            == f"the transport is {NEUTRAL_TOKEN} over stdio"
        )

    def test_scrubs_chinese_gateway_with_context(self):
        assert (
            scrub_identity("请求经过网关到达 kiro-cli")
            == f"请求经过{NEUTRAL_TOKEN}到达 {NEUTRAL_TOKEN}"
        )

    def test_leaves_bare_gateway_without_context(self):
        assert (
            scrub_identity("deploy the payment gateway next sprint")
            == "deploy the payment gateway next sprint"
        )

    def test_leaves_bare_chinese_gateway_without_context(self):
        assert scrub_identity("支付网关下周上线") == "支付网关下周上线"


# ---------------------------------------------------------------------------
# False positives that must survive
# ---------------------------------------------------------------------------

class TestFalsePositives:
    """Named products and code identifiers stay verbatim."""

    @pytest.mark.parametrize(
        "raw",
        [
            "configure AWS API Gateway",
            "the Kong gateway routes /api",
            "nginx gateway mode",
            "self.api_gateway = None",
            "APIGatewayHandler",
        ],
    )
    def test_product_and_identifier_names_are_preserved(self, raw: str):
        assert scrub_identity(raw) == raw

    def test_context_in_a_different_sentence_does_not_leak_over(self):
        text = "kiro-cli is running.\nNow configure the payment gateway."
        assert scrub_identity(text) == (
            f"{NEUTRAL_TOKEN} is running.\n"
            "Now configure the payment gateway."
        )

    def test_version_numbers_do_not_break_the_context_window(self):
        # "v2.26.1" contains dots; treating them as sentence boundaries would
        # cut the runtime context off from the word it is meant to license.
        text = "backed by kiro-cli v2.26.1, requests go through the gateway"
        assert "gateway" not in scrub_identity(text)

    def test_agent_banner_without_the_product_name_is_scrubbed(self):
        assert (
            scrub_identity("the CLI Agent v2.26.1 answered")
            == f"{NEUTRAL_TOKEN} answered"
        )

    def test_generic_cli_agent_phrase_is_preserved(self):
        # No version marker, so this is ordinary prose about a CLI agent.
        assert scrub_identity("we shipped a CLI agent") == "we shipped a CLI agent"


# ---------------------------------------------------------------------------
# Edges
# ---------------------------------------------------------------------------

class TestEdges:
    """Null/empty handling and the opt-out switch."""

    def test_none_passes_through(self):
        assert scrub_identity(None) is None

    def test_empty_string_passes_through(self):
        assert scrub_identity("") == ""

    def test_text_without_hits_is_unchanged(self):
        assert scrub_identity("hello world") == "hello world"

    def test_disabled_scrubber_is_identity(self):
        scrubber = IdentityScrubber(enabled=False)
        assert scrubber.apply("runs in kiro-gateway") == "runs in kiro-gateway"


# ---------------------------------------------------------------------------
# Streaming — brand tokens split across deltas
# ---------------------------------------------------------------------------

class TestStreamIdentityScrubber:
    """Chunks that end mid-brand-token must not leak the fragment."""

    def test_holds_a_partial_brand_token(self):
        stream = StreamIdentityScrubber()
        assert stream.push("hi ki") == "hi "
        assert stream.push("ro-cli") == f"{NEUTRAL_TOKEN}"

    def test_releases_text_that_cannot_be_a_brand_prefix(self):
        stream = StreamIdentityScrubber()
        assert stream.push("hello world") == "hello world"

    def test_holds_through_a_long_product_name(self):
        stream = StreamIdentityScrubber()
        assert stream.push("runs in kiro-gate") == "runs in "
        assert stream.push("way now") == f"{NEUTRAL_TOKEN} now"

    def test_holds_a_partial_generic_token(self):
        stream = StreamIdentityScrubber()
        first = stream.push("via the gate")
        second = stream.push("way to kiro-cli")
        assert first + second == f"via {NEUTRAL_TOKEN} to {NEUTRAL_TOKEN}"

    def test_flush_returns_the_held_remainder(self):
        stream = StreamIdentityScrubber()
        stream.push("ends with ki")
        assert stream.flush() == "ki"

    def test_disabled_stream_passes_through_immediately(self):
        stream = StreamIdentityScrubber(enabled=False)
        assert stream.push("kiro-cli") == "kiro-cli"
        assert stream.flush() == ""
