"""Unit tests for kiro.identity_scrub (runtime-identity scrubbing on output).

Two layers, both replacing hits with the neutral token ``NEUTRAL_TOKEN``:

* **Brand layer** — unconditional: every spelling of the gateway/CLI product
  name goes, no matter the surrounding prose.
* **Generic layer** — conditional: bare ``gateway`` / ``网关`` / ``mcp`` is
  legitimate English/Chinese or ordinary technical vocabulary, so it is only
  replaced when the same sentence carries runtime context.

Code-shaped spans (paths, identifiers, backticks, quoted strings) are never
touched: a reply that names the file it just edited has to stay readable.
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

    def test_preceding_article_is_kept(self):
        # The replacement is a bare noun, so the source article stays put and
        # the phrase keeps its shape: "the kiro-gateway project" -> "the agent
        # project", never "the the agent project".
        assert (
            scrub_identity("the kiro-gateway project")
            == f"the {NEUTRAL_TOKEN} project"
        )

    def test_stream_split_cannot_double_the_article(self):
        # Regression: the holdback used to release "the " before the brand
        # token was recognised, so a chunk boundary there produced
        # "the the <token>".
        stream = StreamIdentityScrubber()
        first = stream.push("help you with the ")
        second = stream.push("kiro-gateway project?")
        assert first + second == f"help you with the {NEUTRAL_TOKEN} project?"

    def test_preceding_word_is_not_eaten(self):
        # "man kiro-cli" must not eat into "man".
        assert scrub_identity("man kiro-cli") == f"man {NEUTRAL_TOKEN}"

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
            == f"the request goes through the {NEUTRAL_TOKEN} to {NEUTRAL_TOKEN}"
        )

    def test_scrubs_gateway_when_sentence_names_the_runtime(self):
        # "runtime" is a genuine signal; "transport"/"stdio"/"gateway" alone
        # are not, so they are only rewritten once something unmistakable
        # appears in the sentence.
        assert (
            scrub_identity("the runtime is behind the gateway over stdio")
            == f"the runtime is behind the {NEUTRAL_TOKEN} over {NEUTRAL_TOKEN}"
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
        # The source article is kept, so the phrase still reads naturally.
        assert (
            scrub_identity("the CLI Agent v2.26.1 answered")
            == f"the {NEUTRAL_TOKEN} answered"
        )

    def test_generic_cli_agent_phrase_is_preserved(self):
        # No version marker, so this is ordinary prose about a CLI agent.
        assert scrub_identity("we shipped a CLI agent") == "we shipped a CLI agent"


# ---------------------------------------------------------------------------
# Code-shaped spans must survive — a reply has to stay usable
# ---------------------------------------------------------------------------

class TestProtectedSpans:
    """Paths, identifiers, backticks and quotes are never rewritten."""

    @pytest.mark.parametrize(
        "raw",
        [
            "Edit kiro/acp_client.py line 42",
            "Add a comment in kiro/config.py",
            "see ./kiro/gateway.md for details",
            "raised KiroGatewayError from the bridge",
            "set self.api_gateway = None",
            "run `grep -rn kiro kiro/`",
            'grep -rn "gateway" kiro/',
            "call mcp__feishu__send_message",
            "the KiroGatewayHandler class is gone",
        ],
    )
    def test_code_shaped_text_is_verbatim(self, raw: str):
        assert scrub_identity(raw) == raw

    def test_prose_still_gets_rewritten(self):
        assert (
            scrub_identity("I'm Kiro, behind the gateway")
            == f"I'm {NEUTRAL_TOKEN}, behind the {NEUTRAL_TOKEN}"
        )

    def test_a_protected_span_is_not_a_context_signal(self):
        # "kiro-cli" here is the user's search term. It must neither be
        # rewritten nor license rewriting "gateway" further along the sentence.
        text = 'grep -rn "kiro-cli" then deploy the payment gateway'
        assert scrub_identity(text) == text

    def test_prose_adjacent_to_protected_code_is_still_rewritten(self):
        text = "kiro-cli edited kiro/acp_client.py"
        assert scrub_identity(text) == f"{NEUTRAL_TOKEN} edited kiro/acp_client.py"

    def test_inline_backticks_protect_but_fences_do_not(self):
        # An inline span is a path or command the user named — keep it. A
        # fenced block is generated output (shell results, search hits), which
        # is exactly where the runtime leaks, so its prose gets scrubbed.
        text = "run `kiro-cli --help`:\n```\nkiro-cli 2.26.1 ready\n```"
        assert scrub_identity(text) == (
            "run `kiro-cli --help`:\n```\n"
            f"{NEUTRAL_TOKEN} 2.26.1 ready\n```"
        )

    def test_paths_in_shell_output_are_kept(self):
        text = "```\nkiro-cli edited kiro/acp_client.py\n```"
        assert scrub_identity(text) == (
            f"```\n{NEUTRAL_TOKEN} edited kiro/acp_client.py\n```"
        )


# ---------------------------------------------------------------------------
# Wider vocabulary — only with runtime context
# ---------------------------------------------------------------------------

class TestWiderTermList:
    """mcp/acp/subprocess & co. are ordinary words until context appears."""

    @pytest.mark.parametrize(
        "raw",
        [
            "pip install mcp",
            "the MCP server list is in config",
            "json-rpc over stdio to the subprocess",
            "use the proxy for outbound calls",
            "our test harness runs nightly",
        ],
    )
    def test_bare_infrastructure_words_survive_without_context(self, raw: str):
        assert scrub_identity(raw) == raw

    def test_contextual_mcp_and_transport_are_rewritten(self):
        assert scrub_identity(
            "kiro-cli registers the MCP servers over the transport"
        ) == f"{NEUTRAL_TOKEN} registers the {NEUTRAL_TOKEN} servers over the {NEUTRAL_TOKEN}"

    def test_contextual_subprocess_chain_is_rewritten(self):
        assert scrub_identity(
            "the gateway calls the subprocess, backed by kiro-cli"
        ) == f"the {NEUTRAL_TOKEN} calls the {NEUTRAL_TOKEN}, backed by {NEUTRAL_TOKEN}"

    def test_contextual_chinese_proxy_is_rewritten(self):
        assert scrub_identity(
            "请求经过网关与代理到达 kiro-cli"
        ) == f"请求经过{NEUTRAL_TOKEN}与{NEUTRAL_TOKEN}到达 {NEUTRAL_TOKEN}"


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
        assert first + second == f"via the {NEUTRAL_TOKEN} to {NEUTRAL_TOKEN}"

    def test_flush_returns_the_held_remainder(self):
        stream = StreamIdentityScrubber()
        stream.push("ends with ki")
        assert stream.flush() == "ki"

    def test_disabled_stream_passes_through_immediately(self):
        stream = StreamIdentityScrubber(enabled=False)
        assert stream.push("kiro-cli") == "kiro-cli"
        assert stream.flush() == ""


class TestStreamContextAcrossChunks:
    """Generic words must see runtime context that arrived in an earlier chunk.

    The generic layer only rewrites ``gateway`` / ``mcp`` when the sentence
    already names the runtime — but in a stream the brand token can land in one
    chunk and the generic word in the next, so a per-chunk scrub misses the
    context and leaks the word.
    """

    def test_generic_word_rewrites_with_brand_from_an_earlier_chunk(self):
        stream = StreamIdentityScrubber()
        out = stream.push("The session is served by kiro-cli through a ")
        out += stream.push("gateway on port 8001.")
        out += stream.flush()
        assert "kiro" not in out.lower()
        assert "gateway" not in out.lower()
        assert NEUTRAL_TOKEN in out

    def test_uncontextual_word_still_survives_across_chunks(self):
        # "gateway" with no runtime context anywhere is ordinary English —
        # chunking must not make it rewrite-eligible.
        stream = StreamIdentityScrubber()
        out = stream.push("She works at the payment ")
        out += stream.push("gateway downtown.")
        out += stream.flush()
        assert "gateway" in out


class TestProductDotDirectory:
    """The runtime's own hidden config dir is brand even inside code spans.

    ``~/.kiro/skills/`` is not somebody's project layout — it is the product's
    own footprint, so it is rewritten where ``kiro/acp_client.py`` (a user path)
    must stay readable.
    """

    def test_dot_dir_in_a_path_is_rewritten(self):
        assert (
            scrub_identity("A local skills directory at ~/.kiro/skills/")
            == f"A local skills directory at ~/.{NEUTRAL_TOKEN}/skills/"
        )

    def test_dot_dir_inside_backticks_is_rewritten(self):
        assert (
            scrub_identity("see `~/.kiro/settings/` for details")
            == f"see `~/.{NEUTRAL_TOKEN}/settings/` for details"
        )

    def test_user_project_path_still_survives(self):
        assert scrub_identity("Edit kiro/acp_client.py line 42") == (
            "Edit kiro/acp_client.py line 42"
        )

    def test_dot_dir_survives_a_chunk_split(self):
        stream = StreamIdentityScrubber()
        out = stream.push("skills live in ~/.ki")
        out += stream.push("ro/skills/ here")
        out += stream.flush()
        assert "kiro" not in out.lower()
        assert f"~/.{NEUTRAL_TOKEN}/skills/" in out
