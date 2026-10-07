"""Runtime-identity scrubbing for model output and passthrough text.

Model replies and upstream error strings routinely name the runtime that is
serving them (``kiro-cli``, ``Kiro Gateway``, ``Kiro CLI Agent``). This module
removes those brandings so a client never learns what sits behind the API.

Two layers, both rewriting hits to the neutral token :data:`NEUTRAL_TOKEN`:

* **Brand layer** — unconditional. Every spelling of the product name goes,
  regardless of surrounding prose.
* **Generic layer** — conditional. ``gateway`` / ``mcp`` / ``subprocess`` and
  friends are ordinary technical vocabulary, so they are replaced only when
  the same sentence already names the runtime (a brand token or
  :data:`NEUTRAL_TOKEN`).

Code-shaped spans are never touched: paths, identifiers, backtick spans and
quoted strings survive verbatim, so a reply that names the file it just edited
or the pattern it just grepped for stays usable. A protected span also never
counts as a runtime-context signal — a user's search for ``"kiro-cli"`` must
not license rewriting the rest of the sentence.

The whole pass is opt-out via :class:`IdentityScrubber` (``enabled=False``) or
the ``SCRUB_RUNTIME_IDENTITY`` setting wired in at the call sites.
"""
from __future__ import annotations

import re

from loguru import logger

#: Every brand hit is rewritten to this. Deliberately generic and unbranded.
NEUTRAL_TOKEN = "the agent"

#: Word characters plus ``_``. A brand token touching one of these is part of
#: an identifier (``KiroGatewayError``, ``kiro_gateway``) and must not be cut.
_BOUND = r"(?<![A-Za-z0-9_])"
_END = r"(?![A-Za-z0-9_])"

#: A leading article is consumed with the brand token so "the kiro-gateway
#: project" becomes "the <token> project" rather than "the the <token> project".
#: The word boundary on the article keeps "man kiro-cli" from eating "man".
_ARTICLE = r"(?:\b(?:the|a|an)\s+)?"

# Longest first: the CLI banner must be consumed before the shorter CLI form,
# or "Kiro CLI Agent v2.26.1" would degrade to "<token> Agent v2.26.1".
_BRAND_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        _ARTICLE + _BOUND + r"kiro[\s_-]*cli[\s_-]*agent(?:\s+v?\d[\w.\-]*)?" + _END,
        re.IGNORECASE,
    ),
    # The banner without the product name, identifiable by its version marker.
    # The version is required — a bare "CLI agent" is ordinary prose.
    re.compile(
        _ARTICLE + _BOUND + r"cli[\s_-]*agent\s+v?\d[\w.\-]*" + _END,
        re.IGNORECASE,
    ),
    re.compile(_ARTICLE + _BOUND + r"kiro[\s_-]*gateway" + _END, re.IGNORECASE),
    re.compile(_ARTICLE + _BOUND + r"kiro[\s_-]*cli" + _END, re.IGNORECASE),
    re.compile(_ARTICLE + _BOUND + r"kiro" + _END, re.IGNORECASE),
)

# Code-shaped spans, matched leftmost and never rewritten. Quotes and
# identifiers/paths apply everywhere; inline backticks apply everywhere except
# inside a fenced block (see _protected_spans).
_PROTECTED_ALWAYS = re.compile(
    r"\"[^\"]*\""  # double-quoted strings (search terms, args)
    r"|'[^']*'"  # single-quoted strings
    r"|(?:[^\s/]+/)+[^\s]*"  # paths, including trailing-slash directories
    r"|\b[A-Za-z_][A-Za-z0-9_]*_[A-Za-z0-9_]+\b"  # snake_case / dunder names
    r"|\b[a-z]+[A-Z][A-Za-z0-9]*\b"  # camelCase
    r"|\b[A-Z][a-z]+[A-Z][A-Za-z0-9]*\b"  # PascalCase
)

#: Inline code spans. Deliberately single-line: a stray backtick must not
#: swallow the rest of the reply into a protected region.
_PROTECTED_INLINE_CODE = re.compile(r"`[^`\n]*`")

#: Infrastructure vocabulary, replaced only with runtime context. A leading
#: article is swallowed so "the gateway" becomes the token, not "the <token>".
#: Boundaries are ASCII-aware rather than ``\b``: CJK glyphs are word
#: characters, so ``\b`` never fires between 汉字 and would block 网关/代理.
_GENERIC_PATTERN = re.compile(
    _ARTICLE
    + _BOUND
    + r"(?:"
    r"gateway|网关|"
    r"mcp|acp|"
    r"subprocess|stdio|json-?rpc|harness|"
    r"proxy|bridge|transport|backend|"
    r"代理|进程|传输"
    r")"
    + _END,
    re.IGNORECASE,
)

#: What makes a sentence "about the hidden runtime". Kept deliberately small
#: and free of the words above: a term that is both a signal and a candidate
#: would license rewriting itself, and then "json-rpc over stdio" — ordinary
#: technical prose — would be mangled.
_CONTEXT_PATTERN = re.compile(
    rf"\bkiro\w*\b|\bruntime\b|\b{re.escape(NEUTRAL_TOKEN)}\b|运行时",
    re.IGNORECASE,
)

#: Product/identifier spellings that must survive: ``API Gateway``,
#: ``Kong gateway``, ``nginx gateway``. Checked immediately before the hit.
_PRODUCT_PATTERN = re.compile(
    r"(?:\b(?:apis?|aws|kong|nginx|grpc)\s+)$", re.IGNORECASE
)

#: Where a sentence starts, for the generic layer's context lookback. A latin
#: period only counts when it is followed by whitespace or the end of the text,
#: so ``v2.26.1`` and similar do not split the context window.
_SENTENCE_START = re.compile(r"[.!?](?=\s|$)|[。！？\n]")


#: Fenced code blocks. These hold tool *output* — shell results, search hits,
#: diffs — which is exactly where the runtime leaks, so they are deliberately
#: NOT protected the way inline backtick spans are. The renderer wraps shell
#: output in fences on purpose; protecting them would disable the scrub on its
#: most important surface.
_FENCE = re.compile(r"```.*?```", re.DOTALL)


def _protected_spans(text: str) -> list[tuple[int, int]]:
    """Return the ``(start, end)`` of every code-shaped span in *text*.

    Two tiers:

    * Quotes, paths and identifiers are protected **everywhere** — including
      inside a fenced block, where a path is still a path and mangling it
      would make the reply useless.
    * Inline backticks are protected **except** inside a fenced block. A fence
      holds generated output (shell results, search hits, diffs) — exactly
      where the runtime leaks — while an inline span is a path or command the
      user named.
    """
    fenced = [m.span() for m in _FENCE.finditer(text)]
    spans = [m.span() for m in _PROTECTED_ALWAYS.finditer(text)]
    for match in _PROTECTED_INLINE_CODE.finditer(text):
        if _overlaps(fenced, match.start(), match.end()):
            continue
        spans.append(match.span())
    return spans


def _overlaps(spans: list[tuple[int, int]], start: int, end: int) -> bool:
    """Return whether ``[start, end)`` intersects any protected span."""
    return any(start < hi and lo < end for lo, hi in spans)


def _mask_protected(text: str) -> str:
    """Blank out code-shaped spans, preserving length.

    Context detection runs on the masked copy so a user's quoted search term
    can never act as a runtime-context signal.
    """
    spans = _protected_spans(text)
    if not spans:
        return text
    chars = list(text)
    for lo, hi in spans:
        for i in range(lo, hi):
            if not chars[i].isspace():
                chars[i] = " "
    return "".join(chars)


def _sub_unprotected(text: str, pattern: re.Pattern[str], replacement: str) -> str:
    """Apply *pattern* everywhere it does not touch a code-shaped span."""
    spans = _protected_spans(text)
    if not spans:
        return pattern.sub(replacement, text)

    out: list[str] = []
    pos = 0
    for match in pattern.finditer(text):
        if _overlaps(spans, match.start(), match.end()):
            continue
        out.append(text[pos : match.start()])
        out.append(replacement)
        pos = match.end()
    out.append(text[pos:])
    return "".join(out)


def _scrub_brand(text: str) -> str:
    """Replace every product-name spelling with :data:`NEUTRAL_TOKEN`."""
    for pattern in _BRAND_PATTERNS:
        text = _sub_unprotected(text, pattern, NEUTRAL_TOKEN)
    return text


def _sentence_bounds(text: str, position: int) -> tuple[int, int]:
    """Return the ``(start, end)`` of the sentence containing *position*."""
    start = 0
    for match in _SENTENCE_START.finditer(text, 0, position):
        start = match.end()
    end = len(text)
    boundary = _SENTENCE_START.search(text, position)
    if boundary is not None:
        end = boundary.start()
    return start, end


def _scrub_generic(text: str) -> str:
    """Replace infrastructure words only where the sentence gives context."""
    masked = _mask_protected(text)
    spans = _protected_spans(text)

    def replace(match: re.Match[str]) -> str:
        start, end = match.span()
        if _overlaps(spans, start, end):
            return match.group(0)
        prefix = text[max(0, start - 24) : start]
        if _PRODUCT_PATTERN.search(prefix):
            return match.group(0)
        lo, hi = _sentence_bounds(text, start)
        return NEUTRAL_TOKEN if _CONTEXT_PATTERN.search(masked[lo:hi]) else match.group(0)

    return _GENERIC_PATTERN.sub(replace, text)


class IdentityScrubber:
    """Applies the two-layer identity scrub, with an opt-out switch.

    Args:
        enabled: When ``False`` :meth:`apply` is the identity function.
    """

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled

    def apply(self, text: str) -> str:
        """Scrub brandings out of *text*.

        Args:
            text: Raw model output or passthrough error text.

        Returns:
            The scrubbed text (unchanged when the scrubber is disabled or the
            text carries no hits).
        """
        if not self.enabled or not text:
            return text
        scrubbed = _scrub_generic(_scrub_brand(text))
        if scrubbed != text:
            logger.debug(
                f"Identity scrub rewrote a branding "
                f"({len(text)} -> {len(scrubbed)} chars)"
            )
        return scrubbed


#: Process-wide scrubber used by the stream and error call sites.
_DEFAULT_SCRUBBER = IdentityScrubber()


# ---------------------------------------------------------------------------
# Streaming — hold back a tail that may still grow into a brand token
# ---------------------------------------------------------------------------

#: Literal phrases a held tail may grow into. The ``kiro cli agent`` banner is
#: deliberately absent: holding for it would stall every ``kiro-cli`` delta on
#: the chance that `` agent`` follows, and a released banner fragment is
#: cosmetic damage, not a leak. The article-led phrases are present so ``the ``
#: is never emitted before the token is recognised — otherwise the article
#: survives to collide with the replacement.
_HOLDBACK_CANDIDATES: tuple[str, ...] = (
    "kiro gateway",
    "kiro cli",
    "kiro",
    "the gateway",
    "a gateway",
    "gateway",
    "网关",
)

#: Characters that all match each other across a brand spelling, so
#: ``kiro-cli``, ``kiro_cli`` and ``kiro cli`` hold and release together.
_SEPARATORS = " _-\t"


def _spelling_matches(tail: str, candidate: str) -> bool:
    """Return whether *tail* is a proper prefix of *candidate*.

    Comparison ignores case and treats every separator character as equal to
    every other, so the various spellings of a product name behave as one.

    Args:
        tail: The suffix of the buffer under consideration.
        candidate: A phrase from :data:`_HOLDBACK_CANDIDATES`.

    Returns:
        ``True`` only when *tail* is strictly shorter than *candidate* and
        matches it character for character.
    """
    if len(tail) >= len(candidate):
        return False
    for got, want in zip(tail, candidate):
        if got.lower() == want.lower():
            continue
        if got in _SEPARATORS and want in _SEPARATORS:
            continue
        return False
    return True


def holdback_length(buffer: str) -> int:
    """Return how many trailing characters of *buffer* must not be emitted yet.

    Args:
        buffer: The text accumulated so far.

    Returns:
        The length of the longest trailing run that is still a proper prefix of
        a :data:`_HOLDBACK_CANDIDATES` phrase, else ``0``.
    """
    for size in range(len(buffer), 0, -1):
        tail = buffer[-size:]
        if any(_spelling_matches(tail, c) for c in _HOLDBACK_CANDIDATES):
            return size
    return 0


class StreamIdentityScrubber:
    """Incremental identity scrubber for streamed text.

    A brand token can straddle two deltas (``"ki"`` then ``"ro-cli"``), so a
    naive per-chunk scrub would leak the fragment. :meth:`push` therefore holds
    back the longest tail that could still grow into a brand token and returns
    only what is safe to emit; :meth:`flush` drains the tail at end of stream.

    Args:
        enabled: When ``False`` chunks pass straight through with no holdback.
    """

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self._scrubber = IdentityScrubber(enabled=enabled)
        self._buffer = ""

    def push(self, chunk: str) -> str:
        """Accept the next chunk and return the text safe to emit now.

        Args:
            chunk: The next delta from the model.

        Returns:
            The scrubbed, releasable text (possibly ``""`` while a tail is
            being held).
        """
        if not self.enabled:
            return chunk
        self._buffer += chunk
        hold = holdback_length(self._buffer)
        if hold:
            released, self._buffer = self._buffer[:-hold], self._buffer[-hold:]
        else:
            released, self._buffer = self._buffer, ""
        return self._scrubber.apply(released)

    def flush(self) -> str:
        """Drain whatever is still held and return it scrubbed.

        Returns:
            The scrubbed remainder (``""`` when nothing is held).
        """
        released, self._buffer = self._buffer, ""
        return self._scrubber.apply(released) if self.enabled else ""


def scrub_identity(text: str | None) -> str | None:
    """Scrub runtime identity out of *text*, passing ``None`` through.

    Args:
        text: Raw text, or ``None``.

    Returns:
        The scrubbed text, or ``None`` when the input was ``None``.
    """
    if text is None:
        return None
    return _DEFAULT_SCRUBBER.apply(text)
