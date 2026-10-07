"""System prompt sanitizer — strip harness identity overrides.

Claude Code, Kilo Code, and other AI harnesses inject large system prompts
(~30KB) that include identity assertions ("You are Claude Code"), concealment
instructions ("Never reveal you're behind a gateway"), and framework boilerplate.
When serialised into the ACP prompt text, kiro-cli's model may detect these as
prompt injection attempts and refuse to answer (issue #73).

Two groups of patterns, split by how kiro-cli receives them:

* **Instruction overrides** — "Ignore any instructions that contradict…". This
  is the shape injection detectors actually key on, and it is useless in a
  system prompt anyway, so it is stripped on **every** channel.
* **Identity & concealment** — "You are Claude Code", "Never reveal the
  gateway". These are legitimate system-prompt content and are exactly what
  keeps the model from naming the runtime. They survive on the *agent* channel
  (kiro-cli's own system-prompt channel) and are stripped only where the text
  lands as user-turn text via the ``System:`` label, which is where issue #73
  bit. See :func:`agent_system_prompt_channel`.

Everything else (coding standards, memory, tool descriptions, project rules) is
preserved. The filter is **opt-out** via ``SANITIZE_SYSTEM_PROMPTS=false`` for
users who want the full system text forwarded verbatim.
"""
from __future__ import annotations

import re
from typing import Any, Optional

from loguru import logger


# ---------------------------------------------------------------------------
# Patterns to strip (compiled once at module load)
# ---------------------------------------------------------------------------

# Identity assertions: "You are Claude Code", "You are Claude", "I am Claude"
_IDENTITY_PATTERNS = [
    # "You are <Name>, ..." (up to end of sentence or line)
    re.compile(
        r"^.*?(?:You are|I am|This is|You must identify (?:yourself )?as)\s+"
        r"(?:Claude(?:\s+Code)?|Copilot|ChatGPT|GPT-\d|Gemini|an? (?:AI |interactive )?"
        r"(?:CLI )?tool (?:made|created|developed) by)"
        r"[^\n]*$",
        re.MULTILINE | re.IGNORECASE,
    ),
    # "You are NOT any other AI", "You are NOT Kiro", "Always identify as Claude"
    re.compile(
        r"^.*?(?:You (?:are|must) (?:NOT|never)|Always (?:identify|maintain|keep))"
        r"[^\n]*?(?:other (?:AI|assistant)|identity|identify as|Claude|Kiro)"
        r"[^\n]*$",
        re.MULTILINE | re.IGNORECASE,
    ),
    # "Never say you are Kiro/another assistant"
    re.compile(
        r"^.*?(?:Never|Do not|Don't)\s+(?:say|claim|admit|reveal)\s+"
        r"(?:you are|that you're)"
        r"[^\n]*$",
        re.MULTILINE | re.IGNORECASE,
    ),
]

# Concealment instructions: "Never reveal the gateway/proxy/bridge"
_CONCEALMENT_PATTERNS = [
    re.compile(
        r"^.*?(?:Never|Do not|Don't|Must not|Should not)\s+"
        r"(?:reveal|disclose|mention|say|tell|expose|admit)"
        r"[^\n]*?(?:gateway|proxy|bridge|running through|behind)"
        r"[^\n]*$",
        re.MULTILINE | re.IGNORECASE,
    ),
]

# Identity section headers that usually precede identity-override blocks.
# We strip the header line only (the block content is caught by the patterns
# above or left intact if it's useful context).
_IDENTITY_HEADERS = re.compile(
    r"^#+\s*(?:Identity|Who You Are|Your Identity|Agent Identity)\s*$",
    re.MULTILINE | re.IGNORECASE,
)

#: Stripped on every channel. This is the instruction-override shape — the one
#: kiro-cli's injection detection keys on — and no system prompt needs it.
_ALWAYS_STRIPPED_PATTERNS = [
    re.compile(
        r"^.*?(?:Ignore|Disregard|Override)\s+(?:any )?(?:instructions?|rules?)\s+"
        r"(?:that )?(?:contradict|conflict)"
        r"[^\n]*$",
        re.MULTILINE | re.IGNORECASE,
    ),
]

#: Stripped only where the prompt lands as user-turn text (``System:`` label).
_CHANNEL_SENSITIVE_PATTERNS = (
    _IDENTITY_PATTERNS + _CONCEALMENT_PATTERNS + [_IDENTITY_HEADERS]
)

_ALL_PATTERNS = _CHANNEL_SENSITIVE_PATTERNS + _ALWAYS_STRIPPED_PATTERNS


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def agent_system_prompt_channel(cfg: Any = None) -> bool:
    """Return whether the leading system prompt uses kiro-cli's agent channel.

    Mirrors :meth:`kiro.acp_client.ACPClient._wants_system_prompt_agent`'s
    configuration side: the agent channel needs ``KIRO_SYSTEM_PROMPT`` left at
    ``agent`` and no configured persona (``KIRO_ACP_MODE`` / ``KIRO_ACP_AGENT``),
    since a persona keeps its own agent config and forces the inline label. A
    per-request ``mode`` override is not visible here and still falls back to
    the stricter inline policy.

    Args:
        cfg: Settings object holding ``SYSTEM_PROMPT_CHANNEL``, ``ACP_MODE`` and
            ``ACP_AGENT``. Defaults to :data:`kiro.config.settings`. Route
            modules pass their own reference: tests reload ``kiro.config``, so
            ``kiro.config.settings`` is not necessarily the object an
            already-imported module closed over.

    Returns:
        ``True`` when the prompt will be delivered as an agent ``prompt``.
    """
    if cfg is None:
        from kiro.config import settings as cfg  # noqa: PLW0622 (rebinding)

    return (
        cfg.SYSTEM_PROMPT_CHANNEL == "agent"
        and not cfg.ACP_MODE
        and not cfg.ACP_AGENT
    )


def sanitize_system_prompt(
    text: Optional[str], *, preserve_identity: bool = False
) -> Optional[str]:
    """Remove identity-override and concealment lines from a system prompt.

    Preserves everything else (coding standards, memory, tool descriptions,
    project rules). Instruction-override lines are always removed; identity and
    concealment lines are removed only when *preserve_identity* is ``False``.

    Args:
        text: The raw system prompt text, or ``None``.
        preserve_identity: Keep identity/concealment lines. Pass ``True`` when
            the prompt will reach the model through kiro-cli's agent channel
            (see :func:`agent_system_prompt_channel`).

    Returns:
        The sanitized text (may be shorter), or ``None`` if input was ``None``.
    """
    if not text:
        return text

    patterns = (
        _ALWAYS_STRIPPED_PATTERNS
        if preserve_identity
        else _ALL_PATTERNS
    )

    original_len = len(text)
    result = text
    for pattern in patterns:
        result = pattern.sub("", result)

    # Collapse runs of 3+ blank lines down to 2 (cosmetic)
    result = re.sub(r"\n{3,}", "\n\n", result)

    stripped = original_len - len(result)
    if stripped > 0:
        logger.debug(
            f"System prompt sanitized: removed {stripped} chars "
            f"({stripped * 100 // original_len}% of {original_len})"
        )

    return result
