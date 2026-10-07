"""
Deliver a harness system prompt through kiro-cli's own system-prompt channel.

ACP has no system role: ``session/prompt`` carries role-less content blocks, so
the gateway used to serialise a harness system prompt into the user turn as a
``System:`` label. Models treat that as user-supplied text. Verified against a
live kiro-cli 2.28.0 probe (claude-opus-5.5) with a Claude-Desktop-style system
prompt: asked "who are you", the model answered as Kiro in 5 of 6 turns when
the prompt arrived as a ``System:`` label, and as Claude in 6 of 6 turns when
it arrived as a custom agent's ``prompt``.

kiro-cli's documented channel is a custom agent: an agent config's ``prompt``
field is added to kiro-cli's system prompt. Live probe facts this module relies
on:

* kiro-cli rescans ``~/.kiro/agents`` on every ``session/new``, so an agent
  written while the process runs is selectable via ``session/set_mode``.
* The file can be deleted right after ``session/set_mode`` succeeds; the
  session keeps the prompt. Each request uses its own uniquely named file, and
  concurrent sessions stay isolated.
* An agent with ``tools: ["*"]`` and ``includeMcpJson: true`` matches
  kiro_default: same context size, workspace ``AGENTS.md`` still loaded,
  ``session/new`` ``mcpServers`` still honoured.

kiro-cli's own built-in system prompt is always kept. This changes only where
the harness prompt lands, not how many tokens it costs.
"""
from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

from loguru import logger

# File/agent name prefix for every ephemeral agent this gateway writes. Used to
# recognise (and never advertise or leak) them. Deliberately unbranded: the
# agent's name and description sit in kiro-cli's agent directory and may be
# echoed into the system prompt it assembles, so neither may name the product.
EPHEMERAL_AGENT_PREFIX = "sysctx-"

# Ephemeral files older than this are crash leftovers. Another gateway instance
# sharing the directory only keeps its file for the duration of one
# session/new + session/set_mode, so this age never touches a live one.
STALE_AGENT_SECONDS = 600

# Appended to every harness prompt installed through the agent channel. It
# keeps the model from describing the machinery behind the reply, and is
# phrased without any product name so the guard cannot itself be the leak.
#
# The second paragraph exists because the first one alone does not win: the
# harness system prompt, tool descriptions and project docs all push toward a
# full capability tour on a bare "hi".
IDENTITY_GUARD = (
    "Identity & scope: you are the assistant for this coding session, in the "
    "persona defined above. Do not describe your runtime, transport, tooling "
    "backend, or how your replies reach the user. If asked what you are or "
    "what you run on, answer in terms of that persona and continue with the "
    "user's actual task.\n"
    "\n"
    "Greetings and identity questions need no preamble. A greeting gets one "
    "line; \"who are you\" gets one sentence naming the persona. Never open "
    "with a list of what you can do, a tour of the project, or how requests "
    "reach you — offer those only once the user states a task."
)

# Neutral description for the ephemeral agent config.
EPHEMERAL_AGENT_DESCRIPTION = "Session context carrier"

_LEADING_SYSTEM_ROLES = ("system", "developer")


def default_agents_dir() -> Path:
    """Return kiro-cli's global agent directory (``~/.kiro/agents``).

    Returns:
        The directory path (may not exist yet).
    """
    return Path.home() / ".kiro" / "agents"


def _role_and_content(message: Any) -> tuple[str | None, Any]:
    """Read ``role``/``content`` from a ``PromptMessage`` or a plain dict."""
    if isinstance(message, dict):
        return message.get("role"), message.get("content")
    return getattr(message, "role", None), getattr(message, "content", None)


def split_leading_system(messages: list[Any]) -> tuple[str, list[Any]]:
    """Split the leading system/developer messages off a conversation.

    Only the contiguous run before the first ``user``/``assistant`` turn is a
    system prompt; a system message later in the conversation (e.g. a
    mid-session reminder) stays in the transcript, in order. A message whose
    content is not a plain string (e.g. image-bearing) ends the run so nothing
    non-text is moved.

    Args:
        messages: ``PromptMessage`` objects or ``{"role", "content"}`` dicts.

    Returns:
        ``(system_text, remaining_messages)``; ``system_text`` is the leading
        texts joined by blank lines (``""`` when there are none).
    """
    texts: list[str] = []
    index = 0
    for index, message in enumerate(messages):
        role, content = _role_and_content(message)
        if role not in _LEADING_SYSTEM_ROLES or not isinstance(content, str):
            break
        if content.strip():
            texts.append(content.strip())
    else:
        index = len(messages)
    return "\n\n".join(texts), list(messages[index:])


def build_agent_config(name: str, prompt: str) -> dict[str, Any]:
    """Build an ephemeral agent config equivalent to kiro_default plus *prompt*.

    The returned ``prompt`` is the harness system prompt with
    :data:`IDENTITY_GUARD` appended, so the model is told not to describe its
    runtime even when the harness prompt says nothing about it.

    Args:
        name: The agent (and mode) id.
        prompt: The harness system prompt.

    Returns:
        A kiro-cli agent config dict.
    """
    body = prompt.rstrip()
    combined = f"{body}\n\n{IDENTITY_GUARD}" if body else IDENTITY_GUARD
    return {
        "name": name,
        "description": EPHEMERAL_AGENT_DESCRIPTION,
        "prompt": combined,
        "tools": ["*"],
        "includeMcpJson": True,
    }


def write_agent(prompt: str, agents_dir: Path | None = None) -> tuple[str, Path]:
    """Atomically write a uniquely named ephemeral agent file.

    The file is written under a non-``.json`` temporary name and renamed into
    place, so a concurrent kiro-cli rescan never reads a partial config.

    Args:
        prompt: The harness system prompt.
        agents_dir: Target directory (defaults to :func:`default_agents_dir`).

    Returns:
        ``(agent_name, file_path)``.

    Raises:
        OSError: If the directory or file cannot be written.
    """
    directory = agents_dir or default_agents_dir()
    directory.mkdir(parents=True, exist_ok=True)
    name = f"{EPHEMERAL_AGENT_PREFIX}{uuid.uuid4().hex[:16]}"
    path = directory / f"{name}.json"
    tmp = directory / f".{name}.tmp"
    tmp.write_text(json.dumps(build_agent_config(name, prompt)), encoding="utf-8")
    os.replace(tmp, path)
    return name, path


def remove_agent(path: Path) -> None:
    """Delete an ephemeral agent file, logging (never raising) on failure.

    Args:
        path: The file written by :func:`write_agent`.
    """
    try:
        path.unlink(missing_ok=True)
    except OSError as exc:
        logger.warning(f"Could not remove ephemeral agent file {path}: {exc}")


def cleanup_stale_agents(
    agents_dir: Path | None = None, max_age: float = STALE_AGENT_SECONDS
) -> int:
    """Remove ephemeral agent files left behind by a crashed gateway.

    Args:
        agents_dir: Directory to scan (defaults to :func:`default_agents_dir`).
        max_age: Minimum age in seconds before a file counts as stale.

    Returns:
        The number of files removed.
    """
    directory = agents_dir or default_agents_dir()
    if not directory.is_dir():
        return 0
    cutoff = time.time() - max_age
    removed = 0
    for candidate in directory.iterdir():
        name = candidate.name.lstrip(".")
        if not name.startswith(EPHEMERAL_AGENT_PREFIX):
            continue
        try:
            if candidate.stat().st_mtime < cutoff:
                candidate.unlink()
                removed += 1
        except OSError as exc:
            logger.debug(f"Skipping stale agent cleanup for {candidate}: {exc}")
    if removed:
        logger.info(f"Removed {removed} stale ephemeral agent file(s) from {directory}")
    return removed


def is_ephemeral_agent(mode_id: str) -> bool:
    """Return ``True`` for a mode id created by :func:`write_agent`."""
    return str(mode_id).startswith(EPHEMERAL_AGENT_PREFIX)
