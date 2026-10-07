"""Unit tests for kiro.system_prompt_agent (harness system prompt → agent prompt)."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from kiro.acp_models import PromptMessage
from kiro.config import _parse_system_prompt_channel
from kiro.system_prompt_agent import (
    EPHEMERAL_AGENT_PREFIX,
    build_agent_config,
    cleanup_stale_agents,
    is_ephemeral_agent,
    remove_agent,
    split_leading_system,
    write_agent,
)


class TestSplitLeadingSystemSuccess:
    """Only the leading system/developer run is a system prompt."""

    def test_leading_system_and_developer_joined_in_order(self):
        messages = [
            PromptMessage(role="system", content="You are Claude."),
            PromptMessage(role="developer", content="sandbox: read-only"),
            PromptMessage(role="user", content="hi"),
        ]

        system, rest = split_leading_system(messages)

        assert system == "You are Claude.\n\nsandbox: read-only"
        assert [(m.role, m.content) for m in rest] == [("user", "hi")]

    def test_mid_conversation_system_stays_in_transcript(self):
        messages = [
            PromptMessage(role="system", content="base"),
            PromptMessage(role="user", content="q1"),
            PromptMessage(role="system", content="reminder"),
            PromptMessage(role="user", content="q2"),
        ]

        system, rest = split_leading_system(messages)

        assert system == "base"
        assert [m.content for m in rest] == ["q1", "reminder", "q2"]

    def test_accepts_plain_dicts(self):
        system, rest = split_leading_system([
            {"role": "system", "content": "s"}, {"role": "user", "content": "u"},
        ])

        assert system == "s"
        assert rest == [{"role": "user", "content": "u"}]

    def test_no_system_returns_messages_unchanged(self):
        messages = [PromptMessage(role="user", content="u"),
                    PromptMessage(role="system", content="late")]

        system, rest = split_leading_system(messages)

        assert system == ""
        assert rest == messages


class TestSplitLeadingSystemEdgeCases:
    """Unusual shapes never move non-text content or lose turns."""

    def test_block_list_system_ends_the_run(self):
        image_system = PromptMessage(
            role="system", content=[{"type": "image", "mimeType": "image/png", "data": "x"}]
        )
        messages = [PromptMessage(role="system", content="text"), image_system,
                    PromptMessage(role="user", content="u")]

        system, rest = split_leading_system(messages)

        assert system == "text"
        assert rest[0] is image_system

    def test_blank_system_contributes_nothing(self):
        system, rest = split_leading_system([
            PromptMessage(role="system", content="   "),
            PromptMessage(role="user", content="u"),
        ])

        assert system == ""
        assert len(rest) == 1

    def test_all_system_leaves_no_turn(self):
        system, rest = split_leading_system([PromptMessage(role="system", content="only")])

        assert system == "only"
        assert rest == []

    def test_empty_input(self):
        assert split_leading_system([]) == ("", [])


class TestAgentFiles:
    """Ephemeral agent files are written atomically and cleaned up."""

    def test_write_agent_mirrors_kiro_default(self, tmp_path: Path):
        name, path = write_agent("You are Claude.", tmp_path / "agents")

        config = json.loads(path.read_text())
        assert path.parent == tmp_path / "agents"
        assert path.name == f"{name}.json"
        assert is_ephemeral_agent(name)
        assert config == build_agent_config(name, "You are Claude.")
        assert config["tools"] == ["*"] and config["includeMcpJson"] is True
        # The temporary file was renamed into place, not left behind.
        assert sorted(p.name for p in path.parent.iterdir()) == [path.name]

    def test_write_agent_names_are_unique(self, tmp_path: Path):
        first, _ = write_agent("a", tmp_path)
        second, _ = write_agent("a", tmp_path)

        assert first != second

    def test_write_agent_raises_oserror_when_unwritable(self, tmp_path: Path):
        blocker = tmp_path / "file"
        blocker.write_text("x")

        with pytest.raises(OSError):
            write_agent("p", blocker / "agents")

    def test_remove_agent_is_idempotent(self, tmp_path: Path):
        _, path = write_agent("p", tmp_path)

        remove_agent(path)
        remove_agent(path)

        assert not path.exists()

    def test_remove_agent_logs_instead_of_raising(self, tmp_path: Path, monkeypatch):
        _, path = write_agent("p", tmp_path)

        def boom(self, missing_ok=False):
            raise PermissionError("denied")

        monkeypatch.setattr(Path, "unlink", boom)
        remove_agent(path)  # must not raise

    def test_cleanup_removes_only_stale_ephemeral_files(self, tmp_path: Path):
        _, stale = write_agent("old", tmp_path)
        _, fresh = write_agent("new", tmp_path)
        stale_tmp = tmp_path / f".{EPHEMERAL_AGENT_PREFIX}dead.tmp"
        stale_tmp.write_text("{}")
        user_agent = tmp_path / "code.json"
        user_agent.write_text("{}")
        old = time.time() - 3600
        for path in (stale, stale_tmp, user_agent):
            os.utime(path, (old, old))

        removed = cleanup_stale_agents(tmp_path, max_age=600)

        assert removed == 2
        assert not stale.exists() and not stale_tmp.exists()
        assert fresh.exists() and user_agent.exists()

    def test_cleanup_missing_dir_is_noop(self, tmp_path: Path):
        assert cleanup_stale_agents(tmp_path / "missing") == 0

    def test_client_default_dir_is_isolated_in_tests(self, _isolated_agents_dir):
        from kiro.acp_client import ACPClient

        assert ACPClient()._agents_dir == _isolated_agents_dir

    @pytest.mark.parametrize("mode_id, expected", [
        (f"{EPHEMERAL_AGENT_PREFIX}abc", True),
        ("kiro_default", False),
        ("code", False),
    ])
    def test_is_ephemeral_agent(self, mode_id, expected):
        assert is_ephemeral_agent(mode_id) is expected


class TestSystemPromptChannelConfig:
    """KIRO_SYSTEM_PROMPT parsing: agent by default, inline on request."""

    @pytest.mark.parametrize("raw", ["", "agent", "AGENT", "true", "on", "nonsense"])
    def test_agent_channel(self, raw):
        assert _parse_system_prompt_channel(raw) == "agent"

    @pytest.mark.parametrize("raw", ["inline", "label", "off", "false", "0", "no", " Inline "])
    def test_inline_channel(self, raw):
        assert _parse_system_prompt_channel(raw) == "inline"


class TestIdentityHygiene:
    """The ephemeral agent must not name the runtime to kiro-cli's model.

    The agent's ``name`` and ``description`` sit in ``~/.kiro/agents`` and may
    well be echoed into the system prompt kiro-cli assembles, so neither may
    carry the product name. The prompt also gains an identity guard telling the
    model not to describe its runtime — phrased without brand words so the
    guard cannot itself be the leak.
    """

    def test_prefix_carries_no_brand_words(self):
        lowered = EPHEMERAL_AGENT_PREFIX.lower()
        assert "kiro" not in lowered
        assert "gateway" not in lowered

    def test_agent_config_carries_no_brand_words(self):
        config = build_agent_config("any-name", "You are Claude.")
        blob = json.dumps(config).lower()
        assert "kiro" not in blob
        assert "gateway" not in blob
        assert "网关" not in blob

    def test_prompt_keeps_the_harness_text_and_adds_a_guard(self):
        config = build_agent_config("any-name", "You are Claude.")
        prompt = config["prompt"]
        assert prompt.startswith("You are Claude.")
        lowered = prompt.lower()
        assert "runtime" in lowered
        assert "transport" in lowered

    def test_guard_alone_carries_no_brand_words(self):
        guard = build_agent_config("any-name", "")["prompt"]
        lowered = guard.lower()
        assert "kiro" not in lowered
        assert "gateway" not in lowered

    def test_written_agent_matches_the_config(self, tmp_path: Path):
        name, path = write_agent("You are Claude.", tmp_path)
        config = json.loads(path.read_text(encoding="utf-8"))
        assert config == build_agent_config(name, "You are Claude.")
        assert "kiro" not in name.lower()
        assert "gateway" not in name.lower()
