"""The proxy's own HTTP surface must not name the runtime or its stack.

A client that never asks the model anything can still read the docs pages,
the health payload, static error envelopes and the app metadata — and each of
those used to carry the product name, the protocol name, or the web framework
serving the replies. Every surface checked here has to be as unbranded as the
response bodies covered by the never-names tests in the shim suites.
"""
from __future__ import annotations

import re
from unittest.mock import MagicMock

import main
import pytest

#: Words that would tell a client what sits behind the API. Checked against
#: strings the *gateway* authors (app metadata, health, static errors), never
#: against model prose — there the scrubber allows ordinary technical words.
_BRANDED = re.compile(
    r"kiro|gateway|fastapi|uvicorn|\bacp\b|\bcli\b",
    re.IGNORECASE,
)


def _assert_unbranded(text: str) -> None:
    hit = _BRANDED.search(text or "")
    assert hit is None, f"branded token {hit.group(0)!r} in {text!r}"


class TestDocsSurfaceIsGone:
    """Swagger/ReDoc/OpenAPI are the loudest leak: they print the app title,
    the route docstrings and the schema names for anyone who asks."""

    @pytest.mark.parametrize("path", ["/docs", "/redoc", "/openapi.json"])
    def test_docs_endpoint_is_not_served(self, sync_client, path):
        resp = sync_client.get(path)
        assert resp.status_code == 404
        _assert_unbranded(resp.text)

    def test_app_metadata_is_unbranded(self):
        _assert_unbranded(main.app.title)
        _assert_unbranded(main.app.description or "")


class TestHealthIsUnbranded:
    """``/health`` is public; its payload used to say ``acp-cli-bridge``."""

    def test_health_payload_is_unbranded(self, sync_client):
        resp = sync_client.get("/health")
        assert resp.status_code == 200
        _assert_unbranded(resp.text)

    def test_health_still_reports_status_and_mode(self, sync_client):
        body = sync_client.get("/health").json()
        assert body["status"] == "ok"
        assert "mode" in body
        assert "version" in body


class TestStaticErrorsAreUnbranded:
    """Error envelopes authored by the gateway must not describe its path."""

    def test_embeddings_error_is_unbranded(self, sync_client, openai_headers):
        resp = sync_client.post(
            "/v1/embeddings",
            headers=openai_headers,
            json={"model": "text-embedding-3-small", "input": "hi"},
        )
        assert resp.status_code == 501
        _assert_unbranded(resp.text)
        # The OpenAI-native envelope stays so client back-off still works.
        assert resp.json()["error"]["code"] == "embeddings_not_supported"


class TestServerHeaderIsGone:
    """``Server: uvicorn`` names the web stack on every response."""

    def test_uvicorn_server_header_is_disabled(self):
        kwargs = main._uvicorn_kwargs("127.0.0.1", 8000)
        assert kwargs["server_header"] is False


class TestModelNotFoundErrorIsUnbranded:
    """``ModelNotAvailableError`` text named the product ("this gateway")."""

    def test_model_not_found_is_unbranded(self, monkeypatch, sync_client, openai_headers):
        monkeypatch.setattr("kiro.routes_openai_shim.settings.MODEL_VALIDATION", "strict")
        # Validation needs a non-empty catalogue to reject against; an empty
        # one is treated as "not discovered yet" and lets everything through.
        sync_client.app.state.shim_service.available_models = lambda: [
            {"id": "claude-sonnet-4.6"}
        ]
        resp = sync_client.post(
            "/v1/chat/completions",
            headers=openai_headers,
            json={"model": "no-such-model", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert resp.status_code == 404
        _assert_unbranded(resp.text)


class TestSessionCwdDoesNotReadTheGatewayTree:
    """The model must not land in the gateway's own source tree.

    ``kiro-cli`` loads ``AGENTS.md`` / ``CLAUDE.md`` from the session cwd and
    feeds them to the model as project context — which is how the reply came to
    recite the FastAPI/uvicorn/session-new architecture verbatim. The fallback
    cwd must honour ``ACP_WORKSPACE_DIR`` instead of the process cwd, so a
    harness that sends no workspace hint does not get the gateway's own docs.
    """

    def test_cwd_falls_back_to_acp_workspace_dir(self, monkeypatch, tmp_path):
        from kiro.acp_client import ACPClient, settings as _acp_settings

        monkeypatch.setattr(_acp_settings, "ACP_WORKSPACE_DIR", str(tmp_path))
        assert ACPClient._derive_cwd(None) == str(tmp_path)

    def test_explicit_filesystem_root_still_wins(self, monkeypatch, tmp_path):
        from kiro.acp_client import ACPClient, settings as _acp_settings

        workspace = tmp_path / "workspace"
        workspace.mkdir()
        root = tmp_path / "harness"
        root.mkdir()
        monkeypatch.setattr(_acp_settings, "ACP_WORKSPACE_DIR", str(workspace))
        caps = MagicMock(filesystem=[{"path": str(root)}])
        assert ACPClient._derive_cwd(caps) == str(root)
