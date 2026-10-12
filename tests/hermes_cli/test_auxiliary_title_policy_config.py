"""Title-only policy registration and real config-loader/caller contracts."""
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest

from agent import auxiliary_client as ac
from agent import title_generator
from agent.auxiliary_routing_policy import strict_provider_policy
from agent.auxiliary_task_config import _get_auxiliary_task_config
from hermes_cli import config as config_mod
from hermes_cli.config_defaults import DEFAULT_CONFIG
from hermes_constants import reset_hermes_home_override, set_hermes_home_override

TASK = "title_generation"
ENDPOINT = "https://title.invalid/v1"
MODEL = "title-model-fixture"
KEY_ENV = "OFFLINE_TITLE_ROUTE_KEY"
KEY = "synthetic-title-key"
ROUTE = {"provider_policy": "strict", "provider": "custom", "model": MODEL,
         "base_url": ENDPOINT, "key_env": KEY_ENV, "api_mode": "chat_completions"}


@pytest.fixture
def title_wire(monkeypatch, tmp_path):
    """Real loaders, title caller, OpenAI SDK and HTTPX; in-memory transport only."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    token = set_hermes_home_override(str(tmp_path))
    monkeypatch.setattr("hermes_cli.plugins.get_plugin_auxiliary_tasks", list)
    monkeypatch.setattr(ac, "_client_cache", {})
    forbidden = {}
    for name in ("_peek_pool_entry", "_pool_cache_hint", "_recoverable_pool_provider",
                 "_refresh_provider_credentials", "_recover_provider_pool",
                 "_nous_portal_account_has_fresh_paid_access", "_read_main_api_key_if_same_origin",
                 "_resolve_api_key_provider", "_try_configured_fallback_chain",
                 "_try_main_fallback_chain", "_try_payment_fallback",
                 "_try_configured_fallback_for_unavailable_client", "_try_main_agent_model_fallback",
                 "resolve_provider_client"):
        forbidden[name] = Mock(side_effect=AssertionError("forbidden: " + name))
        monkeypatch.setattr(ac, name, forbidden[name])
    key_lookup = Mock(side_effect=lambda name: KEY if name == KEY_ENV else "")
    monkeypatch.setattr(ac, "_scoped_key_env", key_lookup)
    named_key_lookup = Mock(side_effect=lambda name, default="": KEY if name == KEY_ENV else default)
    monkeypatch.setattr("hermes_cli.runtime_provider_custom.get_secret_str", named_key_lookup)
    state = SimpleNamespace(path=tmp_path / "config.yaml", status=200, requests=[],
                            key_lookup=key_lookup, named_key_lookup=named_key_lookup)

    def respond(transport, request):
        state.requests.append(request)
        if state.status == 307:
            return httpx.Response(307, headers={"location": "https://other.invalid/chat/completions"})
        if state.status != 200:
            return httpx.Response(state.status, json={"error": {"message": "synthetic payment error"}})
        return httpx.Response(200, json={"id": "fixture", "object": "chat.completion", "created": 0,
            "model": MODEL, "choices": [{"index": 0, "message": {"role": "assistant",
            "content": '{"title":"Offline title integration"}'}, "finish_reason": "stop"}]})

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", respond)
    try:
        yield state
        for mock in forbidden.values():
            mock.assert_not_called()
    finally:
        for client, _, _ in ac._client_cache.values():
            client.close()
        reset_hermes_home_override(token)


def write_title_config(wire, route, **root):
    # JSON is YAML; fixture files only, never the live configuration.
    wire.path.write_text(json.dumps({**root, "auxiliary": {TASK: route}}), encoding="utf-8")


def test_title_policy_registry_preserves_legacy_default():
    title = DEFAULT_CONFIG["auxiliary"]["title_generation"]
    assert title["provider_policy"] == "legacy"
    assert not strict_provider_policy(title)
    assert title["provider"] == "auto"
    assert not title["model"]


@pytest.mark.parametrize("route", [{}, {"provider_policy": "legacy"}])
def test_existing_title_config_stays_legacy_without_rewrite(title_wire, route):
    write_title_config(title_wire, route)
    before = title_wire.path.read_bytes()
    loaded = _get_auxiliary_task_config(TASK)
    assert loaded["provider_policy"] == "legacy"
    assert not strict_provider_policy(loaded)
    assert title_wire.path.read_bytes() == before
    assert not strict_provider_policy(_get_auxiliary_task_config("memory_query_rewrite"))


def test_field_only_strict_on_auto_declines_title_before_client(title_wire, monkeypatch):
    write_title_config(title_wire, {"provider_policy": "strict"})
    constructor = Mock(side_effect=AssertionError("no client for incomplete route"))
    monkeypatch.setattr(ac, "_create_openai_client", constructor)
    assert _get_auxiliary_task_config(TASK)["provider"] == "auto"
    assert title_generator.generate_title("Offline title integration") is None
    assert title_generator.derive_title("Offline title integration") == "Offline title integration"
    constructor.assert_not_called()
    title_wire.key_lookup.assert_not_called()
    title_wire.named_key_lookup.assert_not_called()
    assert not title_wire.requests


@pytest.mark.parametrize("named", [False, True], ids=["inline", "named"])
@pytest.mark.parametrize("status", [200, 307, 402])
def test_complete_strict_route_from_loader_to_title_wire(title_wire, named, status):
    route = dict(ROUTE)
    providers = {}
    if named:
        route["provider"] = "custom:offline-title"
        providers["offline-title"] = {key: route.pop(key) for key in ("base_url", "key_env", "api_mode")}
    write_title_config(title_wire, route, providers=providers)
    title_wire.status = status
    result = title_generator.generate_title("Verify offline integration routing")
    assert result == ("Offline title integration" if status == 200 else None)
    assert len(title_wire.requests) == 1
    request = title_wire.requests[0]
    assert str(request.url) == ENDPOINT + "/chat/completions"
    assert json.loads(request.content)["model"] == MODEL
    assert request.headers["authorization"] == "Bearer " + KEY
    if named:
        assert title_wire.named_key_lookup.call_count >= 1
        for call in title_wire.named_key_lookup.call_args_list:
            assert call.args[0] == KEY_ENV
        title_wire.key_lookup.assert_not_called()
    else:
        title_wire.key_lookup.assert_called_once_with(KEY_ENV)
        title_wire.named_key_lookup.assert_not_called()
    for client, _, _ in ac._client_cache.values():
        assert client._client.follow_redirects is False
        assert client._client._trust_env is False
        assert not client._client._mounts
        assert client.max_retries == 0
    assert not strict_provider_policy(_get_auxiliary_task_config("memory_query_rewrite"))


@pytest.mark.parametrize("loss", ["delete", "empty", "malformed-task", "cold-yaml", "warm-yaml"])
def test_real_loader_policy_loss_is_not_a_persistent_deny_rule(title_wire, loss):
    if loss != "cold-yaml":
        write_title_config(title_wire, dict(ROUTE))
        assert strict_provider_policy(_get_auxiliary_task_config(TASK))
    if loss == "delete":
        title_wire.path.unlink()
    elif loss == "empty":
        title_wire.path.write_text("{}", encoding="utf-8")
    elif loss == "malformed-task":
        write_title_config(title_wire, "invalid-task")
    else:
        title_wire.path.write_text("auxiliary: [", encoding="utf-8")
    assert strict_provider_policy(_get_auxiliary_task_config(TASK)) is (loss == "warm-yaml")
    if loss.endswith("yaml"):
        assert isinstance(config_mod.load_config_readonly(), config_mod.FailedConfigRead)
