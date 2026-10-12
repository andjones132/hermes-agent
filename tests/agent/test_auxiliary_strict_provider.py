"""Offline spend-boundary regressions for explicit auxiliary routing.

The test doubles represent two different destinations; neither is contacted.
Legacy tests record the intentional availability-first behavior, not a spend
approval. Strict tests require route failure rather than implicit cloud use.
"""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from agent import auxiliary_client as ac

TASK = "title_generation"
LOCAL = "http://dgx.invalid/v1"
REMOTE = "https://cloud.invalid/v1"
MODEL = "dgx/glm-5.3-flash"
MESSAGES = [{"role": "user", "content": "Name this synthetic test conversation"}]


def response(text="offline title"):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
        content=text, tool_calls=None))])


def failure(kind):
    if kind == "connection":
        return ConnectionError("Connection error")
    if kind == "timeout":
        return TimeoutError("Request timed out")
    status, text = {
        "quota": (429, "Weekly usage limit reached: usage_limit_reached"),
        "rate": (429, "Rate limit exceeded"),
        "payment": (402, "Insufficient credits"),
        "auth": (401, "Invalid API key"),
        "model": (400, "Model is not supported with this route"),
        "parameter": (400, "Unsupported parameter: temperature"),
    }[kind]
    exc = RuntimeError(text)
    exc.status_code = status
    return exc


def client(base, async_mode=False):
    create = AsyncMock(return_value=response()) if async_mode else Mock(return_value=response())
    return SimpleNamespace(base_url=base, api_key="test-only-not-a-secret",
                           chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


@pytest.fixture
def routing(monkeypatch):
    """Real resolution/ladder, mocked clients, credentials and fallback edges."""
    cfg = {"provider": "custom", "model": MODEL, "base_url": LOCAL,
           "api_key": "test-only-not-a-secret", "api_mode": "chat_completions"}
    monkeypatch.setattr(ac, "_get_auxiliary_task_config", lambda task: cfg if task == TASK else {})
    local, remote = client(LOCAL), client(REMOTE)
    factory = Mock(return_value=local)
    monkeypatch.setattr(ac, "_create_openai_client", factory)
    monkeypatch.setattr("openai.AsyncOpenAI", factory)
    monkeypatch.setattr("httpx.Client", Mock())
    monkeypatch.setattr("httpx.AsyncClient", Mock())
    monkeypatch.setattr(ac, "_to_async_client", lambda c, m, **kw: (c, m))
    # This test is about Hermes resolution and recovery, not the Relay wire adapter.
    monkeypatch.setattr(ac, "_relay_sync_completion", lambda c, kw, **opts: c.chat.completions.create(**kw))

    async def async_wire(c, kw, **opts):
        return await c.chat.completions.create(**kw)

    monkeypatch.setattr(ac, "_relay_async_completion", async_wire)
    mocks = {}
    none_routes = (
        "_try_configured_fallback_chain", "_try_main_fallback_chain",
        "_try_payment_fallback", "_try_configured_fallback_for_unavailable_client",
    )
    for name in none_routes:
        mocks[name] = Mock(return_value=(None, None, ""))
        monkeypatch.setattr(ac, name, mocks[name])
    mocks["_try_main_agent_model_fallback"] = Mock(return_value=(remote, "cloud-model", "openrouter"))
    monkeypatch.setattr(ac, "_try_main_agent_model_fallback", mocks["_try_main_agent_model_fallback"])
    for name in ("_refresh_provider_credentials", "_recover_provider_pool",
                 "_nous_portal_account_has_fresh_paid_access"):
        mocks[name] = Mock(return_value=False)
        monkeypatch.setattr(ac, name, mocks[name])
    for name in ("_peek_pool_entry", "_pool_cache_hint", "_recoverable_pool_provider"):
        mocks[name] = Mock(return_value=None)
        monkeypatch.setattr(ac, name, mocks[name])
    for name in ("_read_main_api_key_if_same_origin", "_scoped_key_env"):
        mocks[name] = Mock(return_value="")
        monkeypatch.setattr(ac, name, mocks[name])
    mocks["_resolve_api_key_provider"] = Mock(return_value=(None, None))
    monkeypatch.setattr(ac, "_resolve_api_key_provider", mocks["_resolve_api_key_provider"])
    monkeypatch.setattr(ac, "_transient_retry_count", lambda: 1)
    monkeypatch.setattr(ac.time, "sleep", lambda _: None)
    monkeypatch.setattr(ac, "_client_cache", {})
    monkeypatch.setattr(ac, "remember_temperature_rejection", lambda *args: None)
    ac._reset_aux_unhealthy_cache()
    yield SimpleNamespace(cfg=cfg, local=local, remote=remote, factory=factory, mocks=mocks)
    ac._reset_aux_unhealthy_cache()


def invoke(async_mode=False, **kw):
    if async_mode:
        return asyncio.run(ac.async_call_llm(task=TASK, messages=MESSAGES, **kw))
    return ac.call_llm(task=TASK, messages=MESSAGES, **kw)


def use_async(routing, async_mode):
    if async_mode:
        routing.local.chat.completions.create = AsyncMock(return_value=response())
        routing.remote.chat.completions.create = AsyncMock(return_value=response("cloud title"))


@pytest.mark.parametrize("async_mode", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("kind", ["connection", "timeout", "quota", "rate", "payment", "model"])
def test_legacy_explicit_pin_reaches_implicit_main(routing, async_mode, kind):
    use_async(routing, async_mode)
    routing.local.chat.completions.create.side_effect = failure(kind)
    result = invoke(async_mode)
    assert result is not None
    routing.mocks["_try_main_agent_model_fallback"].assert_called_once()
    routing.remote.chat.completions.create.assert_called_once()


@pytest.mark.parametrize("async_mode", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("kind", ["connection", "timeout", "quota", "rate", "payment", "auth", "model"])
def test_strict_pin_never_resolves_or_calls_another_route(routing, async_mode, kind):
    routing.cfg["provider_policy"] = "strict"
    # Strict wins over an explicitly populated task fallback list as well.
    routing.cfg["fallback_chain"] = [{"provider": "openrouter", "model": "cloud-model"}]
    use_async(routing, async_mode)
    exc = failure(kind)
    routing.local.chat.completions.create.side_effect = exc
    with pytest.raises(type(exc)) as caught:
        invoke(async_mode)
    assert caught.value is exc
    routing.remote.chat.completions.create.assert_not_called()
    for mock in routing.mocks.values():
        mock.assert_not_called()


@pytest.mark.parametrize("async_mode", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("kind", ["connection", "parameter"])
def test_strict_same_client_retries_can_succeed(routing, async_mode, kind):
    routing.cfg["provider_policy"] = "strict"
    use_async(routing, async_mode)
    good = response()
    routing.local.chat.completions.create.side_effect = [failure(kind), good]
    assert invoke(async_mode, temperature=0.2) is good
    assert routing.local.chat.completions.create.call_count == 2
    for mock in routing.mocks.values():
        mock.assert_not_called()


@pytest.mark.parametrize("missing", ["provider", "model", "base_url", "api_key"])
def test_strict_incomplete_route_stops_before_credentials(routing, missing):
    routing.cfg["provider_policy"] = "strict"
    routing.cfg.pop(missing)
    with pytest.raises((ValueError, ac.AuxiliaryClientUnavailable)):
        invoke()
    routing.factory.assert_not_called()
    for mock in routing.mocks.values():
        mock.assert_not_called()


def test_strict_unavailable_client_cannot_rescue_via_task_or_auto(routing, monkeypatch):
    routing.cfg["provider_policy"] = "strict"
    cached = Mock(return_value=(None, None))
    monkeypatch.setattr(ac, "_get_cached_client", cached)
    with pytest.raises(ac.AuxiliaryClientUnavailable):
        invoke()
    assert cached.call_count == 1
    for mock in routing.mocks.values():
        mock.assert_not_called()


@pytest.mark.parametrize("policy", [True, "strcit", "", None])
def test_invalid_policy_is_not_silently_legacy(routing, policy):
    routing.cfg["provider_policy"] = policy
    with pytest.raises(ValueError, match="provider_policy"):
        invoke()
    routing.factory.assert_not_called()


def test_explicit_legacy_policy_still_allows_fallback(routing):
    routing.cfg["provider_policy"] = "legacy"
    routing.local.chat.completions.create.side_effect = failure("connection")
    invoke()
    routing.remote.chat.completions.create.assert_called_once()


@pytest.mark.parametrize("async_mode", [False, True], ids=["sync", "async"])
def test_named_strict_route_resolves_only_its_configured_key_env(routing, monkeypatch, async_mode):
    routing.cfg.update(provider_policy="strict", provider="custom:dgx-lab")
    routing.cfg.pop("api_key")
    routing.cfg.pop("base_url")
    lookup = Mock(return_value={"base_url": LOCAL, "key_env": "OFFLINE_DGX_KEY"})
    monkeypatch.setattr("hermes_cli.runtime_provider._get_named_custom_provider", lookup)
    key_lookup = routing.mocks["_scoped_key_env"]
    key_lookup.return_value = "test-only-not-a-secret"
    use_async(routing, async_mode)
    exc = failure("payment")
    routing.local.chat.completions.create.side_effect = exc
    with pytest.raises(RuntimeError) as caught:
        invoke(async_mode)
    assert caught.value is exc
    lookup.assert_called_once_with("custom:dgx-lab")
    key_lookup.assert_called_once_with("OFFLINE_DGX_KEY")
    for name, mock in routing.mocks.items():
        if name != "_scoped_key_env":
            mock.assert_not_called()


@pytest.mark.parametrize("selected", ["auto", "main", "moa", "missing-provider", "openai-codex"])
def test_strict_cannot_discover_missing_or_unsupported_provider(routing, monkeypatch, selected):
    routing.cfg.update(provider_policy="strict", provider=selected)
    lookup = Mock(return_value=None)
    monkeypatch.setattr("hermes_cli.runtime_provider._get_named_custom_provider", lookup)
    with pytest.raises(ValueError):
        invoke()
    routing.factory.assert_not_called()
    for mock in routing.mocks.values():
        mock.assert_not_called()


@pytest.mark.parametrize("async_mode", [False, True], ids=["sync", "async"])
def test_strict_constructor_failure_does_not_enter_resolution_rescue(routing, async_mode):
    routing.cfg["provider_policy"] = "strict"
    exc = RuntimeError("offline client construction failed")
    routing.factory.side_effect = exc
    with pytest.raises(RuntimeError) as caught:
        invoke(async_mode)
    assert caught.value is exc
    for mock in routing.mocks.values():
        mock.assert_not_called()


def test_strict_cache_respects_profile_and_provider_eviction(routing, monkeypatch):
    routing.cfg["provider_policy"] = "strict"
    monkeypatch.setattr(ac, "hermes_home_key", lambda: "offline-profile-a")
    invoke()
    monkeypatch.setattr(ac, "hermes_home_key", lambda: "offline-profile-b")
    invoke()
    assert routing.factory.call_count == 2
    ac._evict_cached_clients("custom")
    invoke()
    assert routing.factory.call_count == 3
    monkeypatch.setattr(ac, "hermes_home_key", lambda: "offline-profile-a")
    invoke()
    assert routing.factory.call_count == 3


def test_strict_never_uses_legacy_cached_client(routing):
    invoke()
    routing.cfg["provider_policy"] = "strict"
    invoke()
    invoke()
    assert routing.factory.call_count == 2  # distinct strict cache; third call reuses it
    assert len(ac._client_cache) == 2
    for attempt in routing.local.chat.completions.create.call_args_list:
        assert attempt.kwargs["model"] == MODEL


@pytest.mark.parametrize("async_mode", [False, True], ids=["sync", "async"])
def test_strict_policy_snapshot_survives_config_mutation(routing, async_mode):
    routing.cfg["provider_policy"] = "strict"
    use_async(routing, async_mode)

    def fail_and_change_config(**kw):
        routing.cfg["provider_policy"] = "legacy"
        raise failure("payment")

    routing.local.chat.completions.create.side_effect = fail_and_change_config
    with pytest.raises(RuntimeError):
        invoke(async_mode)
    for mock in routing.mocks.values():
        mock.assert_not_called()


@pytest.mark.parametrize("async_mode", [False, True], ids=["sync", "async"])
def test_strict_disables_redirects_and_ambient_proxy(routing, async_mode):
    import httpx
    routing.cfg["provider_policy"] = "strict"
    use_async(routing, async_mode)
    invoke(async_mode)
    constructor = httpx.AsyncClient if async_mode else httpx.Client
    constructor.assert_called_once_with(follow_redirects=False, trust_env=False)
    assert routing.factory.call_args.kwargs["base_url"] == LOCAL


@pytest.mark.parametrize("kind", ["connection", "timeout", "quota", "payment", "auth", "model"])
def test_title_pilot_declines_model_upgrade_without_cloud(routing, monkeypatch, kind):
    from agent import title_generator
    routing.cfg["provider_policy"] = "strict"
    monkeypatch.setattr(title_generator, "_title_config", lambda: routing.cfg)
    routing.local.chat.completions.create.side_effect = failure(kind)
    assert title_generator.generate_title("Investigate offline routing") is None
    assert title_generator.derive_title("Investigate offline routing") == "Investigate offline routing"
    for mock in routing.mocks.values():
        mock.assert_not_called()


def test_strict_pilot_reads_real_isolated_task_config(routing, monkeypatch, tmp_path):
    import json
    from agent.auxiliary_task_config import _get_auxiliary_task_config
    from agent import title_generator

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr("hermes_cli.plugins.get_plugin_auxiliary_tasks", list)
    config = {"auxiliary": {TASK: {**routing.cfg, "provider_policy": "strict"}}}
    # JSON is a YAML subset; only synthetic configuration under isolated tmp_path.
    (tmp_path / "config.yaml").write_text(json.dumps(config), encoding="utf-8")
    monkeypatch.setattr(ac, "_get_auxiliary_task_config", _get_auxiliary_task_config)
    assert _get_auxiliary_task_config(TASK)["provider_policy"] == "strict"
    routing.local.chat.completions.create.side_effect = failure("quota")
    assert title_generator.generate_title("Offline configuration pilot") is None
    for mock in routing.mocks.values():
        mock.assert_not_called()


def test_strict_ignores_main_runtime_cloud_credentials(routing):
    routing.cfg["provider_policy"] = "strict"
    invoke(main_runtime={"provider": "openrouter", "base_url": REMOTE,
                         "api_key": "synthetic-cloud-key", "model": "cloud-model"})
    assert routing.factory.call_args.kwargs["api_key"] == routing.cfg["api_key"]
    assert routing.factory.call_args.kwargs["base_url"] == LOCAL
    for mock in routing.mocks.values():
        mock.assert_not_called()


@pytest.mark.parametrize("async_mode", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("kind", ["connection", "timeout", "quota", "payment", "auth", "model"])
def test_strict_parameter_retry_failure_stays_closed(routing, async_mode, kind):
    routing.cfg["provider_policy"] = "strict"
    use_async(routing, async_mode)
    narrowed = failure(kind)
    routing.local.chat.completions.create.side_effect = [failure("parameter"), narrowed]
    with pytest.raises(type(narrowed)) as caught:
        invoke(async_mode, temperature=0.2)
    assert caught.value is narrowed
    assert routing.local.chat.completions.create.call_count == 2
    for mock in routing.mocks.values():
        mock.assert_not_called()


@pytest.mark.parametrize("mode", ["responses", "codex_responses", "anthropic"])
def test_strict_rejects_native_adapters_before_credentials(routing, mode):
    routing.cfg.update(provider_policy="strict", api_mode=mode)
    with pytest.raises(ValueError, match="chat_completions"):
        invoke()
    routing.factory.assert_not_called()


@pytest.mark.parametrize("endpoint", ["", "file:///test", "http://user:pass@dgx.invalid/v1"])
def test_strict_rejects_incomplete_or_embedded_credential_urls(routing, endpoint):
    routing.cfg.update(provider_policy="strict", base_url=endpoint)
    with pytest.raises(ValueError, match="base_url"):
        invoke()
    routing.factory.assert_not_called()
