"""Offline regressions for automatic Kanban worker safe routing.

Automatic workers freeze their scoped route and fallback policy through a real
scratch board claim and persisted launch policy. Provider resolution and the
agent's API boundary are stubbed; the request-failure tests use a real SDK
exception, classifier and terminal result builder. No provider client, network
endpoint, or credential store is contacted.
"""

from __future__ import annotations

import json
import socket
from types import SimpleNamespace
from unittest.mock import patch
import urllib.request

import pytest

from gateway.session_kanban import _kanban_worker_policy_config, _kanban_worker_route_params
from hermes_cli.auth import AuthError
from hermes_cli.fallback_config import get_fallback_chain
from hermes_cli.kanban_db import KANBAN_TERMINAL_PROVIDER_EXIT_CODE
from hermes_cli.runtime_provider import resolve_runtime_with_fallback
from hermes_cli.turn_exit import credential_failure_flags, turn_exit_code


SUBSCRIPTION_ROUTE = {
    "provider": "openai-codex",
    "model": "gpt-6.1-sol-900k",
}
PAID_FALLBACKS = [
    {"provider": "custom:bifrost", "model": "ollama/glm-5.2"},
    {"provider": "xai-oauth", "model": "grok-4.6"},
    {"provider": "copilot", "model": "claude-sonnet-5.5"},
    {"provider": "openai-codex", "model": "gpt-5.5"},
    {"provider": "openai-codex", "model": "gpt-5.6-terra-900k"},
]


@pytest.fixture(autouse=True)
def _forbid_network(monkeypatch):
    """These regressions are pure routing checks; any network path is a test bug."""

    def blocked(*args, **kwargs):
        raise AssertionError(f"network access forbidden in offline routing test: {args!r} {kwargs!r}")

    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.setattr(socket, "getaddrinfo", blocked)
    monkeypatch.setattr(urllib.request, "urlopen", blocked)


def _safe_route_config():
    return {
        "model": {"provider": "openai-codex", "default": "interactive-chat-default"},
        "fallback_providers": list(PAID_FALLBACKS),
        "fallback_model": {"provider": "openai-codex", "model": "gpt-5.6-terra-900k"},
        "kanban": {
            "worker_provider": SUBSCRIPTION_ROUTE["provider"],
            "worker_model": SUBSCRIPTION_ROUTE["model"],
            "worker_fallback_providers": [],
            "worker_fail_fast_on_route_unavailable": True,
        },
        "auxiliary": {"free_only": True},
    }


def _task(**overrides):
    base = {
        "provider_override": None,
        "model_override": None,
        "reasoning_effort": None,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def test_automatic_worker_launch_selects_verified_subscription_route():
    route = _kanban_worker_route_params(_task(), _safe_route_config())

    assert route == SUBSCRIPTION_ROUTE


def test_automatic_worker_subscription_unavailable_fails_fast_without_paid_fallback():
    config = _kanban_worker_policy_config(_safe_route_config())
    attempted = []

    def fail_or_paid(**kwargs):
        attempted.append((kwargs.get("requested"), kwargs.get("target_model")))
        if kwargs.get("requested") != SUBSCRIPTION_ROUTE["provider"]:
            raise AssertionError(f"paid fallback route was attempted: {kwargs!r}")
        raise AuthError(
            "model catalog miss",
            provider="openai-codex",
            code="model_not_found",
            relogin_required=True,
        )

    with patch("hermes_cli.runtime_provider.resolve_runtime_provider", side_effect=fail_or_paid):
        with pytest.raises(AuthError, match="automatic paid fallback disabled") as excinfo:
            resolve_runtime_with_fallback(
                config,
                requested=SUBSCRIPTION_ROUTE["provider"],
                target_model=SUBSCRIPTION_ROUTE["model"],
            )

    assert attempted == [(SUBSCRIPTION_ROUTE["provider"], SUBSCRIPTION_ROUTE["model"])]
    flags = credential_failure_flags(excinfo.value)
    assert turn_exit_code(None, kanban_worker=True, **flags) == KANBAN_TERMINAL_PROVIDER_EXIT_CODE


def test_worker_fallback_control_suppresses_paid_entries_without_mutating_profile_config():
    profile_config = _safe_route_config()
    scoped = _kanban_worker_policy_config(profile_config)

    assert get_fallback_chain(scoped) == []
    assert scoped["fallback_model"] is None
    assert scoped["fallback_providers"] == []
    assert get_fallback_chain(profile_config) == PAID_FALLBACKS
    assert profile_config["fallback_model"] == {"provider": "openai-codex", "model": "gpt-5.6-terra-900k"}


def test_fail_fast_control_alone_suppresses_inherited_fallback_entries():
    profile_config = _safe_route_config()
    profile_config["kanban"].pop("worker_fallback_providers")

    scoped = _kanban_worker_policy_config(profile_config)

    assert get_fallback_chain(scoped) == []
    assert get_fallback_chain(profile_config) == PAID_FALLBACKS


def test_explicit_local_route_override_remains_reachable_and_unaffected():
    local_task = _task(provider_override="custom:bifrost", model_override="dgx/glm-5.3-flash")
    route = _kanban_worker_route_params(local_task, _safe_route_config())

    assert route == {"provider": "custom:bifrost", "model": "dgx/glm-5.3-flash"}


def test_worker_safe_routing_preserves_configured_auxiliary_free_only():
    assert _kanban_worker_policy_config(_safe_route_config())["auxiliary"]["free_only"] is True


def test_worker_safe_routing_defaults_are_registered():
    from hermes_cli.config_defaults import DEFAULT_CONFIG

    assert DEFAULT_CONFIG["kanban"]["worker_provider"] == ""
    assert DEFAULT_CONFIG["kanban"]["worker_model"] == ""
    assert DEFAULT_CONFIG["kanban"]["worker_fallback_providers"] is None
    assert DEFAULT_CONFIG["kanban"]["worker_fail_fast_on_route_unavailable"] is False


def test_safe_route_config_is_json_serializable_and_contains_no_credentials():
    """Regression guard: test fixtures stay deterministic and credential-free."""
    rendered = json.dumps(_safe_route_config(), sort_keys=True)

    assert "api_key" not in rendered
    assert "token" not in rendered.lower()
    assert "secret" not in rendered.lower()


@pytest.mark.parametrize("merged_defaults", [False, True])
def test_none_worker_fallback_inherits_legacy_profile_chain(merged_defaults):
    from copy import deepcopy
    from hermes_cli.config import DEFAULT_CONFIG, _deep_merge

    config = _safe_route_config()
    config["kanban"] = {"worker_fallback_providers": None}
    if merged_defaults:
        config = _deep_merge(deepcopy(DEFAULT_CONFIG), config)
    original = deepcopy(config)
    assert get_fallback_chain(_kanban_worker_policy_config(config)) == PAID_FALLBACKS
    assert config == original


@pytest.mark.parametrize("declared", [None, []])
def test_fail_fast_overrides_none_or_empty_fallback(declared):
    config = _safe_route_config()
    config["kanban"]["worker_fallback_providers"] = declared
    assert get_fallback_chain(_kanban_worker_policy_config(config)) == []


@pytest.mark.parametrize("controls", [
    {"worker_fallback_providers": "openrouter"},
    {"worker_fallback_providers": {}},
    {"worker_fallback_providers": False},
    {"worker_fallback_providers": [None]},
    {"worker_fallback_providers": [{}]},
    {"worker_fallback_providers": [{"provider": "openrouter", "model": 123}]},
    {"worker_fallback_providers": [{"provider": " ", "model": "test"}]},
    {"worker_fail_fast_on_route_unavailable": "true"},
    {"worker_fail_fast_on_route_unavailable": 1},
    {"worker_fail_fast_on_route_unavailable": None},
    {"worker_fail_fast_on_route_unavailable": True, "worker_fallback_providers": PAID_FALLBACKS},
    {"worker_provider": False},
    {"worker_model": ["test"]},
])
def test_malformed_worker_controls_rejected_before_resolution(controls, claimed_worker):
    config = _safe_route_config()
    config["kanban"] = controls
    with patch("hermes_cli.runtime_provider.resolve_runtime_provider") as resolver:
        with pytest.raises(ValueError, match=r"kanban\.worker_"):
            claimed_worker(config)
        resolver.assert_not_called()


def test_explicit_worker_chain_replaces_profile_and_legacy_fallbacks():
    config = _safe_route_config()
    config["kanban"] = {"worker_fallback_providers": [SUBSCRIPTION_ROUTE]}
    assert get_fallback_chain(_kanban_worker_policy_config(config)) == [SUBSCRIPTION_ROUTE]


@pytest.fixture
def claimed_worker(tmp_path, monkeypatch):
    """Real board + claim + home, never a caller-invented launch policy or global DB."""
    from contextlib import closing
    from dataclasses import asdict
    from pathlib import Path
    from gateway.session_contract import Principal
    from gateway.session_kanban import build_kanban_policy
    from gateway.session_policy import restore_policy
    from hermes_cli import kanban_db as kb
    from hermes_cli.kanban_db_connect import connect
    from hermes_cli.config import atomic_config_write

    home = tmp_path / "profile"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(tmp_path / "boards"))
    for key in ("HERMES_KANBAN_DB", "HERMES_KANBAN_BOARD", "HERMES_DELEGATED_CHILD"):
        monkeypatch.delenv(key, raising=False)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    kb.create_board("routing")
    actor = Principal("owner", str(home), frozenset({"session:create"}), "transport")
    authority = SimpleNamespace(profile_id=str(home))
    connection = SimpleNamespace(authority=authority, actor=actor, native_owner=True)

    def build(config, **overrides):
        # Use the real loader's scratch profile later; no config reader is mocked.
        atomic_config_write(home / "config.yaml", config)
        with closing(connect(board="routing")) as conn:
            task_id = kb.create_task(conn, title="Scoped routing", assignee="default",
                workspace_kind="dir", workspace_path=str(workspace), **overrides)
            kb.recompute_ready(conn)
            task = kb.claim_task(conn, task_id)
            assert task is not None
        params = dict(board="routing", task_id=task.id, run_id=task.current_run_id, claim_lock=task.claim_lock)
        policy, _secrets = build_kanban_policy(connection, params, config)
        persisted = tmp_path / "policy.json"
        persisted.write_text(json.dumps(asdict(policy)), encoding="utf-8")
        restored = restore_policy(json.loads(persisted.read_text(encoding="utf-8")))
        assert restored == policy
        context = json.loads(restored.kanban_json)
        assert Path(context["db"]).is_relative_to(tmp_path)
        assert context["task_id"] == task.id and context["claim_lock"] == task.claim_lock
        return restored, authority, home

    return build


@pytest.mark.parametrize("override", [None, "local", "subscription"])
@pytest.mark.parametrize("fail_fast", [False, True])
def test_full_claim_freeze_restore_and_worker_fallback_consumers(claimed_worker, override, fail_fast):
    from copy import deepcopy
    from gateway.session_worker_construct import _fallback_chain, route_runtime, worker_api_key

    config = _safe_route_config()
    config["kanban"]["worker_fail_fast_on_route_unavailable"] = fail_fast
    original = deepcopy(config)
    pins = {
        None: {},
        "local": {"provider_override": "custom:local-test", "model_override": "local-model"},
        "subscription": {"provider_override": "openai-codex", "model_override": "explicit-subscription"},
    }[override]
    policy, authority, _home = claimed_worker(config, **pins)
    provider = pins.get("provider_override", SUBSCRIPTION_ROUTE["provider"])
    model = pins.get("model_override", SUBSCRIPTION_ROUTE["model"])
    assert policy.provider == provider and policy.model == model
    assert policy.source == "kanban" and not policy.ignore_user_config
    assert get_fallback_chain(policy.config()) == []
    assert policy.config()["auxiliary"]["free_only"] is True
    assert config == original

    refreshed = []
    def refresh():
        refreshed.append(True)
        return PAID_FALLBACKS
    authority.runner = SimpleNamespace(_refresh_fallback_model=refresh)
    chain = _fallback_chain(authority, policy)
    assert chain is None and refreshed == []
    reachable = {"provider": provider, "model": model, "api_key": "offline-only"}
    with patch("hermes_cli.runtime_provider.resolve_runtime_provider", return_value=reachable) as resolver:
        runtime, fallback = route_runtime(policy, policy.config(), model, None)
        assert runtime == reachable and fallback is None
        assert resolver.call_count == 1
        assert resolver.call_args.kwargs["requested"] == provider
        assert resolver.call_args.kwargs["target_model"] == model
    primary = AuthError("model catalog miss", provider=provider, code="model_not_found", relogin_required=True)
    with patch("hermes_cli.runtime_provider.resolve_runtime_provider", side_effect=primary) as resolver:
        with pytest.raises(AuthError, match="no other provider attempted") as error:
            route_runtime(policy, policy.config(), model, None)
        assert error.value is primary and error.value.code == "model_not_found"
        assert error.value.relogin_required is True
        assert provider + "/" + model in str(error.value)
        assert "model catalog miss" in str(error.value)
        assert resolver.call_count == 1
        resolver.reset_mock()
        with pytest.raises(AuthError):
            worker_api_key(policy, {"fallback_model": chain})
        assert resolver.call_count == 1
        assert resolver.call_args.kwargs["requested"] == provider
        assert resolver.call_args.kwargs["target_model"] == model


def test_auxiliary_main_fallback_uses_frozen_kanban_scope(claimed_worker):
    from agent import auxiliary_client as auxiliary
    from gateway.session_policy import policy_scope
    from hermes_cli.config import load_config_readonly

    policy, authority, home = claimed_worker(_safe_route_config())
    before = (home / "config.yaml").read_bytes()
    # The real profile reader still has the paid chain: only the worker scope may suppress it.
    assert get_fallback_chain(load_config_readonly()) == PAID_FALLBACKS
    with patch.object(auxiliary, "resolve_provider_client", return_value=(None, None)) as resolver:
        with policy_scope(policy, authority=authority):
            assert auxiliary._try_main_fallback_chain("compression", "openai-codex") == (None, None, "")
        resolver.assert_not_called()
        # Leaving the worker context restores ordinary top-level fallback behavior.
        auxiliary._try_main_fallback_chain("compression", "openai-codex")
        assert resolver.call_count > 0
    assert (home / "config.yaml").read_bytes() == before


def test_normal_defaults_leave_auxiliary_and_worker_routes_opt_in():
    from hermes_cli.config import DEFAULT_CONFIG

    assert DEFAULT_CONFIG["auxiliary"]["free_only"] is False
    assert DEFAULT_CONFIG["kanban"]["worker_fallback_providers"] is None
    assert DEFAULT_CONFIG["kanban"]["worker_fail_fast_on_route_unavailable"] is False


@pytest.mark.parametrize("declared", ["absent", None, []])
def test_fail_fast_always_suppresses_inherited_and_legacy_fallback(declared):
    config = _safe_route_config()
    config["kanban"] = {"worker_fail_fast_on_route_unavailable": True}
    if declared != "absent":
        config["kanban"]["worker_fallback_providers"] = declared
    scoped = _kanban_worker_policy_config(config)
    assert get_fallback_chain(scoped) == []
    primary = AuthError("model catalog miss", provider="openai-codex", code="model_not_found",
                        relogin_required=True, retryable=False, retry_after=3)
    with patch("hermes_cli.runtime_provider.resolve_runtime_provider", side_effect=primary) as resolver:
        with pytest.raises(AuthError, match="no other provider attempted") as error:
            resolve_runtime_with_fallback(scoped, requested="openai-codex", target_model="worker-model")
        assert error.value is primary
        assert (primary.provider, primary.code, primary.relogin_required, primary.retryable, primary.retry_after) == (
            "openai-codex", "model_not_found", True, False, 3)
        assert resolver.call_count == 1
        setting = ("kanban.worker_fallback_providers=[]" if declared == []
                   else "kanban.worker_fail_fast_on_route_unavailable=true")
        assert setting in str(primary)


def test_full_merged_defaults_inherit_frozen_chain_and_ordinary_sessions_still_refresh(claimed_worker):
    from copy import deepcopy
    from dataclasses import replace
    from gateway.session_worker_construct import _fallback_chain, route_runtime
    from hermes_cli.config import DEFAULT_CONFIG, _deep_merge

    config = _safe_route_config()
    config["kanban"] = {}
    config = _deep_merge(deepcopy(DEFAULT_CONFIG), config)
    policy, authority, home = claimed_worker(config)
    before = (home / "config.yaml").read_bytes()
    refreshed = []
    def refresh():
        refreshed.append(True)
        return [SUBSCRIPTION_ROUTE]
    authority.runner = SimpleNamespace(_refresh_fallback_model=refresh)
    assert _fallback_chain(authority, policy) == PAID_FALLBACKS
    assert refreshed == []
    ordinary = replace(policy, source="cli", kanban_json=None)
    assert _fallback_chain(authority, ordinary) == [SUBSCRIPTION_ROUTE]
    assert refreshed == [True]
    expected = {"provider": PAID_FALLBACKS[0]["provider"], "model": PAID_FALLBACKS[0]["model"]}
    primary = AuthError("model catalog miss", provider=policy.provider, code="model_not_found")
    with patch("hermes_cli.runtime_provider.resolve_runtime_provider", side_effect=[primary, expected]) as resolver:
        runtime, entry = route_runtime(policy, policy.config(), policy.model, None)
        assert runtime == expected and entry == PAID_FALLBACKS[0]
        assert [call.kwargs["requested"] for call in resolver.call_args_list] == [policy.provider, expected["provider"]]
    assert "automatic paid fallback disabled" not in str(primary)
    assert (home / "config.yaml").read_bytes() == before


def _request_failure_agent(policy):
    """Only the API/agent boundary is fake; classification and terminal copy are real."""
    import httpx
    import openai
    from unittest.mock import Mock
    from agent.api_error_summary import ApiErrorSummaryMixin
    from agent.error_classifier import classify_api_error
    from agent.turn_recovery import nonretryable_client_error_result

    error = openai.NotFoundError("Unknown model", response=httpx.Response(404,
        request=httpx.Request("POST", "https://offline.invalid/responses")),
        body={"error": {"code": "model_not_found", "message": "Unknown model"}})
    agent = SimpleNamespace(provider=policy.provider, model=policy.model, session_id="offline-worker",
        _fallback_chain=[], _fallback_model=None, _fallback_activated=False,
        _flush_status_buffer=Mock(), _emit_diagnostic_status=Mock(), _persist_session=Mock(),
        _vprint=Mock(), log_prefix="", verbose_logging=False,
        _summarize_api_error=ApiErrorSummaryMixin._summarize_api_error)
    originals = []

    def run(*args, **kwargs):
        classified = classify_api_error(error, provider=agent.provider, model=agent.model)
        result = nonretryable_client_error_result(agent, error, classified, status_code=404,
            api_kwargs=None, api_messages=[], messages=[], conversation_history=[],
            api_call_count=1, approx_tokens=10, provider=agent.provider,
            base_url="https://offline.invalid", model=agent.model, delivered="Partial work retained.")
        # Existing turn accounting must survive the worker's diagnostic projection.
        result.update(prompt_tokens=17, completion_tokens=5, estimated_cost_usd=0.0,
                      cost_status="known", failure_resets_at=123)
        originals.append(dict(result))
        return result

    agent.run_conversation = Mock(side_effect=run)
    return agent, originals, error


@pytest.mark.parametrize("control", ["empty_chain", "fail_fast"])
@pytest.mark.parametrize("override", [False, True])
def test_managed_terminal_request_failure_names_scoped_route_without_losing_verdict(claimed_worker, control, override):
    from dataclasses import asdict
    from agent.managed_worker import accept_result, result_frame
    from gateway.session_kanban import run_worker_turns

    config = _safe_route_config()
    if control == "fail_fast":
        config["kanban"].pop("worker_fallback_providers")
    pins = {"provider_override": "custom:local-test", "model_override": "local-model"} if override else {}
    policy, _authority, home = claimed_worker(config, **pins)
    frame = {"policy": asdict(policy), "text": "Do the task"}
    agent, originals, error = _request_failure_agent(policy)
    before = (home / "config.yaml").read_bytes()
    with patch("gateway.session_kanban._record_worker_result") as record:
        result = run_worker_turns(agent, frame, [])

    assert result["failure_reason"] == "model_not_found" and result["failure_retryable"] is False
    assert result["final_response"].startswith(originals[0]["final_response"])
    assert policy.provider + "/" + policy.model in result["final_response"]
    assert "model_not_found" in result["final_response"]
    assert ("ChatGPT/Codex OAuth subscription route" in result["final_response"]) is (not override)
    setting = ("kanban.worker_fallback_providers=[]" if control == "empty_chain"
               else "kanban.worker_fail_fast_on_route_unavailable=true")
    assert "automatic paid fallback disabled by " + setting in result["final_response"]
    assert "no other provider attempted" in result["final_response"]
    assert {k: v for k, v in result.items() if k != "final_response"} == {
        k: v for k, v in originals[0].items() if k != "final_response"}
    assert "automatic paid fallback disabled" not in str(error)
    assert agent.run_conversation.call_count == 1
    record.assert_called_once_with(json.loads(policy.kanban_json), KANBAN_TERMINAL_PROVIDER_EXIT_CODE,
                                   result["final_response"][-500:])
    accepted, usage = accept_result(result_frame(result))
    assert accepted["final_response"] == result["final_response"]
    assert accepted["error"] == originals[0]["error"] and accepted["partial"] is True
    assert usage == {"input_tokens": 17, "output_tokens": 5, "total_tokens": 22}
    assert (home / "config.yaml").read_bytes() == before


@pytest.mark.parametrize("evidence", ["frozen_chain", "legacy_chain", "scoped_chain", "construct_chain",
                                      "agent_chain", "agent_legacy", "activated", "advanced", "route_changed"])
def test_managed_failure_does_not_claim_no_fallback_when_evidence_disagrees(claimed_worker, evidence):
    from dataclasses import asdict
    from gateway.session_kanban import run_worker_turns

    policy, _authority, _home = claimed_worker(_safe_route_config())
    frozen = policy.config()
    request = json.loads(policy.request_json)
    request["construct_v1"] = {"fallback_model": None, "prefill_messages": None, "ephemeral_system_prompt": None}
    agent, originals, _error = _request_failure_agent(policy)
    target, key, value = {
        "frozen_chain": (frozen, "fallback_providers", PAID_FALLBACKS),
        "legacy_chain": (frozen, "fallback_model", PAID_FALLBACKS[0]),
        "scoped_chain": (frozen["kanban"], "worker_fallback_providers", PAID_FALLBACKS),
        "construct_chain": (request["construct_v1"], "fallback_model", PAID_FALLBACKS),
        "agent_chain": (vars(agent), "_fallback_chain", PAID_FALLBACKS),
        "agent_legacy": (vars(agent), "_fallback_model", PAID_FALLBACKS[0]),
        "activated": (vars(agent), "_fallback_activated", True),
        "advanced": (vars(agent), "_fallback_index", 1),
        "route_changed": (vars(agent), "provider", "custom:bifrost"),
    }[evidence]
    target[key] = value
    frame = {"policy": {**asdict(policy), "config_json": json.dumps(frozen), "request_json": json.dumps(request)},
             "text": "Do the task"}
    with patch("gateway.session_kanban._record_worker_result"):
        result = run_worker_turns(agent, frame, [])
    assert result == originals[0]
    assert agent.run_conversation.call_count == 1


@pytest.mark.parametrize("case", ["ordinary_chat", "global_disabled_only", "inherited_chain", "scoped_chain"])
def test_managed_diagnostic_requires_worker_scoped_suppression(claimed_worker, case):
    from dataclasses import asdict, replace
    from gateway.session_kanban import run_worker_turns

    config = _safe_route_config()
    if case != "ordinary_chat":
        config["kanban"] = {}
    if case == "global_disabled_only":
        config.update(fallback_providers=[], fallback_model=None)
    elif case == "scoped_chain":
        config["kanban"]["worker_fallback_providers"] = PAID_FALLBACKS
    policy, _authority, _home = claimed_worker(config)
    if case == "ordinary_chat":
        policy = replace(policy, source="cli", kanban_json=None)
    agent, originals, _error = _request_failure_agent(policy)
    with patch("gateway.session_kanban._record_worker_result") as record:
        result = run_worker_turns(agent, {"policy": asdict(policy), "text": "Do the task"}, [])
    assert result == originals[0]
    if case == "ordinary_chat":
        record.assert_not_called()


@pytest.mark.parametrize("verdict", [{"failed": False}, {"interrupted": True},
                                    {"failure_retryable": True}, {"failure_reason": "content_policy_blocked"}])
def test_scoped_diagnostic_leaves_other_terminal_verdicts_unchanged(claimed_worker, verdict):
    from dataclasses import asdict
    from gateway.session_kanban import run_worker_turns

    policy, _authority, _home = claimed_worker(_safe_route_config())
    agent, _originals, _error = _request_failure_agent(policy)
    result = {**agent.run_conversation(), **verdict}
    agent.run_conversation.return_value = result
    agent.run_conversation.side_effect = None
    with patch("gateway.session_kanban._record_worker_result"), \
            patch("agent.kanban_turn_recovery.recover_failed_kanban_turns"):
        projected = run_worker_turns(agent, {"policy": asdict(policy), "text": "Do the task"}, [])
    assert projected == result
