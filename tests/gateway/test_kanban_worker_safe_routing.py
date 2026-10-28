"""Offline regressions for automatic Kanban worker safe routing.

Automatic workers use a Kanban-scoped launch route and must fail closed when
that subscription route is unavailable. These tests use only in-process stubs;
no provider client, network endpoint, or credential store is contacted.
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
            "model catalog miss: automatic paid fallback disabled by kanban.worker_fallback_providers=[]; "
            "no paid fallback providers were attempted",
            provider=SUBSCRIPTION_ROUTE["provider"],
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


def test_worker_safe_routing_preserves_auxiliary_free_only_default():
    from hermes_cli.config_defaults import DEFAULT_CONFIG

    assert DEFAULT_CONFIG["auxiliary"]["free_only"] is True


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
