"""Product-neutral mapped memory credential contracts.

A backend containing a credential is insufficient: config must explicitly map
MEMORY_API_KEY. Hydration is cached, installed scopes are snapshots, and rotation
or revocation requires cache reset + hydration + explicit scope refresh. These
contracts exercise real config parsing, registry application and scoped reads;
only the external secret source is synthetic. No vendor plugin/SDK is imported.
"""
import json
import os
from pathlib import Path

import pytest

from agent import secret_scope
from agent.secret_sources import registry
from agent.secret_sources.base import ErrorKind, FetchResult, SecretSource
from hermes_cli import env_loader


class MappedMemorySource(SecretSource):
    name = "fixture_memory_vault"
    label = "Fixture memory vault"
    scheme = "fixturememory"
    shape = "mapped"

    def __init__(self):
        self.values = {"fixturememory://memory/key": "synthetic-memory-v1"}
        self.failure = None
        self.calls = 0

    def fetch(self, cfg, home_path):
        self.calls += 1
        if self.failure:
            # Even a nonconforming backend carrying partial secrets cannot publish
            # them once the result is marked non-ok.
            return FetchResult(secrets={"MEMORY_API_KEY": "synthetic-partial"}).fail(
                "credential rejected by fixture service", self.failure
            )
        return FetchResult(secrets={
            name: self.values[ref] for name, ref in cfg.get("env", {}).items()
            if ref in self.values
        })


@pytest.fixture
def memory_source(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "launch"))
    monkeypatch.delenv("HERMES_MANAGED_DIR", raising=False)
    registry._reset_registry_for_tests()
    env_loader.reset_secret_source_cache()
    source = MappedMemorySource()
    assert registry.register_source(source)
    scope_token = secret_scope.set_secret_scope(None)
    multiplex_token = secret_scope.set_multiplex_context(True)
    try:
        yield source
    finally:
        secret_scope.reset_multiplex_context(multiplex_token)
        secret_scope.reset_secret_scope(scope_token)
        registry._reset_registry_for_tests()
        env_loader.reset_secret_source_cache()


def write_home(home, *, mapped=True, override=True, enabled=True):
    home.mkdir(parents=True, exist_ok=True)
    section = {"enabled": enabled, "override_existing": override,
               "env": {"MEMORY_API_KEY": "fixturememory://memory/key"} if mapped else {}}
    config = {"secrets": {"fixture_memory_vault": section}}
    (home / "config.yaml").write_text(json.dumps(config), encoding="utf-8")
    return config


@pytest.mark.parametrize("mapped", [False, True], ids=["mapping-omitted", "mapping-present"])
def test_only_explicit_mapping_reaches_scoped_memory_consumer(tmp_path, memory_source, mapped):
    home = tmp_path / "profile"
    config = write_home(home, mapped=mapped)
    target = {}
    report = registry.apply_all(config["secrets"], home, environ=target)
    expected = {"MEMORY_API_KEY": "synthetic-memory-v1"} if mapped else {}
    assert target == expected
    assert set(report.provenance) == set(expected)
    assert env_loader.hydrate_profile_secret_sources(home) == expected
    token = secret_scope.set_secret_scope(secret_scope.build_profile_secret_scope(home), profile_home=str(home))
    try:
        assert secret_scope.get_secret("MEMORY_API_KEY") == expected.get("MEMORY_API_KEY")
    finally:
        secret_scope.reset_secret_scope(token)


@pytest.mark.parametrize("override", [False, True], ids=["stale-kept-by-policy", "explicit-override"])
def test_mapping_obeys_explicit_override_for_shell_and_dotenv(tmp_path, memory_source, override):
    home = tmp_path / "profile"
    config = write_home(home, override=override)
    (home / ".env").write_text("MEMORY_API_KEY=synthetic-stale\n", encoding="utf-8")
    target = {"MEMORY_API_KEY": "synthetic-stale"}
    report = registry.apply_all(config["secrets"], home, environ=target)
    expected = "synthetic-memory-v1" if override else "synthetic-stale"
    assert target["MEMORY_API_KEY"] == expected
    if override:
        assert report.provenance["MEMORY_API_KEY"].overrode_env
        assert report.provenance["MEMORY_API_KEY"].authoritative
    else:
        assert report.sources[0].skipped_existing == ["MEMORY_API_KEY"]
    env_loader.hydrate_profile_secret_sources(home)
    token = secret_scope.set_secret_scope(secret_scope.build_profile_secret_scope(home), profile_home=str(home))
    try:
        assert secret_scope.get_secret("MEMORY_API_KEY") == expected
    finally:
        secret_scope.reset_secret_scope(token)


@pytest.mark.parametrize("change", ["rotate", "remove-mapping", "disable-source"])
def test_refresh_replaces_installed_snapshot_and_revokes_removed_key(tmp_path, memory_source, change):
    home = tmp_path / "profile"
    write_home(home)
    env_loader.hydrate_profile_secret_sources(home)
    snapshot = secret_scope.build_profile_secret_scope(home)
    token = secret_scope.set_secret_scope(snapshot, profile_home=str(home))
    try:
        if change == "rotate":
            memory_source.values["fixturememory://memory/key"] = "synthetic-memory-v2"
        else:
            write_home(home, mapped=change != "remove-mapping", enabled=change != "disable-source")
        assert env_loader.hydrate_profile_secret_sources(home)["MEMORY_API_KEY"] == "synthetic-memory-v1"
        env_loader.reset_secret_source_cache(home)
        env_loader.hydrate_profile_secret_sources(home)
        assert secret_scope.get_secret("MEMORY_API_KEY") == "synthetic-memory-v1"
        assert secret_scope.refresh_installed_secret_scope(home)
        expected = "synthetic-memory-v2" if change == "rotate" else None
        assert secret_scope.get_secret("MEMORY_API_KEY") == expected
        assert snapshot.get("MEMORY_API_KEY") == expected
    finally:
        secret_scope.reset_secret_scope(token)


def test_profile_miss_never_falls_back_to_launch_or_sibling_key(tmp_path, monkeypatch, memory_source):
    home_a, home_b = tmp_path / "profile-a", tmp_path / "profile-b"
    write_home(home_a)
    write_home(home_b, mapped=False)
    monkeypatch.setenv("MEMORY_API_KEY", "synthetic-launch-only")
    before = dict(os.environ)
    env_loader.hydrate_profile_secret_sources(home_a)
    assert env_loader.hydrate_profile_secret_sources(home_b) == {}
    assert dict(os.environ) == before
    for home, expected in [(home_a, "synthetic-memory-v1"), (home_b, None), (home_a, "synthetic-memory-v1")]:
        token = secret_scope.set_secret_scope(secret_scope.build_profile_secret_scope(home), profile_home=str(home))
        try:
            assert secret_scope.get_secret("MEMORY_API_KEY") == expected
        finally:
            secret_scope.reset_secret_scope(token)
    with pytest.raises(secret_scope.UnscopedSecretError):
        secret_scope.get_secret("MEMORY_API_KEY")


@pytest.mark.parametrize("kind", [ErrorKind.AUTH_FAILED, ErrorKind.AUTH_EXPIRED])
def test_controlled_auth_failure_publishes_no_partial_values_and_actionable_diagnostic(
        tmp_path, monkeypatch, memory_source, capsys, caplog, kind):
    home = tmp_path / "profile"
    config = write_home(home)
    memory_source.failure = kind
    target = {}
    report = registry.apply_all(config["secrets"], home, environ=target)
    assert target == {}
    assert not report.applied_any
    assert report.sources[0].result.error_kind == kind
    assert env_loader.hydrate_profile_secret_sources(home) == {}
    monkeypatch.delenv("MEMORY_API_KEY", raising=False)
    env_loader.reset_secret_source_cache(home)
    env_loader._apply_external_secret_sources(home)
    diagnostic = capsys.readouterr().err + caplog.text
    assert "credential rejected" in diagnostic
    assert "hermes secrets fixture_memory_vault setup" in diagnostic
    for raw in ["synthetic-memory-v1", "synthetic-partial"]:
        assert raw not in diagnostic
    assert secret_scope.build_profile_secret_scope(home).get("MEMORY_API_KEY") is None
