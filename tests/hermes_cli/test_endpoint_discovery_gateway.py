"""Authenticated gateway discovery must distinguish its dashboard from its API."""

import asyncio

import httpx
import pytest


@pytest.mark.parametrize("prefix", ["", "/openai"])
@pytest.mark.parametrize("entered_suffix", ["", "/v1/"])
@pytest.mark.parametrize("api_status", [200, 401, 403])
def test_gateway_discovery_preserves_prefix_and_auth(monkeypatch, prefix, entered_suffix, api_status):
    from hermes_cli.web_routers import config_env as mod
    from hermes_cli.web_models import CustomEndpointUpdate

    base = "http://gateway.test" + prefix
    seen = []

    def respond(request):
        seen.append(request.url.path)
        assert request.headers["authorization"] == "Bearer fixture-key"
        if request.url.path == prefix + "/v1/models":
            return httpx.Response(api_status, json={"data": [{"id": "dgx/fixture-model"}]})
        return httpx.Response(200, headers={"content-type": "text/html"}, text="<!doctype html><title>Dashboard</title>")

    monkeypatch.setattr(mod, "_endpoint_probe_client", lambda *args: httpx.AsyncClient(transport=httpx.MockTransport(respond)))
    # Only discovery is under test; never perform an inference request.
    async def transport_probe(*args):
        return ""
    monkeypatch.setattr(mod, "_probe_transport_route", transport_probe)
    body = CustomEndpointUpdate(name="gateway", model="", base_url=base + entered_suffix, api_key="fixture-key")
    result = asyncio.run(mod.validate_custom_endpoint(body))

    assert result["ok"] is (api_status == 200)
    assert result["models"] == (["dgx/fixture-model"] if api_status == 200 else [])
    if api_status == 200:
        assert result["resolved_base_url"] == base + "/v1"
    else:
        assert "rejected" in result["message"]
    expected = [prefix + "/models", prefix + "/v1/models"] if not entered_suffix else [prefix + "/v1/models"]
    assert seen == expected


@pytest.mark.parametrize("storage", ["providers", "custom_providers"])
@pytest.mark.parametrize("key_field", ["key_env", "api_key"])
@pytest.mark.parametrize("submitted_key, changed_url, expected_auth", [
    (None, False, "Bearer stored-fixture-key"),
    ("replacement-fixture-key", False, "Bearer replacement-fixture-key"),
    ("", False, None),
    (None, True, None),
    (None, "path", None),
    (None, "unknown", None),
])
def test_saved_endpoint_discovery_uses_only_its_own_credential(monkeypatch, storage, key_field, submitted_key, changed_url, expected_auth):
    from hermes_cli.config import get_config_path, save_config
    from hermes_cli.web_routers import config_env as mod
    from hermes_cli.web_models import CustomEndpointUpdate

    monkeypatch.setenv("HERMES_CUSTOM_BIFROST_API_KEY", "stored-fixture-key")
    entry = {"name": "Bifrost", "enabled": True, "discover_models": False,
             "base_url": "http://gateway.test/openai/v1", "model": "dgx/fixture-model",
             key_field: "HERMES_CUSTOM_BIFROST_API_KEY" if key_field == "key_env" else "stored-fixture-key"}
    cfg = {"model": {"provider": "nous", "default": "user-selected-model"},
           storage: {"bifrost": entry} if storage == "providers" else [entry]}
    save_config(cfg)
    before = get_config_path().read_bytes()
    seen = []

    def respond(request):
        seen.append(request.headers.get("authorization"))
        return httpx.Response(200, json={"data": [{"id": "dgx/fixture-model"}]})

    monkeypatch.setattr(mod, "_endpoint_probe_client", lambda *args: httpx.AsyncClient(transport=httpx.MockTransport(respond)))
    async def transport_probe(*args):
        return ""
    monkeypatch.setattr(mod, "_probe_transport_route", transport_probe)
    base = "http://gateway.test/openai/v1/"
    if changed_url is True:
        base = "http://other.test/v1"
    elif changed_url == "path":
        base = "http://gateway.test/other/v1"
    endpoint_id = "unknown" if changed_url == "unknown" else "bifrost"
    body = CustomEndpointUpdate(id=endpoint_id, name="Bifrost", base_url=base,
                                model="dgx/fixture-model", api_key=submitted_key)
    result = asyncio.run(mod.validate_custom_endpoint(body))

    assert result["models"] == ["dgx/fixture-model"]
    assert seen == [expected_auth]
    assert get_config_path().read_bytes() == before
    assert "stored-fixture-key" not in str(result)


def test_saved_endpoint_validation_binds_requested_profile(monkeypatch, tmp_path):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from hermes_cli import web_server_profiles as profiles
    from hermes_cli.config import save_config
    from hermes_cli.web_routers import config_env as mod

    profile_home = tmp_path / "secondary"
    profile_home.mkdir()
    (profile_home / ".env").write_text("HERMES_CUSTOM_BIFROST_API_KEY=secondary-fixture-key\n")
    monkeypatch.setenv("HERMES_CUSTOM_BIFROST_API_KEY", "wrong-launch-profile-key")
    with profiles._hermes_home_scope(profile_home):
        save_config({"providers": {"bifrost": {"base_url": "http://gateway.test/v1",
                     "key_env": "HERMES_CUSTOM_BIFROST_API_KEY", "enabled": True}}})
    monkeypatch.setattr(profiles, "_resolve_profile_dir", lambda name: profile_home)
    seen = []

    def respond(request):
        seen.append(request.headers.get("authorization"))
        return httpx.Response(200, json={"data": [{"id": "dgx/fixture-model"}]})

    monkeypatch.setattr(mod, "_endpoint_probe_client", lambda *args: httpx.AsyncClient(transport=httpx.MockTransport(respond)))
    async def transport_probe(*args):
        return ""
    monkeypatch.setattr(mod, "_probe_transport_route", transport_probe)
    app = FastAPI()
    app.include_router(mod.router)
    with TestClient(app) as client:
        result = client.post("/api/providers/custom-endpoints/validate?profile=secondary", json={
            "id": "bifrost", "name": "Bifrost", "base_url": "http://gateway.test/v1", "model": "",
        })
    assert result.status_code == 200
    assert result.json()["models"] == ["dgx/fixture-model"]
    assert seen == ["Bearer secondary-fixture-key"]


@pytest.mark.parametrize("root_payload", [None, {"data": []}, {"data": [{"id": "native-model"}]}])
def test_native_alternate_and_empty_catalog_remain_supported(monkeypatch, root_payload):
    from hermes_cli.web_routers import config_env as mod

    seen = []
    def respond(request):
        seen.append(request.url.path)
        if request.url.path == "/v1/models":
            return httpx.Response(404)
        if root_payload is None:
            return httpx.Response(200, headers={"content-type": "text/html"}, text="<!doctype html>")
        return httpx.Response(200, json=root_payload)

    monkeypatch.setattr(mod, "_endpoint_probe_client", lambda *args: httpx.AsyncClient(transport=httpx.MockTransport(respond)))
    resolved, response = asyncio.run(mod._probe_openai_compatible_models("http://native.test/v1", None))
    assert seen == ["/v1/models", "/models"]
    assert resolved == "http://native.test"
    if root_payload is None:
        assert "text/html" in mod._no_models_probe_message(response, resolved)
    else:
        assert response.json() == root_payload
