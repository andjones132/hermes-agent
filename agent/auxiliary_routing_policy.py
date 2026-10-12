"""Auxiliary route selection and opt-in fail-closed policy.

Strict is deliberately bounded to pinned OpenAI-compatible chat routes (inline
``custom`` or a named custom entry). Native/OAuth/discovery routes fail validation
rather than claiming a spend boundary their adapters have not been audited for.
"""
from typing import Any, NamedTuple, Optional
from urllib.parse import urlsplit


class ResolvedAuxRoute(NamedTuple):
    client: Any
    final_model: Optional[str]
    resolved_provider: str
    effective_provider: str


def strict_provider_policy(task_cfg: dict) -> bool:
    policy = task_cfg.get("provider_policy", "legacy")
    if policy not in ("legacy", "strict"):
        raise ValueError("auxiliary provider_policy must be 'legacy' or 'strict'")
    return policy == "strict"


def strict_route_options(task_cfg, *, provider, model, base_url, api_key, api_mode):
    """Resolve only the explicitly selected route, before any legacy resolver.

    No main-runtime inference, key_cmd, credential pool, ambient API-key fallback,
    or provider discovery. An explicit key_env is the only environment lookup.
    """
    from agent import auxiliary_client as ac

    selected = (provider or task_cfg.get("provider") or "").strip()
    pinned_model = (model or task_cfg.get("model") or "").strip()
    if not selected or selected in ("auto", "main", "moa") or not pinned_model:
        raise ValueError("strict auxiliary routing requires an explicit provider and model")
    entry = {}
    if selected != "custom":
        from hermes_cli.runtime_provider import _get_named_custom_provider
        entry = _get_named_custom_provider(selected) or {}
        if not entry:
            raise ValueError("strict auxiliary routing requires custom or a named custom provider")
    endpoint = (base_url or task_cfg.get("base_url") or entry.get("base_url") or "").strip()
    parsed = urlsplit(endpoint)
    if parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("strict auxiliary routing requires an explicit HTTP(S) base_url without userinfo")
    mode = api_mode or task_cfg.get("api_mode") or entry.get("api_mode") or "chat_completions"
    if mode != "chat_completions":
        raise ValueError("strict auxiliary routing currently supports chat_completions only")
    key = api_key or task_cfg.get("api_key")
    if not key:
        key_env = task_cfg.get("api_key_env") or task_cfg.get("key_env")
        if key_env:
            key = ac._scoped_key_env(key_env)
        else:
            key = entry.get("api_key")
            key_env = entry.get("api_key_env") or entry.get("key_env")
            if not key and key_env:
                key = ac._scoped_key_env(key_env)
    if not isinstance(key, str) or not key.strip():
        raise ValueError("strict auxiliary routing requires a route-specific api_key or key_env")
    # Fully concrete route; no named-provider/default/model-prefix rediscovery.
    return "custom", pinned_model, endpoint, key, mode


def strict_cache_key(provider, model, base_url, api_key, api_mode, async_mode):
    """Never inspect a pool/main route while constructing a strict cache key."""
    import hashlib
    from agent import auxiliary_client as ac
    if provider != "custom" or not all((model, base_url, api_key)) or api_mode != "chat_completions":
        raise ValueError("strict client requires a complete concrete chat route")
    return (ac.hermes_home_key(), provider, "strict", model, base_url,
            hashlib.sha256(api_key.encode()).hexdigest(), async_mode)


def strict_client(model, base_url, api_key, async_mode):
    """No native adapters, endpoint inference, proxies or HTTP redirect escape."""
    import httpx
    from agent import auxiliary_client as ac
    if async_mode:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=api_key, base_url=base_url, max_retries=0,
                             http_client=httpx.AsyncClient(follow_redirects=False, trust_env=False))
    else:
        client = ac._create_openai_client(
            api_key=api_key, base_url=base_url,
            http_client=httpx.Client(follow_redirects=False, trust_env=False))
    return client, model


def resolve_call_client(
    task: Optional[str], *, provider: Optional[str], model: Optional[str], base_url: Optional[str],
    api_key: Optional[str], resolved_provider: str, resolved_model: Optional[str],
    resolved_base_url: Optional[str], resolved_api_key: Optional[str],
    resolved_api_mode: Optional[str], main_runtime: Optional[dict[str, Any]], async_mode: bool, strict_provider: bool = False,
) -> ResolvedAuxRoute:
    """Resolve the client for one aux call: vision chain, or cached text client with the
    explicit-provider fallback_chain / auto-chain rescue; RuntimeError when nothing is configured."""
    from agent import auxiliary_client as ac

    if strict_provider:
        client, final_model = ac._get_cached_client(
            resolved_provider, resolved_model, async_mode=async_mode, base_url=resolved_base_url,
            api_key=resolved_api_key, api_mode=resolved_api_mode, task=task, strict_provider=True)
        if client is None:
            raise ac.AuxiliaryClientUnavailable("Strict auxiliary provider is unavailable")
        return ResolvedAuxRoute(client, final_model, resolved_provider, resolved_provider)
    effective_provider = resolved_provider
    if task == "vision":
        effective_provider, client, final_model = ac.resolve_vision_provider_client(
            provider=resolved_provider if resolved_provider != "auto" else provider,
            model=resolved_model or model, base_url=resolved_base_url or base_url,
            api_key=resolved_api_key or api_key, async_mode=async_mode, main_runtime=main_runtime,
        )
        if client is None and resolved_provider != "auto" and not resolved_base_url:
            ac.logger.warning("Vision provider %s unavailable, falling back to auto vision backends",
                           resolved_provider)
            effective_provider, client, final_model = ac.resolve_vision_provider_client(
                provider="auto", model=resolved_model, async_mode=async_mode,
                main_runtime=main_runtime)
        if client is not None:
            resolved_provider = effective_provider or resolved_provider
    else:
        client, final_model = ac._get_cached_client(
            resolved_provider, resolved_model, async_mode=async_mode, base_url=resolved_base_url,
            api_key=resolved_api_key, api_mode=resolved_api_mode, main_runtime=main_runtime,
            task=task)
        effective_provider = ac._effective_provider_for_client(client, resolved_provider)
        if client is None:
            # Explicit provider with no credentials: honor the task fallback_chain before
            # raising (fallback entries may use OAuth / credential-pool auth).
            _explicit = (resolved_provider or "").strip().lower()
            if _explicit and _explicit not in {"auto", "openrouter", "custom"}:
                fb_client, fb_model, fb_label = ac._try_configured_fallback_for_unavailable_client(
                    task, _explicit)
                if fb_client is None:
                    nous_detail = ac.nous_credential_failure_detail() if _explicit == "nous" else None
                    raise ac.AuxiliaryClientUnavailable(
                        nous_detail or ac.missing_provider_credentials_message(_explicit))
                client, final_model = fb_client, fb_model
                if async_mode:
                    client, final_model = ac._to_async_client(
                        fb_client, fb_model or "", is_vision=(task == "vision"))
                resolved_provider = fb_label or resolved_provider
                effective_provider = resolved_provider
            # Auto/custom with no credentials: walk the full auto chain (not just OpenRouter).
            # model=None so each provider uses its own default.
            if client is None and not resolved_base_url:
                ac.logger.info("Auxiliary %s: provider %s unavailable, trying auto-detection chain",
                            task or "call", resolved_provider)
                client, final_model = ac._get_cached_client(
                    "auto", async_mode=async_mode, main_runtime=main_runtime, task=task)
                effective_provider = ac._effective_provider_for_client(client, "auto")
    if client is None:
        raise ac.AuxiliaryClientUnavailable(f"No LLM provider configured for task={task} "
                                         f"provider={resolved_provider}. Run: hermes setup")
    return ResolvedAuxRoute(client, final_model, resolved_provider, effective_provider)
