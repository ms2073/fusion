"""Configuration: typed YAML for routing policy, env for secrets and binding."""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

log = logging.getLogger("fusion.config")

# Every reasonable way of writing "here are my keys" is accepted, because the
# whole point of more keys is more rate-limit headroom - the format should never
# be the reason one goes unused.
_INDEXED_KEY = re.compile(r"^NIM_API_KEY_(\d+)$")  # NIM_API_KEY_1, _2, _3, ...
_PLAIN_KEY = "NIM_API_KEY"                          # a single key
_KEY_LIST = "NIM_API_KEYS"                          # comma/space separated
_KEY_FILE = "NIM_API_KEYS_FILE"                     # one per line, for secrets

_SEPARATORS = re.compile(r"[,;\s]+")
_PLACEHOLDERS = frozenset({"changeme", "your-key-here", "replace-me", "none", "unset"})


class UpstreamConfig(BaseModel):
    base_url: str = "https://integrate.api.nvidia.com/v1"
    timeout_seconds: float = 600.0
    connect_timeout_seconds: float = 15.0
    # Max silence between bytes, not total duration - a long answer streams
    # fine, but a model that accepts the request and then stalls gives up
    # quickly so we can fail over instead of hanging for the full timeout.
    stall_timeout_seconds: float = 30.0
    #: Extra silence allowed per 100k tokens of prompt. Time-to-first-token
    #: grows with the payload - the model reads it all before saying anything -
    #: so a flat ceiling condemns every large request as stalled.
    stall_timeout_per_100k_tokens: float = 45.0
    #: Silence allowed *after* the model has started answering. Deliberately far
    #: looser than the stall budget: before the first token, abandoning a model
    #: is free, but once the client holds bytes there is no failover left, so
    #: killing a model that merely paused to think destroys a real answer.
    idle_timeout_seconds: float = 300.0


class RateLimitConfig(BaseModel):
    default_rpm: int = 36
    cooldown_base_seconds: float = 5.0
    cooldown_max_seconds: float = 120.0
    # A model queued behind other users stays queued for a while, so park it
    # for longer than the 429 base. Without this, every request re-pays the
    # full stall timeout rediscovering the same busy model.
    stall_cooldown_seconds: float = 20.0
    # A 404 is conclusive about a model, so it is parked far longer than a
    # transient failure - but not forever, since entitlements do change.
    unavailable_seconds: float = 900.0
    breaker_failure_threshold: int = 5
    breaker_reset_seconds: float = 60.0


class HedgeConfig(BaseModel):
    """Race several models rather than waiting on one.

    A queued NIM model answers with silence, and which models are queued flips
    within seconds. Sequential failover therefore pays the stall timeout again
    and again; racing pays only the fastest responder.
    """

    enabled: bool = True
    # How long the leader gets alone before a second model is also tried. Long
    # enough that a healthy primary costs exactly one request.
    delay_seconds: float = 2.5
    max_parallel: int = 3


class RoutingConfig(BaseModel):
    default_role: str = "thinker"
    long_context_tokens: int = 200_000
    chain_enabled: bool = True
    chain_plan_max_tokens: int = 900
    verifier_enabled: bool = False
    ambiguous_escalation: bool = True


class ClassifierConfig(BaseModel):
    # `model` would collide with pydantic's protected namespace otherwise.
    model_config = ConfigDict(protected_namespaces=())

    model: str = "nvidia/llama-3.3-nemotron-super-49b-v1.5"
    max_tokens: int = 4
    timeout_seconds: float = 8.0


class SessionConfig(BaseModel):
    ttl_seconds: float = 3600.0
    max_entries: int = 512


class RoleConfig(BaseModel):
    """An ordered preference list. Index 0 is primary, the rest are failover."""

    models: list[str] = Field(default_factory=list)


class ModelConfig(BaseModel):
    context: int = 128_000
    vision: bool = False
    rpm: int | None = None


class FusionConfig(BaseModel):
    upstream: UpstreamConfig = Field(default_factory=UpstreamConfig)
    rate_limit: RateLimitConfig = Field(default_factory=RateLimitConfig)
    hedge: HedgeConfig = Field(default_factory=HedgeConfig)
    routing: RoutingConfig = Field(default_factory=RoutingConfig)
    classifier: ClassifierConfig = Field(default_factory=ClassifierConfig)
    session: SessionConfig = Field(default_factory=SessionConfig)
    roles: dict[str, RoleConfig] = Field(default_factory=dict)
    models: dict[str, ModelConfig] = Field(default_factory=dict)
    lanes: dict[str, str | None] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_references(self) -> FusionConfig:
        """Catch config typos at load time rather than at request time."""
        if self.routing.default_role not in self.roles:
            raise ValueError(
                f"routing.default_role {self.routing.default_role!r} is not a "
                f"defined role (have: {sorted(self.roles)})"
            )
        for role, cfg in self.roles.items():
            if not cfg.models:
                raise ValueError(f"role {role!r} has an empty model list")
        for lane, role in self.lanes.items():
            if role is not None and role not in self.roles:
                raise ValueError(
                    f"lane {lane!r} maps to undefined role {role!r} "
                    f"(have: {sorted(self.roles)})"
                )
        return self

    def models_for_role(self, role: str) -> list[str]:
        """Preference-ordered candidates, primary first."""
        cfg = self.roles.get(role)
        if cfg is None:
            raise KeyError(f"unknown role {role!r}")
        return list(cfg.models)

    def primary_for_role(self, role: str) -> str:
        return self.models_for_role(role)[0]

    def rpm_for(self, model: str) -> int:
        override = self.models.get(model, ModelConfig()).rpm
        return override if override is not None else self.rate_limit.default_rpm

    def supports_vision(self, model: str) -> bool:
        return self.models.get(model, ModelConfig()).vision

    def context_for(self, model: str) -> int:
        return self.models.get(model, ModelConfig()).context

    def all_models(self) -> list[str]:
        """Every model this config could ever dispatch to, deduped, stable order."""
        seen: dict[str, None] = {}
        for cfg in self.roles.values():
            for name in cfg.models:
                seen.setdefault(name, None)
        seen.setdefault(self.classifier.model, None)
        return list(seen)

    def vision_models(self) -> list[str]:
        return [m for m in self.all_models() if self.supports_vision(m)]


class Settings(BaseSettings):
    """Env-supplied: where to listen, how loud to log, where policy lives."""

    model_config = SettingsConfigDict(
        env_prefix="FUSION_", env_file=".env", extra="ignore"
    )

    host: str = "0.0.0.0"
    port: int = 8787
    log_level: str = "INFO"
    config_path: Path = Path("config.yaml")


@dataclass(frozen=True, slots=True)
class ApiKey:
    value: str
    origin: str

    @property
    def masked(self) -> str:
        return mask_key(self.value)


def mask_key(value: str) -> str:
    """Enough to tell two keys apart, never enough to use one."""
    if len(value) <= 14:
        return "***"
    return f"{value[:10]}...{value[-4:]}"


def _clean(raw: str) -> str:
    return raw.strip().strip("\"'").strip()


def _is_placeholder(value: str) -> bool:
    lowered = value.lower()
    return not value or "..." in value or lowered in _PLACEHOLDERS


def _split(raw: str) -> list[str]:
    return [part for part in _SEPARATORS.split(raw.strip()) if part]


def load_api_key_entries(env: dict[str, str] | None = None) -> list[ApiKey]:
    """Collect API keys from every supported form, in a stable order.

    Accepted, and freely mixable:

        NIM_API_KEY=nvapi-a                  one key
        NIM_API_KEY_1=nvapi-a                numbered, any count
        NIM_API_KEY_2=nvapi-b
        NIM_API_KEYS=nvapi-a,nvapi-b         a list in one variable
        NIM_API_KEYS_FILE=/run/secrets/nim   one per line, for docker secrets

    Duplicates collapse and placeholders are dropped, so a half-filled .env
    degrades to however many real keys it holds rather than failing requests
    against a bogus one.
    """
    source = dict(os.environ if env is None else env)
    found: list[ApiKey] = []

    def add(raw: str, origin: str) -> None:
        for candidate in _split(raw):
            cleaned = _clean(candidate)
            if not _is_placeholder(cleaned):
                found.append(ApiKey(cleaned, origin))

    if plain := source.get(_PLAIN_KEY):
        add(plain, _PLAIN_KEY)

    indexed: list[tuple[int, str, str]] = []
    for name, value in source.items():
        if (match := _INDEXED_KEY.match(name)) and value:
            indexed.append((int(match.group(1)), name, value))
    for _, name, value in sorted(indexed):
        add(value, name)

    if listed := source.get(_KEY_LIST):
        add(listed, _KEY_LIST)

    if path := source.get(_KEY_FILE):
        try:
            add(Path(_clean(path)).read_text(), f"{_KEY_FILE}:{path}")
        except OSError as exc:
            log.warning("could not read %s=%s: %s", _KEY_FILE, path, exc)

    unique: dict[str, ApiKey] = {}
    for entry in found:
        if entry.value in unique:
            log.debug("ignoring duplicate key from %s", entry.origin)
            continue
        unique[entry.value] = entry
        if not entry.value.startswith("nvapi-"):
            log.warning(
                "key from %s does not look like a NIM key (expected an "
                "'nvapi-' prefix) - it will still be tried",
                entry.origin,
            )
    return list(unique.values())


def load_api_keys(env: dict[str, str] | None = None) -> list[str]:
    return [entry.value for entry in load_api_key_entries(env)]


def load_config(path: Path | str) -> FusionConfig:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    return FusionConfig.model_validate(raw)
