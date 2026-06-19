"""Shared helpers for opt-in integration tests."""

from __future__ import annotations

import os

import pytest
from dotenv import load_dotenv


ENABLED_VALUES = {"1", "true", "yes"}
PROVIDER_REQUIREMENTS = {
    "azure_openai": {
        "llm": [
            "AZURE_OPENAI_ENDPOINT",
            "AZURE_OPENAI_API_KEY",
            "AZURE_OPENAI_LLM_DEPLOYMENT_NAME",
        ],
        "embedder": [
            "AZURE_OPENAI_ENDPOINT",
            "AZURE_OPENAI_API_KEY",
            "AZURE_OPENAI_EMBED_MODEL_DEPLOYMENT_NAME",
        ],
    },
    "openrouter": {
        "llm": ["OPENROUTER_API_KEY", "OPENROUTER_LLM_MODEL"],
        "embedder": ["OPENROUTER_API_KEY", "OPENROUTER_EMBED_MODEL"],
    },
    "openai": {
        "llm": ["OPENAI_API_KEY", "OPENAI_LLM_MODEL"],
        "embedder": ["OPENAI_API_KEY", "OPENAI_EMBED_MODEL"],
    },
    "sentence-transformer": {
        "embedder": [],
    },
}


def require_integration_env(
    run_flag: str,
    required_env: list[str],
    description: str,
    *,
    fail_on_missing: bool = False,
) -> None:
    """Skip disabled integrations and validate their required environment."""
    load_dotenv()

    if os.getenv(run_flag, "").lower() not in ENABLED_VALUES:
        pytest.skip(f"Set {run_flag}=1 to run {description}.")

    missing = [name for name in required_env if not os.getenv(name)]
    if not missing:
        return

    message = f"Missing required {description} env vars: {', '.join(missing)}"
    if fail_on_missing:
        pytest.fail(message)
    pytest.skip(message)


def require_selected_provider_env(
    description: str,
    *,
    fail_on_missing: bool = True,
) -> tuple[str, str]:
    """Validate the explicitly selected LLM and embedder provider settings."""
    load_dotenv()
    llm_provider = os.getenv("LLM_PROVIDER", "azure_openai").lower()
    embedder_provider = os.getenv("EMBEDDER_PROVIDER", "azure_openai").lower()

    selections = (("llm", llm_provider), ("embedder", embedder_provider))
    required: set[str] = set()
    for role, provider in selections:
        provider_requirements = PROVIDER_REQUIREMENTS.get(provider)
        if provider_requirements is None or role not in provider_requirements:
            pytest.fail(f"Unsupported {role} provider for {description}: {provider}")
        required.update(provider_requirements[role])

    missing = sorted(name for name in required if not os.getenv(name))
    if missing:
        message = f"Missing required {description} env vars: {', '.join(missing)}"
        if fail_on_missing:
            pytest.fail(message)
        pytest.skip(message)

    return llm_provider, embedder_provider
