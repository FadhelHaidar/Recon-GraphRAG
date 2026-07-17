"""Single-pass placeholder substitution for prompt templates."""

from __future__ import annotations

import re

_PLACEHOLDER_RE = re.compile(r"\{\{|\}\}|\{(\w+)\}")


def format_template(template: str, **values) -> str:
    """Substitute ``{name}`` placeholders in a single pass.

    ``{{`` and ``}}`` produce literal braces. Unknown placeholders are left
    unchanged. Because substitution is single-pass, placeholder-like text
    inside the substituted values is never re-expanded.
    """

    def replace(match: re.Match) -> str:
        token = match.group(0)
        if token == "{{":
            return "{"
        if token == "}}":
            return "}"
        name = match.group(1)
        return str(values[name]) if name in values else token

    return _PLACEHOLDER_RE.sub(replace, template)


def has_placeholder(template: str, name: str) -> bool:
    """True if ``{name}`` appears unescaped in the template."""
    return any(match.group(1) == name for match in _PLACEHOLDER_RE.finditer(template))
