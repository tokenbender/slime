"""Swappable coding-agent harnesses (Claude Code, Codex, ...)."""

from __future__ import annotations

from .claude_code import ClaudeCodeHarness
from .codex import CodexHarness
from .common import BaseHarness, HarnessContext
from .local_script import LocalScriptHarness

__all__ = [
    "BaseHarness",
    "HarnessContext",
    "ClaudeCodeHarness",
    "CodexHarness",
    "LocalScriptHarness",
]
