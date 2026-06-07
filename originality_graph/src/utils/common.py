"""Shared utilities: configuration loading, logging setup, and repo parsing."""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

_LOGGER_CONFIGURED = False


def load_config(path: str | Path = "configs/config.yaml") -> dict[str, Any]:
    """Load the YAML config into a plain dict.

    Raises FileNotFoundError with a clear message if the file is missing,
    so a misconfigured run fails fast instead of deep inside the pipeline.
    """
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(
            f"Config file not found at '{config_path.resolve()}'. "
            "Run from the project root or pass an explicit path."
        )
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Config root must be a mapping of sections to values.")
    return config


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Configure root logging once and return a named logger."""
    global _LOGGER_CONFIGURED
    if not _LOGGER_CONFIGURED:
        logging.basicConfig(
            level=getattr(logging, level.upper(), logging.INFO),
            format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        _LOGGER_CONFIGURED = True
    return logging.getLogger("originality")


@dataclass(frozen=True)
class RepoRef:
    """A parsed GitHub repository reference."""

    owner: str
    name: str

    @property
    def slug(self) -> str:
        """The canonical lowercase owner/name identifier."""
        return f"{self.owner}/{self.name}".lower()

    @property
    def project_id(self) -> str:
        """The deps.dev project id form: github.com/owner/name."""
        return f"github.com/{self.owner}/{self.name}"


_URL_PATTERN = re.compile(
    r"github\.com[/:]"
    r"(?P<owner>[A-Za-z0-9_.-]+)/"
    r"(?P<name>[A-Za-z0-9_.-]+?)"
    r"(?:\.git)?/?$"
)


def parse_repo(url: str) -> RepoRef:
    """Parse a GitHub URL or owner/name slug into a RepoRef.

    Accepts full https URLs, git URLs, and bare owner/name strings.
    Raises ValueError on anything that cannot be resolved to owner/name.
    """
    cleaned = url.strip()
    if not cleaned:
        raise ValueError("Empty repository reference.")

    match = _URL_PATTERN.search(cleaned)
    if match:
        return RepoRef(owner=match.group("owner"), name=match.group("name"))

    # Fall back to a bare "owner/name" slug.
    parts = cleaned.strip("/").split("/")
    if len(parts) == 2 and all(parts):
        return RepoRef(owner=parts[0], name=parts[1])

    raise ValueError(f"Could not parse a GitHub owner/name from: '{url}'")
