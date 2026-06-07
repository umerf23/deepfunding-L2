"""GitHub REST API client used to enrich and back up deps.dev signals.

We use it for three things:
  - repo metadata (stars, forks, size, age) for normalization
  - the language byte breakdown, which measures first-party source footprint
  - a manifest fallback (counting declared deps) when deps.dev cannot resolve
    a graph, e.g. for repos that publish no package.

A token (env GITHUB_TOKEN) raises the rate limit from 60 to 5000 req/hour.
The client works without one but will be slow and may hit limits across 98 repos.
"""
from __future__ import annotations

import os
import time
from typing import Any

import requests

from ..utils.common import RepoRef, setup_logging

logger = setup_logging()

# Common dependency manifests and a regex-free heuristic for counting entries.
_MANIFEST_FILES = (
    "package.json",
    "Cargo.toml",
    "go.mod",
    "requirements.txt",
    "pyproject.toml",
    "pom.xml",
    "build.gradle",
)


class GitHubClient:
    """Cached, retrying GitHub API wrapper. Token is optional but recommended."""

    def __init__(self, config: dict[str, Any]) -> None:
        gh = config["github"]
        self.base_url: str = gh["base_url"].rstrip("/")
        self.timeout: int = gh["timeout_seconds"]
        self.max_retries: int = gh["max_retries"]
        self.backoff_base: float = gh["backoff_base_seconds"]
        self.sleep: float = config["runtime"]["request_sleep_seconds"]

        token = os.environ.get(gh["token_env_var"], "").strip()
        self.session = requests.Session()
        headers = {"Accept": "application/vnd.github+json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
            logger.info("GitHub client using authenticated token.")
        else:
            logger.warning(
                "No GITHUB_TOKEN set. Running unauthenticated (60 req/hour). "
                "Set GITHUB_TOKEN to avoid rate limits across 98 repos."
            )
        self.session.headers.update(headers)

    def _get(self, url_path: str, params: dict[str, Any] | None = None) -> Any:
        url = f"{self.base_url}{url_path}"
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.session.get(url, params=params, timeout=self.timeout)
                if response.status_code == 404:
                    return None
                if response.status_code in (403, 429):
                    reset = response.headers.get("X-RateLimit-Remaining", "?")
                    wait = self.backoff_base * (2 ** (attempt - 1))
                    logger.warning(
                        "GitHub throttled (remaining=%s), waiting %.1fs", reset, wait
                    )
                    time.sleep(wait)
                    continue
                response.raise_for_status()
                time.sleep(self.sleep)
                return response.json()
            except (requests.RequestException, ValueError) as exc:
                last_error = exc
                wait = self.backoff_base * (2 ** (attempt - 1))
                logger.warning(
                    "GitHub request failed (attempt %d/%d): %s; retry in %.1fs",
                    attempt, self.max_retries, exc, wait,
                )
                time.sleep(wait)
        logger.error("GitHub request permanently failed for %s: %s", url, last_error)
        return None

    def get_repo(self, repo: RepoRef) -> dict[str, Any] | None:
        """Core repo metadata: size (KB), stars, forks, created_at, etc."""
        return self._get(f"/repos/{repo.owner}/{repo.name}")

    def get_languages(self, repo: RepoRef) -> dict[str, int]:
        """Bytes of code per language. Empty dict if unavailable."""
        result = self._get(f"/repos/{repo.owner}/{repo.name}/languages")
        return result if isinstance(result, dict) else {}

    def count_manifest_dependencies(self, repo: RepoRef) -> int:
        """Best-effort count of declared direct dependencies from manifests.

        Used only as a fallback when deps.dev cannot resolve a graph. Reads the
        repo's default-branch tree, finds known manifest files, and counts
        dependency-like lines. Deliberately conservative: it returns 0 rather
        than guessing when it cannot read a manifest.
        """
        meta = self.get_repo(repo)
        if not meta:
            return 0
        default_branch = meta.get("default_branch", "main")
        tree = self._get(
            f"/repos/{repo.owner}/{repo.name}/git/trees/{default_branch}",
            params={"recursive": "1"},
        )
        if not tree or "tree" not in tree:
            return 0

        total = 0
        for node in tree["tree"]:
            path = node.get("path", "")
            filename = path.rsplit("/", 1)[-1]
            if filename in _MANIFEST_FILES and node.get("type") == "blob":
                total += self._count_in_manifest(repo, path, filename)
        return total

    def _count_in_manifest(self, repo: RepoRef, path: str, filename: str) -> int:
        """Fetch one manifest and count dependency entries heuristically."""
        raw = self._get(f"/repos/{repo.owner}/{repo.name}/contents/{path}")
        if not isinstance(raw, dict) or "content" not in raw:
            return 0
        import base64

        try:
            text = base64.b64decode(raw["content"]).decode("utf-8", errors="ignore")
        except (ValueError, TypeError):
            return 0

        if filename == "go.mod":
            return sum(1 for line in text.splitlines() if line.strip().startswith("require"))
        if filename == "requirements.txt":
            return sum(
                1
                for line in text.splitlines()
                if line.strip() and not line.strip().startswith("#")
            )
        if filename in ("package.json", "pyproject.toml", "Cargo.toml"):
            # Count lines that look like "name = version" or "name": "version".
            return sum(
                1
                for line in text.splitlines()
                if ("=" in line or ":" in line) and not line.strip().startswith(("#", "//"))
            ) // 2  # rough deflation for non-dependency config lines
        return 0
