"""Client for the deps.dev v3 REST API (Google Open Source Insights).

deps.dev is free and requires no authentication. We use three endpoints:
  - GetProject              : repo-level metadata (stars, forks, license)
  - GetProjectPackageVersions: maps a repo to its published package versions
  - GetDependencies         : resolved (direct + transitive) dependency graph

The resolved graph is the core originality signal: a large transitive graph
relative to the repo's own footprint means heavy reliance on dependencies.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import requests

from ..utils.common import RepoRef, setup_logging

logger = setup_logging()


class DepsDevClient:
    """Thin, cached, retrying wrapper over the deps.dev v3 API."""

    def __init__(self, config: dict[str, Any]) -> None:
        dd = config["depsdev"]
        self.base_url: str = dd["base_url"].rstrip("/")
        self.timeout: int = dd["timeout_seconds"]
        self.max_retries: int = dd["max_retries"]
        self.backoff_base: float = dd["backoff_base_seconds"]
        self.resolvable_systems: set[str] = {s.upper() for s in dd["resolvable_systems"]}
        self.sleep: float = config["runtime"]["request_sleep_seconds"]

        cache_root = Path(config["paths"]["cache_dir"]) / "depsdev"
        cache_root.mkdir(parents=True, exist_ok=True)
        self.cache_root = cache_root

        self.session = requests.Session()
        self.session.headers.update({"Accept": "application/json"})

    # -- low level ---------------------------------------------------

    def _cache_path(self, key: str) -> Path:
        safe = key.replace("/", "_").replace(":", "_")
        return self.cache_root / f"{safe}.json"

    def _get(self, url_path: str, cache_key: str) -> dict[str, Any] | None:
        """GET a deps.dev path with caching and exponential backoff.

        Returns parsed JSON, or None for a clean 404 (resource not known to
        deps.dev), which is a normal and expected outcome for some repos.
        """
        cache_file = self._cache_path(cache_key)
        if cache_file.is_file():
            with cache_file.open("r", encoding="utf-8") as handle:
                return json.load(handle)

        url = f"{self.base_url}{url_path}"
        last_error: Exception | None = None

        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.session.get(url, timeout=self.timeout)
                if response.status_code == 404:
                    cache_file.write_text("null", encoding="utf-8")
                    return None
                if response.status_code == 429:
                    wait = self.backoff_base * (2 ** (attempt - 1))
                    logger.warning("deps.dev rate limited, waiting %.1fs", wait)
                    time.sleep(wait)
                    continue
                response.raise_for_status()
                payload = response.json()
                cache_file.write_text(json.dumps(payload), encoding="utf-8")
                time.sleep(self.sleep)
                return payload
            except (requests.RequestException, ValueError) as exc:
                last_error = exc
                wait = self.backoff_base * (2 ** (attempt - 1))
                logger.warning(
                    "deps.dev request failed (attempt %d/%d): %s; retrying in %.1fs",
                    attempt, self.max_retries, exc, wait,
                )
                time.sleep(wait)

        logger.error("deps.dev request permanently failed for %s: %s", url, last_error)
        return None

    # -- public endpoints -------------------------------------------

    def get_project(self, repo: RepoRef) -> dict[str, Any] | None:
        """Repo-level metadata. Returns None if deps.dev has no record."""
        path = f"/projects/{requests.utils.quote(repo.project_id, safe='')}"
        return self._get(path, cache_key=f"project_{repo.slug}")

    def get_package_versions(self, repo: RepoRef) -> list[dict[str, Any]]:
        """Published package versions linked to this repo.

        Each entry has versionKey.system / .name / .version. We later pick the
        most recent default version per package to resolve a dependency graph.
        """
        encoded = requests.utils.quote(repo.project_id, safe="")
        path = f"/projects/{encoded}:packageversions"
        payload = self._get(path, cache_key=f"pkgversions_{repo.slug}")
        if not payload:
            return []
        return payload.get("versions", [])

    def get_dependent_count(self, system: str, name: str, version: str) -> int:
        """Number of distinct public packages known to depend on this version.

        Uses the v3alpha :dependents endpoint. This is an authority signal: a
        package many others depend on is foundational (high originality). deps.dev
        documents these counts as indicative of relative popularity, not exact,
        which is fine since the contest is a ranking.
        """
        if system.upper() not in self.resolvable_systems:
            return 0
        enc_name = requests.utils.quote(name, safe="")
        enc_ver = requests.utils.quote(version, safe="")
        # dependents lives in v3alpha; swap the version segment of the base url.
        alpha_base = self.base_url.replace("/v3", "/v3alpha")
        path = (
            f"/systems/{system.upper()}/packages/{enc_name}"
            f"/versions/{enc_ver}:dependents"
        )
        cache_file = self._cache_path(f"dependents_{system}_{name}_{version}")
        if cache_file.is_file():
            with cache_file.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        else:
            url = f"{alpha_base}{path}"
            try:
                response = self.session.get(url, timeout=self.timeout)
                if response.status_code == 404:
                    cache_file.write_text("null", encoding="utf-8")
                    return 0
                response.raise_for_status()
                payload = response.json()
                cache_file.write_text(json.dumps(payload), encoding="utf-8")
                time.sleep(self.sleep)
            except (requests.RequestException, ValueError) as exc:
                logger.warning("deps.dev dependents failed for %s: %s", name, exc)
                return 0
        if not payload:
            return 0
        # Response exposes direct/indirect/total dependent counts.
        return int(payload.get("dependentCount", payload.get("directDependentCount", 0)) or 0)

    def get_dependencies(self, system: str, name: str, version: str) -> dict[str, Any] | None:
        """Resolved dependency graph for one package version.

        Only npm, Cargo, Maven, and PyPI are resolvable; other systems return
        None and the caller falls back to GitHub-derived signals.
        """
        if system.upper() not in self.resolvable_systems:
            return None
        enc_name = requests.utils.quote(name, safe="")
        enc_ver = requests.utils.quote(version, safe="")
        path = (
            f"/systems/{system.upper()}/packages/{enc_name}"
            f"/versions/{enc_ver}:dependencies"
        )
        return self._get(path, cache_key=f"deps_{system}_{name}_{version}")
