"""Ecosystem dependency graph builder.

This is what makes Solution 2 fundamentally different from Solution 1. Rather
than scoring each repo in isolation from hand-crafted features, we build ONE
directed graph spanning all 98 grant repos plus every package in their resolved
dependency trees, then measure each repo's structural position in that network.

Edge direction: A -> B means "A depends on B". So:
  - out-edges  = things this repo relies on   (more  -> less original)
  - in-edges   = things that rely on this repo (more  -> more original)

A repo that is depended upon widely but depends on little is a "source" node:
foundational, high originality. A repo that pulls from many others but is used
by few is a "sink": derivative, low originality. Centrality measures formalize
exactly this source-vs-sink balance across the whole ecosystem at once.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import networkx as nx

from ..ingestion.depsdev_client import DepsDevClient
from ..utils.common import RepoRef, parse_repo, setup_logging

logger = setup_logging()


@dataclass
class GraphBuildResult:
    """The built graph plus the mapping from repo URL to its package node id."""

    graph: nx.DiGraph
    repo_to_node: dict[str, str] = field(default_factory=dict)
    repo_dependent_counts: dict[str, int] = field(default_factory=dict)
    unresolved_repos: list[str] = field(default_factory=list)


class EcosystemGraphBuilder:
    """Builds a directed dependency graph for the full grant cohort."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.depsdev = DepsDevClient(config)
        self.max_nodes: int = config["features"]["max_transitive_nodes"]

    def build(self, repo_urls: list[str]) -> GraphBuildResult:
        graph = nx.DiGraph()
        repo_to_node: dict[str, str] = {}
        dependent_counts: dict[str, int] = {}
        unresolved: list[str] = []

        for url in repo_urls:
            try:
                node_id = self._add_repo(graph, url, dependent_counts)
            except Exception as exc:  # noqa: BLE001 - one repo cannot abort the build
                logger.error("Graph add failed for %s: %s", url, exc)
                node_id = None
            if node_id is None:
                unresolved.append(url.strip())
                # Still register the repo as an isolated node so it gets a score.
                placeholder = f"repo::{url.strip()}"
                graph.add_node(placeholder, is_repo=True, repo_url=url.strip(), resolved=False)
                repo_to_node[url.strip()] = placeholder
            else:
                repo_to_node[url.strip()] = node_id

        logger.info(
            "Built ecosystem graph: %d nodes, %d edges, %d repos resolved, %d unresolved",
            graph.number_of_nodes(), graph.number_of_edges(),
            len(repo_urls) - len(unresolved), len(unresolved),
        )
        return GraphBuildResult(
            graph=graph,
            repo_to_node=repo_to_node,
            repo_dependent_counts=dependent_counts,
            unresolved_repos=unresolved,
        )

    def _add_repo(
        self, graph: nx.DiGraph, url: str, dependent_counts: dict[str, int]
    ) -> str | None:
        """Resolve a repo's package, add its dependency subgraph, return its node id."""
        repo = parse_repo(url)
        versions = self.depsdev.get_package_versions(repo)
        if not versions:
            return None

        version_key = self._select_version(versions)
        if version_key is None:
            return None

        system, name, version = version_key["system"], version_key["name"], version_key["version"]
        node_id = self._node_id(system, name)

        dependent_counts[url.strip()] = self.depsdev.get_dependent_count(system, name, version)

        graph_payload = self.depsdev.get_dependencies(system, name, version)
        if not graph_payload or "nodes" not in graph_payload:
            graph.add_node(node_id, is_repo=True, repo_url=url.strip(), resolved=False)
            return node_id

        self._merge_dependency_graph(graph, graph_payload, root_node_id=node_id, repo_url=url.strip())
        return node_id

    def _merge_dependency_graph(
        self, graph: nx.DiGraph, payload: dict[str, Any], root_node_id: str, repo_url: str
    ) -> None:
        """Merge one resolved dependency graph into the shared ecosystem graph.

        Shared packages (e.g. two repos both depending on lodash) become the same
        node, which is precisely how the cross-repo network forms.
        """
        nodes = payload.get("nodes", [])
        edges = payload.get("edges", [])
        if len(nodes) > self.max_nodes:
            nodes = nodes[: self.max_nodes]

        index_to_id: dict[int, str] = {}
        for idx, node in enumerate(nodes):
            key = node.get("versionKey", {})
            system = key.get("system", "UNKNOWN")
            name = key.get("name", f"unknown_{idx}")
            node_key = self._node_id(system, name) if idx != 0 else root_node_id
            index_to_id[idx] = node_key
            if node_key not in graph:
                graph.add_node(node_key, is_repo=(idx == 0), resolved=True)
            if idx == 0:
                graph.nodes[node_key]["is_repo"] = True
                graph.nodes[node_key]["repo_url"] = repo_url
                graph.nodes[node_key]["resolved"] = True

        for edge in edges:
            src = edge.get("fromNode")
            dst = edge.get("toNode")
            if src in index_to_id and dst in index_to_id:
                # A -> B means A depends on B.
                graph.add_edge(index_to_id[src], index_to_id[dst])

    @staticmethod
    def _select_version(versions: list[dict[str, Any]]) -> dict[str, str] | None:
        chosen = next((v for v in versions if v.get("isDefault")), versions[0])
        key = chosen.get("versionKey", {})
        if not all(k in key for k in ("system", "name", "version")):
            return None
        return {"system": key["system"], "name": key["name"], "version": key["version"]}

    @staticmethod
    def _node_id(system: str, name: str) -> str:
        return f"{system.upper()}::{name}"
