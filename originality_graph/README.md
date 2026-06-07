# Solution 2 - Graph Centrality Originality

Gitcoin Grants Round 24, Level II. Produces an originality score between 0 and 1
for each of the 98 repos. Higher means more original (relies less on dependencies,
more foundational to the ecosystem).

## How this differs from Solution 1

Solution 1 scored each repo in isolation from hand-crafted features. Solution 2
takes a fundamentally different view: it builds ONE directed dependency graph
spanning all 98 grant repos plus every package in their resolved dependency
trees, then measures each repo's structural position in that network.

Edge direction is `A -> B` meaning "A depends on B". The intuition:

- A repo that many others depend on, but which itself depends on little, is a
  **source** node: foundational, high originality.
- A repo that pulls from many others but is used by few is a **sink**: derivative,
  low originality.

We formalize this with network-science measures: PageRank on the reversed graph
(authority of being depended upon), HITS authority, in-degree vs out-degree, and
deps.dev dependent counts (external popularity). Each signal is rank-normalized
across repos, combined into a source-vs-sink balance, and mapped to (0, 1).

This captures something Solution 1 structurally cannot: a repo's position in the
ecosystem relative to the others. Shared dependencies naturally link repos in the
graph, so the model sees the whole web of relationships at once.

## Why this scores well even without a GitHub token

The graph is built entirely from the deps.dev API, which is free and needs no
authentication. Because the signal comes from relationships between repos rather
than each repo's isolated features, it produces a strongly differentiated ranking
across the full 0-to-1 range on deps.dev data alone. (A live 12-repo test spread
scores from 0.01 to 0.99.)

## Setup and run

```bash
pip install -r requirements.txt
make run        # build graph -> score -> artifacts/originality-predictions.csv
make test       # offline unit tests (graphs built by hand, no network)
make serve      # FastAPI service on http://localhost:8000
```

Outputs:
- `artifacts/originality-predictions.csv` - the submission
- `artifacts/ecosystem_graph.gexf` - the full graph, openable in Gephi for inspection
- `data/processed/centrality_features.csv` - per-repo centrality features

## API

Because originality here is cohort-relative (it depends on the whole graph), the
API serves precomputed scores and graph diagnostics rather than scoring new repos
in isolation.

```bash
curl http://localhost:8000/scores                 # full ranked table
curl http://localhost:8000/scores/wevm/viem        # one repo
```

## Optional LightGBM ranking head

Set `model.use_lgbm_rank: true` and `model.lgbm_blend_weight` in the config to
blend a LightGBM regressor (fit on the sample labels) with the centrality score.
It is OFF by default because the sample labels look synthetic, but the path is
fully implemented and cross-validated.

## Honest limitations

- deps.dev resolves graphs for npm, Cargo, Maven, and PyPI only. Repos with no
  published package in those ecosystems (some Solidity, Go, Nim repos) cannot
  join the graph and are scored as isolated nodes, landing near the low end. In
  the 12-repo test, 3 of 12 were unresolved. This is the main accuracy gap and
  is worth discussing in the writeup.
- Dependent counts from deps.dev are documented as indicative of relative
  popularity, not exact, which is acceptable for a ranking task.
- The full 98-repo graph build makes one or more deps.dev calls per repo, so the
  first run takes a few minutes. Responses are cached on disk, so reruns are fast.
