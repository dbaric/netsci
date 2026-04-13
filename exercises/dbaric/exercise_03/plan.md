# Exercise 03 — Wikipedia Vote Network

## Context

Exercise 02 (`origin/excerise-02-dbaric`) completed basic graph description of the SNAP Wikipedia Vote Network. Exercise 03 (`excerise-03-dbaric` branch) extends that work with Lecture 03 metrics: centrality measures, structural metrics, and centrality-based visualization. Student 6 specific focus: compare top in-degree, top out-degree, and top PageRank nodes to show how direction changes importance.

Known graph facts from exercise-02 (used in interpretation cells):

- 7115 nodes, 103689 edges, density 0.002049
- Node 4037: max in-degree (457), Node 2565: max out-degree (893)
- 24 WCCs (largest: 7066 nodes), 5816 SCCs (largest: 1300 nodes)

---

## Branch & Directory

- New branch: `excerise-03-dbaric` from `main`
- Directory: `exercises/excerise-03/dbaric/`

---

## Files to Create

| Path                                              | Source                                |
| ------------------------------------------------- | ------------------------------------- |
| `exercises/excerise-03/dbaric/notebook.ipynb`     | New (mirroring ex-02 style)           |
| `exercises/excerise-03/dbaric/data/wiki-Vote.txt` | Copy from `origin/excerise-02-dbaric` |
| `exercises/excerise-03/dbaric/.gitignore`         | Same as ex-02: `lib/`                 |

No marimo version needed (dbaric's exercise-02 had none).

---

## Notebook Cell Structure

Follows exercise-02 exactly: markdown headers + `print(f"""...""")` style output tables.

### Cell 0 — markdown: Title

```
# Exercise 03 — Wikipedia Vote Network
Centrality and structural analysis of the Wikipedia admin-election vote network.
```

### Cell 1 — code: Imports

```python
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
from pathlib import Path
from pyvis.network import Network
```

### Cell 2 — markdown: Load Data

Reuse ex-02 Cell 2 verbatim — same dataset, same citation:

> **Citation:** J. Leskovec, D. Huttenlocher, and J. Kleinberg. Signed networks in social media. CHI 2010.

### Cell 3 — code: Load Graph

Reuse ex-02 Cell 3 verbatim — read `data/wiki-Vote.txt`, skip `#` lines, parse tab-separated integers, build `nx.DiGraph`. Print node/edge count confirmation.

### Cell 4 — markdown: `## Degree-Based Measures`

### Cell 5 — code: Degree Stats

Reuse ex-02 Cell 5 verbatim — in/out degree stats table, top-5 by each. This satisfies "compute degree-based measures, report in/out separately".

### Cell 6 — markdown: `## Centrality Measures`

### Cell 7 — code: Compute 5 Centralities

```python
# Degree centrality (normalized)
in_deg_cent  = nx.in_degree_centrality(G)
out_deg_cent = nx.out_degree_centrality(G)

# Betweenness — sampled (k=500) for performance on 7k-node graph
betweenness = nx.betweenness_centrality(G, k=500, seed=42)

# PageRank
pagerank = nx.pagerank(G, alpha=0.85)

# Eigenvector centrality (numpy version handles convergence on directed graphs)
try:
    eigenvector = nx.eigenvector_centrality_numpy(G)
except Exception:
    eigenvector = {}

# Closeness — computed on largest WCC undirected projection
largest_wcc_nodes = max(nx.weakly_connected_components(G), key=len)
G_wcc = G.subgraph(largest_wcc_nodes)
closeness = nx.closeness_centrality(G_wcc.to_undirected())
```

Print top-5 for each measure using `print(f"""...""")` tables matching ex-02 style.

### Cell 8 — markdown: `## Structural Metrics`

### Cell 9 — code: Density, Clustering, Path Metric

```python
density = nx.density(G)                               # already known: 0.002049
avg_clustering = nx.average_clustering(G.to_undirected())

# Path metric on largest SCC (state choice clearly)
largest_scc_nodes = max(nx.strongly_connected_components(G), key=len)
G_scc = G.subgraph(largest_scc_nodes).copy()
# Use eccentricity on a sample of 100 nodes (diameter too slow on 1300-node SCC)
sample_nodes = list(G_scc.nodes())[:100]
ecc = nx.eccentricity(G_scc, v=sample_nodes)
approx_diameter = max(ecc.values())
```

Print table with density, avg_clustering, approx_diameter (with note: "computed on largest SCC of 1300 nodes, eccentricity sampled from 100 nodes").

### Cell 10 — markdown: `## Centrality Visualization`

### Cell 11 — code: PageRank-Colored Ego Network (PyVis)

Build subgraph of top-100 PageRank nodes + their mutual edges. Use PyVis (matching ex-02 tool choice):

- Node size ∝ PageRank (scaled 10–60px)
- Node color: red for top PageRank node, gradient lightblue→orange by PageRank rank
- `cdn_resources='in_line'` to avoid Jupyter warning seen in ex-02
- Save as `centrality_viz.html`
- Print: `"Centrality visualization saved as centrality_viz.html"`

### Cell 12 — markdown: `## Comparison: In-Degree vs Out-Degree vs PageRank`

Explain Student 6 focus: direction changes importance — in-degree = received trust, out-degree = active voter, PageRank = authority propagated through chains.

### Cell 13 — code: Build Comparison Table

```python
top_n = 5
in_top  = sorted(in_deg_cent,  key=in_deg_cent.get,  reverse=True)[:top_n]
out_top = sorted(out_deg_cent, key=out_deg_cent.get, reverse=True)[:top_n]
pr_top  = sorted(pagerank,     key=pagerank.get,      reverse=True)[:top_n]

print("## Top 5 Nodes by Metric\n")
print(f"| Rank | In-Degree | Out-Degree | PageRank |")
print(f"|------|-----------|------------|----------|")
for i in range(top_n):
    print(f"| {i+1} | Node {in_top[i]} | Node {out_top[i]} | Node {pr_top[i]} |")
```

### Cell 14 — markdown: `## Conclusion & Interpretation`

### Cell 15 — code: Interpretation

Follow ex-02 Cell 17 style — `print(f"""...""")` with structured sections:

1. Which nodes rank highest for in-degree (most trusted candidates)
2. Which nodes rank highest for out-degree (most active voters — typically different set)
3. Whether PageRank top nodes align with in-degree or out-degree (expected: closer to in-degree, as PageRank propagates incoming authority)
4. Why betweenness and eigenvector may differ further (bridge structure vs recursive prestige)
5. Key takeaway: direction fundamentally changes who appears "important"

---

## Hand-In Checklist (from prompt.md)

- [ ] Table of main metrics (Cell 9 structural metrics table)
- [ ] One centrality-ranked list of top 5 nodes (Cell 13 comparison table)
- [ ] One annotated visualization (`centrality_viz.html`)
- [ ] Short interpretation (Cell 15)

---

## Verification

1. Run all cells top-to-bottom — no errors
2. `centrality_viz.html` exists and opens in browser
3. Comparison table shows 3 columns (in-degree, out-degree, PageRank), 5 rows
4. Interpretation specifically addresses "how direction changes importance" per Student 6 prompt
