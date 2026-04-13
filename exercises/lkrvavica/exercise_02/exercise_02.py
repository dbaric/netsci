import marimo

__generated_with = "0.21.1"
app = marimo.App(
    width="medium",
    app_title="Exercise 02 — Gowalla Graph Analysis",
)


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md("""
    ## ① Environment Setup & Data Loading
    """)
    return


@app.cell
def _():
    import kagglehub
    import os

    path = kagglehub.dataset_download("marquis03/gowalla")
    print("Path to dataset files:", path)
    files = os.listdir(path)
    print("Files:", files)
    return os, path


@app.cell
def _(os, path):
    import pandas as pd

    # Explicitly load the EDGES file (user friendships), NOT the checkins file
    edge_file = os.path.join(path, "Gowalla_edges.txt")
    print("Using:", edge_file)

    df_edges = pd.read_csv(edge_file, sep="\t", header=None, names=["user_a", "user_b"])
    print(f"Raw edge count: {len(df_edges):,}")
    df_edges.head(10)
    return (df_edges,)


@app.cell
def _(mo):
    mo.md("""
    ## ② Build the Graph Object
    """)
    return


@app.cell
def _(df_edges):
    import networkx as nx

    # Gowalla friendships are undirected (mutual)
    G_full = nx.from_pandas_edgelist(df_edges, source="user_a", target="user_b")
    print(f"Full graph — Nodes: {G_full.number_of_nodes():,}  |  Edges: {G_full.number_of_edges():,}")
    return G_full, nx


@app.cell
def _(G_full, nx):
    import random
    random.seed(42)

    # BFS sample from highest-degree node for a dense, well-connected subgraph
    degrees = dict(G_full.degree())
    top_node = max(degrees, key=lambda n: degrees[n])
    print(f"Starting BFS from node {top_node} (degree {degrees[top_node]})")

    bfs_nodes = list(nx.bfs_tree(G_full, top_node).nodes())[:2000]
    G = G_full.subgraph(bfs_nodes).copy()
    print(f"Sampled subgraph — Nodes: {G.number_of_nodes():,}  |  Edges: {G.number_of_edges():,}")
    return G, random


@app.cell
def _(mo):
    mo.md("""
    ## ③ Key Metrics
    """)
    return


@app.cell
def _(G, G_full, nx):
    import statistics

    n_full = G_full.number_of_nodes()
    e_full = G_full.number_of_edges()
    density_full = nx.density(G_full)
    deg_full = [d for _, d in G_full.degree()]
    avg_deg_full = sum(deg_full) / len(deg_full)

    n = G.number_of_nodes()
    e = G.number_of_edges()
    density = nx.density(G)
    deg_seq = sorted([d for _, d in G.degree()], reverse=True)
    avg_deg = sum(deg_seq) / len(deg_seq)
    med_deg = statistics.median(deg_seq)
    max_deg = max(deg_seq)
    min_deg = min(deg_seq)

    components = list(nx.connected_components(G))
    n_components = len(components)
    largest_cc = max(components, key=len)
    avg_clustering = nx.average_clustering(G)

    print("=" * 50)
    print("FULL GRAPH METRICS")
    print(f"  Nodes:      {n_full:,}")
    print(f"  Edges:      {e_full:,}")
    print(f"  Density:    {density_full:.8f}")
    print(f"  Avg Degree: {avg_deg_full:.2f}")
    print()
    print("SAMPLE SUBGRAPH METRICS (BFS-2000 nodes)")
    print(f"  Nodes:            {n:,}")
    print(f"  Edges:            {e:,}")
    print(f"  Density:          {density:.6f}")
    print(f"  Avg Degree:       {avg_deg:.2f}")
    print(f"  Median Degree:    {med_deg}")
    print(f"  Max Degree:       {max_deg}")
    print(f"  Min Degree:       {min_deg}")
    print(f"  Components:       {n_components}")
    print(f"  Largest CC size:  {len(largest_cc):,}")
    print(f"  Avg Clustering:   {avg_clustering:.4f}")
    return (
        avg_clustering,
        avg_deg,
        avg_deg_full,
        deg_seq,
        density,
        density_full,
        e,
        e_full,
        largest_cc,
        max_deg,
        med_deg,
        min_deg,
        n,
        n_components,
        n_full,
    )


@app.cell
def _(
    avg_clustering,
    avg_deg,
    density,
    e,
    e_full,
    largest_cc,
    max_deg,
    med_deg,
    min_deg,
    mo,
    n,
    n_components,
    n_full,
):
    mo.md(f"""
    ### 📊 Metrics Summary Table

    | Metric | Full Graph | Sample (BFS-2000) |
    |---|---|---|
    | **Nodes** | {n_full:,} | {n:,} |
    | **Edges** | {e_full:,} | {e:,} |
    | **Density** | {density:.8f} | — |
    | **Avg Degree** | — | {avg_deg:.2f} |
    | **Median Degree** | — | {med_deg} |
    | **Max Degree** | — | {max_deg} |
    | **Min Degree** | — | {min_deg} |
    | **Components** | — | {n_components} |
    | **Largest CC** | — | {len(largest_cc):,} nodes |
    | **Avg Clustering** | — | {avg_clustering:.4f} |
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## ④ Shortest Path & Cycle
    """)
    return


@app.cell
def _(G, largest_cc, nx, random):
    G_cc = G.subgraph(largest_cc).copy()
    cc_nodes = list(G_cc.nodes())
    random.seed(7)
    src, tgt = random.sample(cc_nodes, 2)

    sp = nx.shortest_path(G_cc, source=src, target=tgt)
    sp_len = nx.shortest_path_length(G_cc, source=src, target=tgt)

    print(f"Shortest path from user {src} → user {tgt}:")
    print(f"  Length: {sp_len} hops")
    print(f"  Path:   {sp}")

    try:
        cycle = nx.find_cycle(G_cc)
        print(f"\nExample cycle (first 6 edges):")
        for edge in cycle[:6]:
            print(f"  {edge[0]} ↔ {edge[1]}")
        if len(cycle) > 6:
            print(f"  ... ({len(cycle)} edges total)")
    except nx.exception.NetworkXNoCycle:
        cycle = None
        print("\nNo cycle found in this sample (tree-like structure).")

    print("\nAdjacency list snippet (first 8 nodes, max 5 neighbors):")
    for _node in list(G_cc.nodes())[:8]:
        _nbrs = list(G_cc.neighbors(_node))[:5]
        _sfx = "..." if G_cc.degree(_node) > 5 else ""
        print(f"  {_node}: {_nbrs}{_sfx}")
    return G_cc, sp, sp_len, src, tgt


@app.cell
def _(mo, sp, sp_len, src, tgt):
    mo.md(f"""
    ### 🔍 Path Analysis

    **Shortest path** from user `{src}` → user `{tgt}`:  
    `{" → ".join(str(n) for n in sp)}`  
    **Length:** {sp_len} hops

    > This demonstrates the classic **small-world** property of social networks — most users
    > are reachable within very few hops despite the network's large size.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## ⑤ Degree Distribution
    """)
    return


@app.cell
def _(deg_seq, mo):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    fig.patch.set_facecolor("#0f1117")

    for ax in axes:
        ax.set_facecolor("#1a1d2e")
        ax.tick_params(colors="#aab0c8")
        for spine in ax.spines.values():
            spine.set_edgecolor("#2e3350")

    axes[0].hist(deg_seq, bins=50, color="#4f8ef7", edgecolor="#0f1117", linewidth=0.3, alpha=0.9)
    axes[0].set_title("Degree Distribution (linear)", color="#e0e4f5", fontsize=12, pad=10)
    axes[0].set_xlabel("Degree", color="#aab0c8")
    axes[0].set_ylabel("Count", color="#aab0c8")

    axes[1].hist(deg_seq, bins=50, color="#f76f8e", edgecolor="#0f1117", linewidth=0.3, alpha=0.9, log=True)
    axes[1].set_xscale("log")
    axes[1].set_title("Degree Distribution (log-log)", color="#e0e4f5", fontsize=12, pad=10)
    axes[1].set_xlabel("Degree (log)", color="#aab0c8")
    axes[1].set_ylabel("Count (log)", color="#aab0c8")

    plt.suptitle("Gowalla Sample — Degree Distribution", color="#e0e4f5", fontsize=14, y=1.02)
    plt.tight_layout()

    # Use mo.mpl.interactive() — no file saving needed
    mo.mpl.interactive(fig)
    return np, plt


@app.cell
def _(mo):
    mo.md("""
    ## ⑥ Graph Visualization (Sample)
    """)
    return


@app.cell
def _(G_cc, mo, np, nx, plt, random):
    random.seed(42)
    viz_nodes_raw = sorted(G_cc.nodes(), key=lambda n: G_cc.degree(n), reverse=True)
    hub = viz_nodes_raw[0]

    ego = nx.ego_graph(G_cc, hub, radius=2)
    if ego.number_of_nodes() > 300:
        _keep = [hub] + random.sample(list(ego.nodes() - {hub}), 299)
        ego = ego.subgraph(_keep).copy()

    print(f"Visualization subgraph: {ego.number_of_nodes()} nodes, {ego.number_of_edges()} edges")
    print(f"Central hub: user {hub} (degree {G_cc.degree(hub)})")

    pos = nx.spring_layout(ego, seed=42, k=0.4)
    node_deg = np.array([ego.degree(n) for n in ego.nodes()])
    node_sizes = 30 + node_deg * 12
    node_colors = ["#f7c948" if n == hub else "#4f8ef7" for n in ego.nodes()]

    fig2, ax2 = plt.subplots(figsize=(12, 10))
    fig2.patch.set_facecolor("#0b0d18")
    ax2.set_facecolor("#0b0d18")
    ax2.axis("off")

    nx.draw_networkx_edges(ego, pos, ax=ax2, alpha=0.18, edge_color="#4f8ef7", width=0.6)
    nx.draw_networkx_nodes(ego, pos, ax=ax2, node_size=node_sizes, node_color=node_colors, alpha=0.88)
    nx.draw_networkx_labels(
        ego, pos, ax=ax2,
        labels={hub: f"hub\n({hub})"},
        font_color="#0b0d18", font_size=7, font_weight="bold"
    )
    ax2.set_title(
        f"Gowalla Ego-Network (2-hop, hub={hub})\n{ego.number_of_nodes()} nodes · {ego.number_of_edges()} edges",
        color="#e0e4f5", fontsize=13, pad=12
    )
    plt.tight_layout()

    # Use mo.mpl.interactive() — no file saving needed
    mo.mpl.interactive(fig2)
    return


@app.cell
def _(mo):
    mo.md("""
    ## ⑦ Interpretation & Reflection
    """)
    return


@app.cell
def _(
    avg_clustering,
    avg_deg,
    avg_deg_full,
    density,
    density_full,
    e_full,
    max_deg,
    med_deg,
    mo,
    n_full,
    sp_len,
):
    mo.md(f"""
     ### 📝 Method Note
        **Commands used:** `kagglehub.dataset_download`, `pd.read_csv` (explicit `Gowalla_edges.txt`),  
        `nx.from_pandas_edgelist`, `nx.density`, `nx.connected_components`, `nx.average_clustering`,  
        `nx.shortest_path`, `nx.find_cycle`, `nx.ego_graph`, `nx.spring_layout`, `mo.mpl.interactive`
 
        ---
    
 
        ### 📋 Summary Table
 
        | Property | Value |
        |---|---|
        | Full graph nodes | {n_full:,} |
        | Full graph edges | {e_full:,} |
        | **Full graph density** | {density_full:.8f} |
        | Full graph avg degree | {avg_deg_full:.2f} |
        | Sample avg degree | {avg_deg:.2f} |
        | Sample median degree | {med_deg} |
        | Sample max degree | {max_deg} |
        | Sample avg clustering | {avg_clustering:.4f} |
        | Example shortest path | {sp_len} hops |
 
        ---
 
        ### 🧭 Interpretation
 
        The Gowalla friendship graph has **{n_full:,} users** and **{e_full:,} mutual friendship edges**.
        The full graph density is {density_full:.8f} — genuinely sparse, meaning the average user (avg degree {avg_deg_full:.1f})
        connects to under 10 out of nearly 200k possible partners. The BFS sample centred on the highest-degree
        hub (node 307, degree 14,730) is denser by construction (density {density:.4f}, avg degree {avg_deg:.1f}),
        because BFS from a supernode naturally pulls in its well-connected neighbourhood rather than a random
        cross-section of the full graph — a sampling bias worth noting. Degree distribution is strongly
        **right-skewed**: median {med_deg} versus max {max_deg} confirms that a tiny number of power-user hubs
        dominate, consistent with a scale-free network growing by preferential attachment. The high
        **average clustering coefficient of {avg_clustering:.4f}** shows that friends of friends are very often
        also friends, pointing to dense local communities around hubs. The shortest path of just **{sp_len} hops**
        between two randomly chosen users (routing through hub 307) is a textbook example of the *small-world*
        effect: high-degree hubs act as shortcuts that collapse geodesic distances across the whole network.
        Lecture 02 concepts — nodes as users, undirected edges as mutual friendships, BFS subgraph sampling,
        adjacency lists, density and degree statistics — provided a clean quantitative description of this
        real-world social graph and lay the foundation for centrality, community detection, and diffusion
        analysis in later exercises.
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
