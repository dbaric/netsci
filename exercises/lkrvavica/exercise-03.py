import marimo

__generated_with = "0.21.1"
app = marimo.App(
    width="medium",
    app_title="Exercise 03 — Gowalla Centrality Analysis",
)


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md("""
    ## ① Load Graph (same as Exercise 02)
    """)
    return


@app.cell
def _():
    import kagglehub
    import os
    import random
    import pandas as pd
    import networkx as nx

    path = kagglehub.dataset_download("marquis03/gowalla")
    edge_file = os.path.join(path, "Gowalla_edges.txt")
    df_edges = pd.read_csv(edge_file, sep="\t", header=None, names=["user_a", "user_b"])

    G_full = nx.from_pandas_edgelist(df_edges, source="user_a", target="user_b")
    print(f"Full graph  — Nodes: {G_full.number_of_nodes():,}  |  Edges: {G_full.number_of_edges():,}")

    # Same BFS sample from Exercise 02
    random.seed(42)
    degrees_full = dict(G_full.degree())
    top_node = max(degrees_full, key=lambda n: degrees_full[n])
    bfs_nodes = list(nx.bfs_tree(G_full, top_node).nodes())[:2000]
    G = G_full.subgraph(bfs_nodes).copy()

    # Largest connected component — required for path-based metrics
    largest_cc = max(nx.connected_components(G), key=len)
    G_cc = G.subgraph(largest_cc).copy()
    print(f"Sample LCC  — Nodes: {G_cc.number_of_nodes():,}  |  Edges: {G_cc.number_of_edges():,}")
    print(f"Note: graph is undirected — no in/out degree split needed.")
    return G_cc, nx, pd


@app.cell
def _(mo):
    mo.md("""
    ## ② Degree-Based Measures
    """)
    return


@app.cell
def _(G_cc, nx, pd):
    deg = dict(G_cc.degree())
    deg_centrality = nx.degree_centrality(G_cc)

    top_deg = sorted(deg.items(), key=lambda x: x[1], reverse=True)[:10]
    df_deg = pd.DataFrame(top_deg, columns=["Node", "Degree"])
    df_deg["Degree Centrality"] = df_deg["Node"].map(deg_centrality).round(4)
    df_deg = df_deg.reset_index(drop=True)
    df_deg.index += 1

    print("Top 10 nodes by degree:")
    print(df_deg.to_string())
    return


@app.cell
def _(mo):
    mo.md("""
    ## ③ Five Centrality Measures
    """)
    return


@app.cell
def _(G_cc, nx):
    print("Computing degree centrality...")
    cent_degree = nx.degree_centrality(G_cc)

    print("Computing betweenness centrality (k=500 approximation)...")
    cent_between = nx.betweenness_centrality(G_cc, k=500, seed=42, normalized=True)

    print("Computing closeness centrality...")
    cent_closeness = nx.closeness_centrality(G_cc)

    print("Computing eigenvector centrality...")
    cent_eigen = nx.eigenvector_centrality(G_cc, max_iter=500)

    print("Computing PageRank...")
    cent_pagerank = nx.pagerank(G_cc, alpha=0.85)

    print("All centrality measures computed.")
    return cent_between, cent_closeness, cent_degree, cent_eigen, cent_pagerank


@app.cell
def _(
    cent_between,
    cent_closeness,
    cent_degree,
    cent_eigen,
    cent_pagerank,
    pd,
):
    nodes = list(cent_degree.keys())
    df_cent = pd.DataFrame({
        "Node":        nodes,
        "Degree":      [round(cent_degree[n],    5) for n in nodes],
        "Betweenness": [round(cent_between[n],   5) for n in nodes],
        "Closeness":   [round(cent_closeness[n], 5) for n in nodes],
        "Eigenvector": [round(cent_eigen[n],     5) for n in nodes],
        "PageRank":    [round(cent_pagerank[n],  6) for n in nodes],
    })

    for col in ["Degree", "Betweenness", "Closeness", "Eigenvector", "PageRank"]:
        top5 = df_cent.nlargest(5, col)[["Node", col]]
        print(f"Top 5 by {col}:\n{top5.to_string(index=False)}\n")
    return (df_cent,)


@app.cell
def _(df_cent, mo):
    def top5_table(df_cent, col):
        rows = df_cent.nlargest(5, col)[["Node", col]].reset_index(drop=True)
        header = f"| Rank | Node | {col} |\n|---|---|---|\n"
        body = "\n".join(f"| {i+1} | {int(r['Node'])} | {r[col]} |" for i, r in rows.iterrows())
        return header + body

    mo.md(f"""
    ### Top 5 by each centrality measure

    **Degree** — raw connectivity
    {top5_table(df_cent, 'Degree')}

    **Betweenness** — bridge/broker role
    {top5_table(df_cent, 'Betweenness')}

    **Closeness** — speed of reaching all others
    {top5_table(df_cent, 'Closeness')}

    **Eigenvector** — connected to well-connected nodes
    {top5_table(df_cent, 'Eigenvector')}

    **PageRank** — global influence via random walk
    {top5_table(df_cent, 'PageRank')}
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## ④ Structural Metrics (Density, Clustering, Path-based)
    """)
    return


@app.cell
def _(G_cc, nx):
    density = nx.density(G_cc)
    avg_clustering = nx.average_clustering(G_cc)

    # Diameter and avg path length are O(n^2) — sample 500 nodes for speed
    import random as _random
    _random.seed(42)
    sample_nodes = _random.sample(list(G_cc.nodes()), min(500, G_cc.number_of_nodes()))

    path_lengths = []
    for src in sample_nodes:
        lengths = nx.single_source_shortest_path_length(G_cc, src)
        path_lengths.extend(lengths.values())

    avg_path_len = sum(path_lengths) / len(path_lengths)
    # Eccentricity via sampled max — true diameter too expensive on 2k nodes
    ecc_sample = {n: max(nx.single_source_shortest_path_length(G_cc, n).values())
                  for n in sample_nodes[:100]}
    approx_diameter = max(ecc_sample.values())

    print(f"Density:                  {density:.6f}")
    print(f"Average clustering:       {avg_clustering:.4f}")
    print(f"Avg shortest path length: {avg_path_len:.4f}  (sampled 500 nodes)")
    print(f"Approx diameter:          {approx_diameter}  (sampled 100 nodes eccentricity)")
    print()
    print("Note: all path metrics run on the largest connected component (2,000 nodes, 1 component).")
    return approx_diameter, avg_clustering, avg_path_len, density


@app.cell
def _(approx_diameter, avg_clustering, avg_path_len, density, mo):
    mo.md(f"""
    ### Structural Metrics Table

    | Metric | Value | Note |
    |---|---|---|
    | Density | {density:.6f} | Fraction of possible edges present |
    | Average clustering | {avg_clustering:.4f} | ~46% of triangles close |
    | Avg shortest path length | {avg_path_len:.4f} | Sampled 500-node estimate |
    | Approx diameter | {approx_diameter} | Max eccentricity, 100-node sample |
    | LCC coverage | 100% | All 2,000 sample nodes in one component |

    > Path-based metrics are computed on the **largest connected component** (the entire sample in this case).
    > True diameter computation is O(n²) so eccentricity was sampled from 100 nodes.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## ⑤ Centrality-Ranked Top 5 Nodes
    """)
    return


@app.cell
def _(
    cent_between,
    cent_closeness,
    cent_degree,
    cent_eigen,
    cent_pagerank,
    pd,
):
    # Build a combined rank table for the top 5 by degree
    top5_nodes = sorted(cent_degree, key=cent_degree.get, reverse=True)[:5]

    df_top5 = pd.DataFrame({
        "Node":        top5_nodes,
        "Degree":      [round(cent_degree[n],    4) for n in top5_nodes],
        "Betweenness": [round(cent_between[n],   4) for n in top5_nodes],
        "Closeness":   [round(cent_closeness[n], 4) for n in top5_nodes],
        "Eigenvector": [round(cent_eigen[n],     4) for n in top5_nodes],
        "PageRank":    [round(cent_pagerank[n],  5) for n in top5_nodes],
    }).reset_index(drop=True)
    df_top5.index += 1

    print("Top 5 nodes (ranked by degree) — all centrality scores:")
    print(df_top5.to_string())
    return (df_top5,)


@app.cell
def _(df_top5, mo):
    rows_md = "\n".join(
        f"| {i+1} | {int(r['Node'])} | {r['Degree']} | {r['Betweenness']} | {r['Closeness']} | {r['Eigenvector']} | {r['PageRank']} |"
        for i, r in df_top5.iterrows()
    )
    mo.md(f"""
    ### Top 5 Nodes — All Centrality Scores

    | Rank | Node | Degree | Betweenness | Closeness | Eigenvector | PageRank |
    |---|---|---|---|---|---|---|
    {rows_md}

    > Ranked by **degree**. Compare columns to see where metrics agree and diverge.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## ⑥ Annotated Visualization — Node Size & Color by Centrality
    """)
    return


@app.cell
def _(G_cc, cent_between, cent_degree, mo, nx):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import numpy as np

    # Visualize ego-network of top hub (same as ex02) — cap at 300 nodes
    import random as _r
    _r.seed(42)
    hub = max(cent_degree, key=cent_degree.get)
    ego = nx.ego_graph(G_cc, hub, radius=2)
    if ego.number_of_nodes() > 300:
        _keep = [hub] + _r.sample(list(ego.nodes() - {hub}), 299)
        ego = ego.subgraph(_keep).copy()

    ego_nodes = list(ego.nodes())
    pos = nx.spring_layout(ego, seed=42, k=0.45)

    # --- Size = degree centrality ---
    sizes = np.array([cent_degree.get(n, 0) for n in ego_nodes])
    sizes = 40 + (sizes / sizes.max()) * 800

    # --- Color = betweenness centrality (highlights brokers) ---
    bw = np.array([cent_between.get(n, 0) for n in ego_nodes])
    norm = mcolors.PowerNorm(gamma=0.4, vmin=bw.min(), vmax=bw.max())
    cmap = plt.cm.plasma
    # Apply norm manually — draw_networkx_nodes doesn't accept norm kwarg
    node_colors = cmap(norm(bw))

    fig, ax = plt.subplots(figsize=(13, 10))
    fig.patch.set_facecolor("#090b14")
    ax.set_facecolor("#090b14")
    ax.axis("off")

    nx.draw_networkx_edges(ego, pos, ax=ax, alpha=0.12, edge_color="#5588cc", width=0.5)

    sc = nx.draw_networkx_nodes(
        ego, pos, ax=ax,
        node_size=sizes,
        node_color=node_colors,
        alpha=0.92,
    )

    # Label top 8 by degree
    top_labels = dict(sorted(cent_degree.items(), key=lambda x: x[1], reverse=True)[:8])
    top_labels = {n: str(n) for n in top_labels if n in ego_nodes}
    nx.draw_networkx_labels(ego, pos, ax=ax, labels=top_labels,
                            font_color="white", font_size=7, font_weight="bold")

    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax,
                        shrink=0.5, pad=0.01, location="right")
    cbar.set_label("Betweenness Centrality", color="#ccd0e0", fontsize=10)
    cbar.ax.yaxis.set_tick_params(color="#ccd0e0")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#ccd0e0", fontsize=8)

    ax.set_title(
        f"Gowalla — Ego-Network of Hub {hub}\n"
        f"Node SIZE = degree centrality  |  Node COLOR = betweenness centrality\n"
        f"{ego.number_of_nodes()} nodes · {ego.number_of_edges()} edges",
        color="#dde0f0", fontsize=12, pad=14, linespacing=1.6
    )

    plt.tight_layout()
    mo.mpl.interactive(fig)
    return


@app.cell
def _(mo):
    mo.md("""
    ## ⑦ Interpretation

    Node 307 dominates every single centrality metric — degree 1.0, betweenness 0.653, closeness 1.0, eigenvector 0.256, PageRank 0.034. This is expected because the sample was built by BFS from node 307, so it is literally at the centre of the subgraph by construction. The more meaningful signal is the gap between rank 1 and rank 2: betweenness drops from 0.653 to 0.017, meaning node 307 controls roughly 38× more shortest paths than the next broker. This makes it not just well-connected but a near-mandatory relay — remove it and the sample subgraph would fragment badly.
    The second tier (nodes 459, 505, 2, 220) shows an interesting split: 459 ranks 2nd in degree and closeness but only 3rd in betweenness, while node 2 ranks 4th in betweenness but 6th in degree. Node 2 has fewer friends than 459 but sits on proportionally more shortest paths — a classic sign of a bridge user connecting otherwise separate clusters rather than a pure local hub. Node 2255 appears only in the eigenvector top 5, meaning it has moderate raw connections but its neighbours are unusually well-connected — a quiet influencer invisible to the other metrics.
    The avg shortest path of 1.98 hops and approx diameter of 2 confirm the sample is essentially a 2-hop star around node 307, which is why all path-based metrics (closeness especially) score so uniformly high across the top nodes — everyone is close to everyone because everyone routes through 307.
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
