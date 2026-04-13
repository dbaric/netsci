import marimo

__generated_with = "0.17.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Exercise 04: Facebook Ego Network

    This notebook keeps the same SNAP Facebook ego network used in Exercises 02 and 03, with ego node **698**.

    Goal:
    test whether the ego is the main articulation point and whether the neighborhood has meaningful backup ties.

    Required input:
    `facebook/698.edges`

    Expected output:
    component counts before and after removals, articulation/bridge analysis, one before/after visualization, and a short resilience note.
    """)
    return


@app.cell
def _():
    from pathlib import Path

    import matplotlib.pyplot as plt
    import networkx as nx
    import numpy as np
    import pandas as pd
    return Path, np, nx, pd, plt


@app.cell
def _(Path):
    EGO_ID = 698

    def resolve_data_dir():
        candidates = []

        if "__file__" in globals():
            candidates.append(Path(__file__).resolve().parent)

        cwd = Path.cwd()
        candidates.extend(
            [
                cwd,
                cwd / "exercises" / "maksimilijankatavic",
            ]
        )

        for candidate in candidates:
            data_dir = candidate / "facebook"
            if data_dir.exists():
                return candidate, data_dir

        raise FileNotFoundError(
            "Could not find the facebook dataset directory relative to the notebook "
            "or repository root."
        )

    NOTEBOOK_DIR, DATA_DIR = resolve_data_dir()
    EDGE_PATH = DATA_DIR / f"{EGO_ID}.edges"

    if not EDGE_PATH.exists():
        raise FileNotFoundError(f"Could not find ego-network edge list at {EDGE_PATH}")
    return EDGE_PATH, EGO_ID


@app.cell
def _(nx):
    def load_ego_network(edge_path, ego_id):
        graph = nx.read_edgelist(edge_path, nodetype=int, create_using=nx.Graph())
        alters = sorted(graph.nodes())
        graph.add_node(ego_id)
        graph.add_edges_from((ego_id, alter) for alter in alters)
        return graph

    def normalize_edge(edge):
        return tuple(sorted(edge))

    def component_layout(
        graph,
        components,
        np_module,
        nx_module,
        anchor_node=None,
        orbit_radius=3.2,
        seed=698,
    ):
        positions = {}

        if anchor_node is not None:
            positions[anchor_node] = np_module.array([0.0, 0.0])

        if not components:
            return positions

        if len(components) == 1:
            centers = [np_module.array([0.0, 0.0])]
        else:
            angles = np_module.linspace(0, 2 * np_module.pi, len(components), endpoint=False)
            centers = [
                orbit_radius * np_module.array([np_module.cos(angle), np_module.sin(angle)])
                for angle in angles
            ]

        for index, component in enumerate(components):
            subgraph = graph.subgraph(component)
            sub_pos = nx_module.spring_layout(
                subgraph,
                seed=seed + index,
                k=1.0 / np_module.sqrt(max(subgraph.number_of_nodes(), 1)),
            )

            raw_positions = np_module.array([sub_pos[node] for node in subgraph.nodes()])
            local_center = raw_positions.mean(axis=0)
            spread = np_module.abs(raw_positions - local_center).max()
            if spread == 0:
                spread = 1.0

            local_scale = 0.7 + 0.12 * np_module.sqrt(subgraph.number_of_nodes())
            for node in subgraph.nodes():
                positions[node] = centers[index] + ((sub_pos[node] - local_center) / spread) * local_scale

        return positions
    return component_layout, load_ego_network, normalize_edge


@app.cell
def _(EDGE_PATH, EGO_ID, load_ego_network, normalize_edge, np, nx, pd):
    G = load_ego_network(EDGE_PATH, EGO_ID)

    base_components = sorted(nx.connected_components(G), key=len, reverse=True)
    base_component_sizes = [len(component) for component in base_components]

    alter_nodes = sorted(node for node in G.nodes() if node != EGO_ID)
    alter_graph = G.subgraph(alter_nodes).copy()
    alter_components = sorted(nx.connected_components(alter_graph), key=len, reverse=True)
    alter_component_sizes = [len(component) for component in alter_components]

    articulation_points = sorted(nx.articulation_points(G))
    bridges = sorted(normalize_edge(edge) for edge in nx.bridges(G))

    edge_betweenness = {
        normalize_edge(edge): score
        for edge, score in nx.edge_betweenness_centrality(G).items()
    }
    max_edge_betweenness = max(edge_betweenness.values())
    critical_edge_candidates = sorted(
        edge
        for edge, score in edge_betweenness.items()
        if np.isclose(score, max_edge_betweenness)
    )

    if bridges:
        critical_edge = bridges[0]
        critical_edge_reason = "bridge"
    else:
        critical_edge = critical_edge_candidates[0]
        critical_edge_reason = "highest edge betweenness"

    def summarize_graph(graph, label, removed_item):
        components = sorted(nx.connected_components(graph), key=len, reverse=True)
        sizes = [len(component) for component in components]
        nodes_remaining = graph.number_of_nodes()
        return {
            "scenario": label,
            "removed item": removed_item,
            "nodes remaining": nodes_remaining,
            "edges remaining": graph.number_of_edges(),
            "connected components": len(components),
            "component sizes": ", ".join(str(size) for size in sizes),
            "largest component size": sizes[0] if sizes else 0,
            "largest component share": round((sizes[0] / nodes_remaining) if sizes else 0.0, 3),
            "connected": nx.is_connected(graph) if nodes_remaining > 0 else False,
        }

    critical_node = articulation_points[0] if articulation_points else max(
        G.nodes(),
        key=lambda node: (
            nx.number_connected_components(nx.subgraph_view(G, filter_node=lambda candidate: candidate != node)),
            G.degree(node),
            -node,
        ),
    )

    G_without_node = G.copy()
    G_without_node.remove_node(critical_node)

    G_without_edge = G.copy()
    G_without_edge.remove_edge(*critical_edge)

    removal_df = pd.DataFrame(
        [
            summarize_graph(G, "full graph", "none"),
            summarize_graph(G_without_node, f"remove node {critical_node}", f"node {critical_node}"),
            summarize_graph(
                G_without_edge,
                f"remove edge {critical_edge}",
                f"edge {critical_edge}",
            ),
        ]
    )

    metrics_df = pd.DataFrame(
        [
            {"metric": "nodes", "value": G.number_of_nodes()},
            {"metric": "edges", "value": G.number_of_edges()},
            {"metric": "connected components", "value": len(base_component_sizes)},
            {"metric": "component sizes", "value": ", ".join(str(size) for size in base_component_sizes)},
            {"metric": "alter components without ego", "value": len(alter_component_sizes)},
            {
                "metric": "alter component sizes without ego",
                "value": ", ".join(str(size) for size in alter_component_sizes),
            },
            {"metric": "articulation points", "value": len(articulation_points)},
            {"metric": "bridges", "value": len(bridges)},
            {"metric": "node connectivity", "value": nx.node_connectivity(G)},
            {"metric": "edge connectivity", "value": nx.edge_connectivity(G)},
        ]
    )

    articulation_df = pd.DataFrame(
        {"articulation point": articulation_points}
    ) if articulation_points else pd.DataFrame({"articulation point": []})

    edge_candidates_df = (
        pd.DataFrame(
            [
                {
                    "edge": edge,
                    "edge betweenness": score,
                    "is bridge": edge in bridges,
                    "chosen critical edge": edge == critical_edge,
                }
                for edge, score in sorted(
                    edge_betweenness.items(),
                    key=lambda item: (-item[1], item[0]),
                )[:10]
            ]
        )
        .assign(edge=lambda df: df["edge"].map(str))
    )
    return (
        G,
        G_without_edge,
        G_without_node,
        alter_component_sizes,
        alter_components,
        articulation_df,
        articulation_points,
        bridges,
        critical_edge,
        critical_edge_candidates,
        critical_edge_reason,
        critical_node,
        edge_candidates_df,
        metrics_df,
        removal_df,
    )


@app.cell(hide_code=True)
def _(
    EGO_ID,
    G,
    articulation_points,
    critical_edge,
    critical_edge_candidates,
    critical_edge_reason,
    mo,
    removal_df,
):
    def _():
        node_row = removal_df.loc[removal_df["scenario"] == f"remove node {EGO_ID}"].iloc[0]
        edge_row = removal_df.loc[
            removal_df["scenario"] == f"remove edge {critical_edge}"
        ].iloc[0]

        if critical_edge_reason == "bridge":
            edge_note = f"The selected critical edge **{critical_edge}** is itself a bridge."
        else:
            tie_note = (
                f" It was chosen from **{len(critical_edge_candidates)}** tied maximum-betweenness edges."
                if len(critical_edge_candidates) > 1
                else ""
            )
            edge_note = (
                f"There are **no bridges**, so I remove edge **{critical_edge}** because it has the "
                f"highest edge betweenness.{tie_note}"
            )

        articulation_note = (
            f"The only articulation point is **{articulation_points[0]}**."
            if articulation_points
            else "There are no articulation points."
        )
        return mo.md(
            f"""
            The full graph has **{G.number_of_nodes()} nodes** and **{G.number_of_edges()} edges**, and it starts as **one connected component**.
            {articulation_note}
            {edge_note}

            Removing node **{EGO_ID}** expands the graph from **1** to **{int(node_row['connected components'])}** components and shrinks the largest component from **{G.number_of_nodes()}** to **{int(node_row['largest component size'])}** nodes.
            Removing edge **{critical_edge}** keeps the graph at **{int(edge_row['connected components'])}** component, so the neighborhood is fragile to one node but not to one edge.
            """
        )


    _()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Connectivity Summary
    """)
    return


@app.cell
def _(metrics_df):
    metrics_df
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Articulation Points and Edge Candidates
    """)
    return


@app.cell
def _(articulation_df, mo):
    if articulation_df.empty:
        mo.md("No articulation points were found.")
    else:
        articulation_df
    return


@app.cell(hide_code=True)
def _(bridges, mo):
    if bridges:
        mo.md(f"Bridges: {bridges}")
    else:
        mo.md("No bridges were found, so no single edge disconnects the graph.")
    return


@app.cell
def _(edge_candidates_df):
    edge_candidates_df.round({"edge betweenness": 4})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Before and After Removals
    """)
    return


@app.cell
def _(removal_df):
    removal_df
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Connectivity Visualization

    Left: original ego network.
    Middle: after removing the articulation point.
    Right: after removing the selected critical edge.
    """)
    return


@app.cell
def _(
    EGO_ID,
    G,
    G_without_edge,
    G_without_node,
    alter_components,
    component_layout,
    critical_edge,
    critical_node,
    np,
    nx,
    plt,
):
    palette = [
        tuple(color)
        for color in plt.cm.Set2(np.linspace(0, 1, max(len(alter_components), 1)))
    ]

    component_color_lookup = {}
    for index, component in enumerate(alter_components):
        for node in component:
            component_color_lookup[node] = palette[index]

    base_pos = component_layout(
        G,
        alter_components,
        np,
        nx,
        anchor_node=EGO_ID,
        orbit_radius=3.3,
        seed=EGO_ID,
    )

    node_removed_components = sorted(
        nx.connected_components(G_without_node),
        key=len,
        reverse=True,
    )
    node_removed_pos = component_layout(
        G_without_node,
        node_removed_components,
        np,
        nx,
        orbit_radius=3.0,
        seed=EGO_ID + 100,
    )

    node_removed_colors = {}
    for index, component in enumerate(node_removed_components):
        for node in component:
            node_removed_colors[node] = palette[index % len(palette)]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    base_alter_nodes = [node for node in G.nodes() if node != EGO_ID]
    base_alter_sizes = [70 + 11 * G.degree(node) for node in base_alter_nodes]
    base_alter_colors = [component_color_lookup[node] for node in base_alter_nodes]

    nx.draw_networkx_edges(
        G,
        base_pos,
        ax=axes[0],
        edge_color="#B8B8B8",
        width=0.9,
        alpha=0.35,
    )
    nx.draw_networkx_edges(
        G,
        base_pos,
        edgelist=[critical_edge],
        ax=axes[0],
        edge_color="#1A1A1A",
        width=2.0,
        alpha=0.9,
    )
    nx.draw_networkx_nodes(
        G,
        base_pos,
        nodelist=base_alter_nodes,
        node_size=base_alter_sizes,
        node_color=base_alter_colors,
        edgecolors="white",
        linewidths=0.5,
        ax=axes[0],
    )
    nx.draw_networkx_nodes(
        G,
        base_pos,
        nodelist=[critical_node],
        node_size=3200,
        node_color="#D1495B",
        edgecolors="white",
        linewidths=1.5,
        ax=axes[0],
    )
    nx.draw_networkx_labels(
        G,
        base_pos,
        labels={
            critical_node: str(critical_node),
            critical_edge[1]: str(critical_edge[1]),
        },
        font_size=10,
        font_weight="bold",
        font_color="white",
        ax=axes[0],
    )
    axes[0].set_title("Original graph\nEgo and chosen edge highlighted")
    axes[0].axis("off")

    node_removed_nodes = list(G_without_node.nodes())
    node_removed_sizes = [70 + 11 * G_without_node.degree(node) for node in node_removed_nodes]
    nx.draw_networkx_edges(
        G_without_node,
        node_removed_pos,
        ax=axes[1],
        edge_color="#B8B8B8",
        width=0.9,
        alpha=0.4,
    )
    nx.draw_networkx_nodes(
        G_without_node,
        node_removed_pos,
        nodelist=node_removed_nodes,
        node_size=node_removed_sizes,
        node_color=[node_removed_colors[node] for node in node_removed_nodes],
        edgecolors="white",
        linewidths=0.5,
        ax=axes[1],
    )
    axes[1].set_title("After removing node 698\nThree disconnected alter groups")
    axes[1].axis("off")

    nx.draw_networkx_edges(
        G_without_edge,
        base_pos,
        ax=axes[2],
        edge_color="#B8B8B8",
        width=0.9,
        alpha=0.35,
    )
    nx.draw_networkx_nodes(
        G_without_edge,
        base_pos,
        nodelist=base_alter_nodes,
        node_size=base_alter_sizes,
        node_color=base_alter_colors,
        edgecolors="white",
        linewidths=0.5,
        ax=axes[2],
    )
    nx.draw_networkx_nodes(
        G_without_edge,
        base_pos,
        nodelist=[EGO_ID],
        node_size=3200,
        node_color="#D1495B",
        edgecolors="white",
        linewidths=1.5,
        ax=axes[2],
    )
    nx.draw_networkx_nodes(
        G_without_edge,
        base_pos,
        nodelist=list(critical_edge),
        node_size=[3200 if node == EGO_ID else 70 + 11 * G.degree(node) for node in critical_edge],
        node_color=[
            (0.8196, 0.2863, 0.3569, 1.0)
            if node == EGO_ID
            else component_color_lookup[node]
            for node in critical_edge
        ],
        edgecolors="#1A1A1A",
        linewidths=2.0,
        ax=axes[2],
    )
    nx.draw_networkx_labels(
        G_without_edge,
        base_pos,
        labels={EGO_ID: str(EGO_ID), critical_edge[1]: str(critical_edge[1])},
        font_size=10,
        font_weight="bold",
        font_color="white",
        ax=axes[2],
    )
    axes[2].set_title("After removing edge {0}\nGraph stays connected".format(critical_edge))
    axes[2].axis("off")

    fig.tight_layout()
    fig
    return


@app.cell(hide_code=True)
def _(
    EGO_ID,
    alter_component_sizes,
    bridges,
    critical_edge,
    critical_edge_candidates,
    mo,
    removal_df,
):
    node_row = removal_df.loc[removal_df["scenario"] == f"remove node {EGO_ID}"].iloc[0]
    edge_row = removal_df.loc[
        removal_df["scenario"] == f"remove edge {critical_edge}"
    ].iloc[0]

    tie_note = (
        f" The chosen edge is one of {len(critical_edge_candidates)} tied maximum-betweenness edges."
        if len(critical_edge_candidates) > 1
        else ""
    )

    bridge_note = (
        "Because there are no bridges, no single edge failure disconnects the graph."
        if not bridges
        else f"The selected edge {critical_edge} is a bridge, so a single edge failure can disconnect the graph."
    )

    mo.md(
        f"""
        ## Interpretation

        This ego network depends heavily on **one critical connector: node {EGO_ID}**.
        The full graph is connected only because the ego ties together three alter groups of sizes **{", ".join(str(size) for size in alter_component_sizes)}**.
        Once the ego is removed, the network breaks into **{int(node_row['connected components'])}** components and the largest connected component falls to **{int(node_row['largest component size'])}** of **{int(node_row['nodes remaining'])}** remaining nodes.

        The edge story is different.
        {bridge_note}{tie_note}
        Removing edge **{critical_edge}** leaves the graph with **{int(edge_row['connected components'])}** connected component and the largest component still covers **{int(edge_row['largest component size'])}** nodes.

        So the neighborhood has **backup ties for individual edges**, but **not for the ego itself**.
        Its main vulnerability is therefore **single-node dependence on the ego**, not single-edge dependence.
        """
    )
    return


if __name__ == "__main__":
    app.run()
