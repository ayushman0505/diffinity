# utils/graph_drawer.py

import networkx as nx
import matplotlib.pyplot as plt
import streamlit as st


def draw_diff_graph(differences, max_nodes=20):
    """
    Visualizes semantic differences using a graph.
    Red nodes = REMOVED, Green nodes = ADDED
    """
    G = nx.DiGraph()
    label_map = {"ADDED": "green", "REMOVED": "red"}

    for label, text, score in differences[:max_nodes]:
        display_text = text[:50] + ("..." if len(text) > 50 else "")
        G.add_node(display_text, color=label_map.get(label, "gray"), label=label)

    pos = nx.spring_layout(G, seed=42)
    node_colors = [G.nodes[n]["color"] for n in G.nodes]

    plt.figure(figsize=(10, 6))
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_color=node_colors,
        node_size=1000,
        font_size=8,
        font_color="white",
        edge_color="#CCCCCC",
    )

    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', label='ADDED', markersize=10, markerfacecolor='green'),
        plt.Line2D([0], [0], marker='o', color='w', label='REMOVED', markersize=10, markerfacecolor='red')
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    plt.title("Semantic Differences Between Documents")
    st.pyplot(plt)
