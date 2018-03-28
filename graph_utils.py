import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

def draw_graph(graph, labels=False, node_size=150, fig_size=8):
    pos = nx.fruchterman_reingold_layout(graph);
    p = plt.figure(figsize=(fig_size,fig_size));
    plt.axis("off");
    nx.draw_networkx_nodes(graph, pos, node_size=node_size, node_color="black");
    nx.draw_networkx_edges(graph, pos, alpha=0.500);
    if(labels):
        nx.draw_networkx_labels(graph, pos, font_color="white");
    plt.show();
    return p
