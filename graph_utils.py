import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

def draw_graph(graph, labels=False, node_size=100, fig_size=8):
    """wraps the networkx draw function in a nice way"""
    pos = nx.fruchterman_reingold_layout(graph);
    p = plt.figure(figsize=(fig_size,fig_size));
    plt.axis("off");
    nx.draw_networkx_nodes(graph, pos, node_size=node_size, node_color="black");
    nx.draw_networkx_edges(graph, pos, alpha=0.500);
    if(labels):
        nx.draw_networkx_labels(graph, pos, font_color="white");
    return p

def get_net(Q):
    """Draw a graph based on a position matrix with dimension Nxd"""
    G = nx.Graph()
    for i in range(0,N):
        for j in range(0,N):
            if (np.linalg.norm(Q[i,:]-Q[j,:])<r):
                if(not G.has_edge(i,j)):
                    G.add_edge(i,j)
    return(G)
