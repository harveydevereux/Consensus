import numpy as np
import networkx as nx

def dynamics(x,L):
    """Most basic distributed consensus algorithm
       It's actually gradient descent of 1/2(xLx)"""
    return -np.dot(L,x)

def dynamics_random_topology(x,graph, proportion):
    """Same as dynamics but randomly rewires
       the graph edge connections."""
    nx.connected_double_edge_swap(graph,np.floor(proportion*len(graph.nodes())))
    L = nx.laplacian_matrix(graph)
    L = L.todense()
    return -np.dot(L,x)
