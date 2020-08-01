import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

def maximum_degree(graph):
    "return (node,degree) for node with max degree in graph"
    m = (0,0)
    for d in graph.degree():
        if (d[1] > m[1]):
            m = d
    return m

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

def fiedler_number(G):
    L = nx.laplacian_matrix(G).todense()
    e = np.linalg.eigvals(L)
    rm = np.argmin(e)
    e = np.delete(e,rm)
    return min(np.real(e))

def get_net(Q):
    """Draw a graph based on a position matrix with dimension Nxd"""
    G = nx.Graph()
    for i in range(0,N):
        for j in range(0,N):
            if (np.linalg.norm(Q[i,:]-Q[j,:])<r):
                if(not G.has_edge(i,j)):
                    G.add_edge(i,j)
    return(G)

def normalized_adjacency_matrix(G):
    A = nx.adjacency_matrix(G).todense().astype(float)
    for i in nx.nodes(G):
        d_i = G.degree(i)
        for j in G.neighbors(i):
            A[i,j]=A[i,j]*(1/d_i)
    return A

def normalized_laplacian_matrix(G):
    A = normalized_adjacency_matrix(G)
    return np.eye(A.shape[0],A.shape[1])-A

def degree_matrix(G):
    D = np.zeros((len(G),len(G)))
    for i in nx.nodes(G):
        D[i,i] = G.degree(i)
    return D.astype(float)

def Q_l_laplacian(G):
    D = degree_matrix(G)
    I = np.eye(D.shape[0],D.shape[1])
    B = np.linalg.inv(I+D)
    A = nx.adjacency_matrix(G).todense().astype(float)
    return I-np.dot(B,(I+A))

# for weighted average consensus sims
# of form Kx^dot = -Lx
def get_weight_matrix(N):
    K = np.eye(N,N)
    for i in range(0,K.shape[0]):
        for j in range(0,K.shape[1]):
            if (i==j):
                if (np.random.rand()>0.5):
                    K[i,j] = K[i,j]+np.random.rand()
    return K

def weight_vector_to_matrix(v):
    K = np.zeros((v.shape[0],v.shape[0]))
    for i in range(0,v.shape[0]):
        for j in range(0,v.shape[0]):
            if (i==j):
                K[i,j] = v[i]
    return K

def vector_dist(v):
    p = np.zeros(len(np.unique(v)))
    for i,val in enumerate(np.unique(v)):
        p[i] = len(v[np.where(v==val)])
    return p/sum(p)
