import numpy as np
import networkx as nx

from graph_utils import normalized_laplacian_matrix as norm_L

def distributed(x,L, *args):
    """Most basic distributed consensus algorithm
       It's actually gradient descent of 1/2(xLx)"""
    return -np.dot(L,x)

def Fax_Murray(x, G):
    return -np.dot(norm_L(G),x)

def distributed_random_topology(x,graph, proportion=0.5, *args):
    """Same as dynamics but randomly rewires
       the graph edge connections."""
    nx.connected_double_edge_swap(graph,np.floor(proportion*len(graph.nodes())))
    L = nx.laplacian_matrix(graph)
    L = L.todense()
    return -np.dot(L,x)

def P_dot(Q,G):
    """For Flock class mostly"""
    def rho_h(z,h=0.2):
        if (0 <= z and z < h):
            return 1
        if (h <= z and z <= 1):
            return (1/2)*(1+np.cos(np.pi*(z-h)/(1-h)))
        else:
            return 0

    def sigma_norm(z,epsilon=0.1):
        return (1/epsilon)*(np.sqrt(1+epsilon*(np.linalg.norm(z))**2)-1)

    def sigma_grad(z,epsilon=0.1):
        return z/(1+epsilon*sigma_norm(z,epsilon))

    def sigma_1(z):
        return z/(np.sqrt(1+z**2))

    def phi(z,a=5,b=5):
        c = np.abs(a-b)/(np.sqrt(4*a*b))
        return (1/2)*((a+b)*sigma_1(z+c)+(a-b))

    def phi_alpha(z,d,r):
        r_alpha = sigma_norm(r)
        d_alpha = sigma_norm(d)
        return rho_h(z/r_alpha)*phi(z-d_alpha)

    u = np.zeros((len(G),2))
    A = nx.adjacency_matrix(G).todense()
    for i in G.nodes():
        for j in G.neighbors(i):
            u[i] = u[i] + phi_alpha(sigma_norm(Q[j,:]-Q[i,:]),d,r)*(sigma_grad(Q[j,:]-Q[i,:]))
            u[i] = u[i] + A[i,j]*(Q[j,:]-Q[i,:])
    return u
