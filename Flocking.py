import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from graph_utils import *

def rho_h(z,h=0.2):
    if (0 <= z and z < h):
        return 1
    if (h <= z and z <= 1):
        return (1/2)*(1+np.cos(np.pi*(z-h)/(1-h)))
    else:
        return 0

class Flock:
    def __init__(self,
                 acceleration,
                 acceleration_args=None,
                 inter_agent_distance=7,
                 communication_range=None,
                 number_of_agents=50,
                 initial_position=None,
                 initial_velocity=None,
                 time_step=0.01,
                 sigma_norm_epsilon = 0.1,
                 phi_a=5,
                 phi_b=5,
                 bump_function=None,
                 ):
        self.N = number_of_agents
        if(initial_position.any()==None):
            self.Q = np.sqrt(250)*np.random.randn(self.N,2)
        else:
            self.Q = initial_position
        if(initial_velocity.any()==None):
            self.P = (10)*np.random.rand(self.N,2)-1
        else:
            self.P = initial_velocity

        if(callable(acceleration)):
            self.P_dot = acceleration
        else:
            print("Argument Error: acceleration must be a function")
        if(acceleration_args != None):
            if(len(acceleration_args)==1):
                self.args = (acceleration_args,1)
            else:
                self.args = acceleration_args

        self.dt = time_step
        # gets the neighbour graph: edge if ||q_j-q_i||<r
        self.e = sigma_norm_epsilon
        self.d = inter_agent_distance
        self.a = phi_a
        self.b = phi_b
        if(communication_range==None):
            self.r = 1.2*self.d
        else:
            self.r = communication_range

        if(bump_function==None):
            self.rho_h = rho_h
        elif(callable(bump_function)):
            self.rho_h = rho_h
        else:
            print("Argument Error: bump_function must be a function")

        self.G = self.get_net(self.Q)

    def run_sim(self,T=10):
        t=0
        while t<T:
            self.G = self.get_net(self.Q)
            Q = self.Q
            self.Q = self.Q+self.P*self.dt
            self.P = self.P+self.P_dot(Q,self.G)*self.dt
            t = t+self.dt

    def plot(self,
             Graph=True,
             fig_size=8,
             with_labels=False,
             node_size=25,
             width=2.5,
             arrow_width=.25):
        if(Graph):
            p = plt.figure(figsize=(fig_size,fig_size))
            unit = np.zeros(self.P.shape)
            norms = np.zeros(self.N)
            for i in range(0,self.N):
                norms[i] = np.linalg.norm(self.P[i])
                unit[i] = self.P[i]/norms[i]
            rel = np.zeros(self.P.shape)
            for i in range(0,self.N):
                rel[i] = unit[i]*(np.linalg.norm(self.P[i])/max(norms))

            for i in range(0,self.N):
                plt.arrow(self.Q[i,0],self.Q[i,1],rel[i,0],rel[i,1],
                          width=arrow_width,
                          edgecolor='green',
                          facecolor='green')


            nx.draw_networkx(self.G,
                             pos=self.Q,
                             edge_color='black',
                             width=width,
                             node_size=node_size,
                             with_labels=with_labels)

            plt.xlim(plt.xlim()[0]-np.abs(unit[0,0]),plt.xlim()[1]+np.abs(unit[0,0]))
            plt.ylim(plt.ylim()[0]-np.abs(unit[0,1]), plt.ylim()[1]+np.abs(unit[0,1]))
            return p
        else:
            p = plt.figure(figsize=(fig_size,fig_size))
            plt.scatter(self.Q[:,0],self.Q[:,1])
            return p



    def get_net(self,Q):
        """Draw a graph based on a position matrix with dimension Nxd"""
        G = nx.Graph()
        for i in range(0,self.N):
            for j in range(0,self.N):
                if (np.linalg.norm(Q[i,:]-Q[j,:])<self.r):
                    if(not G.has_edge(i,j)):
                        G.add_edge(i,j)
        return(G)


    def sigma_norm(self,z):
        return (1/self.e)*(np.sqrt(1+self.e*(np.linalg.norm(z))**2)-1)

    def sigma_grad(self,z):
        return z/(1+self.e*self.sigma_norm(z))

    def sigma_1(self,z):
        return z/(np.sqrt(1+z**2))

    def phi(self,z):
        c = np.abs(self.a-self.b)/(np.sqrt(4*self.a*self.b))
        return (1/2)*((self.a+self.b)*self.sigma_1(z+c)+(self.a-self.b))

    def phi_alpha(self,z):
        r_alpha = self.sigma_norm(self.r)
        d_alpha = self.sigma_norm(self.d)
        return self.rho_h(z/r_alpha)*self.phi(z-d_alpha)
