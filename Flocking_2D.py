import networkx as nx
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
import time

class Flock_Simulation:
    def __init__(self,
                 velocity_graph,
                 position_dynamics,
                 velocity_dynamics,
                 position_dynamics_args=None,
                 velocity_dynamics_args=None,
                 time_step=0.01,
                 position_graph = None,
                 position_init = None,
                 velocity_init = None,
                 distance_threshold=2):
        if(isinstance(velocity_graph,nx.Graph)):
            self.v_graph = velocity_graph
            self.size = len(self.v_graph)
        else:
            print("Argument Error: velocity_graph and position graph must be type",
                  type(nx.Graph()))
        if (position_graph != None and isinstance(position_graph,nx.Graph)):
            self.s_graph = position_graph

        if(callable(position_dynamics),callable(velocity_dynamics)):
            self.r_dot = position_dynamics
            self.v_dot = velocity_dynamics
            if (position_dynamics_args==None):
                self.r_dot_arg = np.ones(1)
            if (velocity_dynamics_args==None):
                self.v_dot_arg = np.ones(1)
            if (isinstance(position_dynamics_args, type(np.ones(1))) and position_dynamics_args!=None):
                self.r_dot_arg = position_dynamics_args
            else:
                print("Argument Error: position_dynamics_args must be np.ndarray type")

            if (isinstance(position_dynamics_args, type(np.ones(1))) and velocity_dynamics_args != None):
                self.v_dot_arg = velocity_dynamics_args
            else:
                print("Argument Error: velocity_dynamics_args must be np.ndarray type")
        else:
            print("Argument Error: position_dynamics and velocity_dynamics must be functions")
        if(position_init==None):
            self.r = np.zeros((self.size,2))
            for i in range(0,self.size):
                self.r[i] = 5*np.random.randn(2)
            self.s_graph = self.S_graph_pos(self.r)
        if(velocity_init==None):
            self.v = np.zeros((self.size,2))
            for i in range(0,self.size):
                self.v[i] = 5*np.random.randn(2)
        self.dt = time_step
        self.R = distance_threshold

    def S_graph_pos(self,positions,R=2):
        S = nx.Graph()
        if (positions.shape[1] != 2):
            size = (int(positions.shape[1]/2),2)
        else:
            size = positions.shape
        for i,r1 in enumerate(positions.reshape(size)):
            for j,r2 in enumerate(positions.reshape(size)):
                if (sum(r1 == r2) < 2 and norm(r1-r2)<R):
                    S.add_edge(i,j)
                else:
                    S.add_node(i)
        return S

    def velocity_angle_agreement(self,v,tol=1e-6):
        unit = np.zeros(v.shape)
        for i in range(0,len(v)):
            unit[i] = v[i]/norm(v[i])
        ref = np.angle(complex(unit[1,0],unit[1,1]))
        for i in range(0,len(unit)):
            if (np.abs(ref - np.angle(complex(unit[i,0],unit[i,1])))>tol):
                return False
        return True

    def run_sim(self):
        t=0
        start = time.time()
        while self.velocity_angle_agreement(self.v)==False:
            u = self.v.copy()
            self.v = self.v + self.dt*self.v_dot(self.v,self.r,self.v_graph, self.s_graph, *self.v_dot_arg)
            self.r = self.r+self.dt*self.r_dot(u,self.r,self.v_graph, self.s_graph, *self.v_dot_arg)
            self.s_graph = self.S_graph_pos(self.r.self.R)
            t = t+self.dt
        print(time.time()-start)

    def run_sim_switch(self):
        t=0
        p=0.75
        proportion=0.75
        start = time.time()
        while self.velocity_angle_agreement(self.v)==False:
            u = self.v.copy()
            self.v = self.v + self.dt*self.v_dot(self.v,self.r,self.v_graph, self.s_graph, *self.v_dot_arg)
            self.r = self.r+ self.dt*self.r_dot(u,self.r,self.v_graph, self.s_graph, *self.v_dot_arg)
            self.s_graph = self.S_graph_pos(self.r,self.R)
            if(np.random.rand(1)<p):
                nx.connected_double_edge_swap(self.v_graph,np.floor(proportion*self.size))
            t = t+self.dt
        print(time.time()-start)

    def plot(self):
        unit = np.zeros(self.v.shape)
        for i in range(0,len(self.v)):
            unit[i] = self.v[i]/norm(self.v[i])

        for i in range(0,self.size):
            plt.arrow(self.r[i,0],self.r[i,1],unit[i,0],unit[i,1],
                      width=.1,
                      edgecolor='green',
                      facecolor='green')


        nx.draw_networkx(self.s_graph, pos=self.r, edge_color='r', width=5)
        nx.draw_networkx(self.v_graph, pos=self.r, edge_color='black', width=2.5)

        plt.xlim(plt.xlim()[0]-unit[0,0],plt.xlim()[1]+unit[0,0])
        plt.ylim(plt.ylim()[0]-unit[0,1], plt.ylim()[1]+unit[0,1])
