import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import time

class Consensus_Simulation:
    """Class to model a general consensus problem
        see DOI: 10.1109/JPROC.2006.887293"""
    def __init__(self,
                 topology,
                 dynamics,
                 dynamics_args,
                 time_step=0.01,
                 x_init=None,
                 convergence_warning=True):
        # check arguments are of the
        # correct form
        if(isinstance(topology,nx.Graph)):
            self.graph = topology
            self.size = len(self.graph)
        else:
            print("Argument Error: topology must be type"
                 , type(nx.Graph()))
        if(callable(dynamics)):
            self.f = dynamics
            self.f_arg = dynamics_args
        else:
            print("Argument Error: dynamics must be a function")
        self.dt = time_step
        # set up initial vector to
        # 1,2,3,...,n
        if(x_init==None):
            self.x_init = np.linspace(1,self.size,self.size)
            self.x_init=self.x_init.reshape(self.size,1)
        self.x = self.x_init.copy()
        # The Laplacian matrix, quite the building block
        # for the algorithms
        self.L = nx.laplacian_matrix(self.graph).todense()
        self.X = list()
        self.T = list()
        # connected graph won't converge
        # maybe there's some algorithm that will
        # though...
        self.warn = convergence_warning

    def disagreement(self):
        """Returns the 'error'/in-homogeneity in the
           decision vector"""
        return 0.5*(np.dot(np.dot(np.transpose(self.x),self.L),self.x)).item(0)

    def agreement(self,tol=1e-6):
        """Test for convergence"""
        if(self.disagreement()<tol):
            return True
        else:
            return False

    def run_sim(self):
        """run the core simulation"""
        t=0
        # could be re-set
        self.x = self.x_init.copy()
        self.X = list()
        self.T = list()
        flag = False

        self.X.append(self.x_init)
        self.T.append(0)
        start = time.time()
        while self.agreement() == False:
            if(t==0 and self.warn and not nx.is_connected(self.graph)):
                print("Graph not connected, consensus algorithm will probably not converge!")
                print("Simulating to 5 seconds...")
                flag = True
            if(flag and time.time()-start>5):
                break
            # core simulation done here
            # very simple discretisation...
            self.x = self.x+self.dt*self.f(self.x,self.f_arg)
            # odd way to test for 1,2,3,etc
            # when arg is float
            if (t-np.floor(t)<1e-2):
                self.X.append(self.x)
                self.T.append(time.time()-start)
            t = t+self.dt
        end = time.time()

    def plot(self):
        """Show the convergence analysis"""
        if(len(self.X)==0 or len(self.T)==0):
            print("Nothing to plot...")
        x = np.array(self.X)
        for i in range(0,x.shape[1]):
            plt.plot(self.T,x[:,i,0])

        plt.plot(np.linspace(0,self.T[-1],10),np.zeros(10)+np.mean(self.x_init), label="Connected graph consensus: "+str(np.mean(self.x_init)),color='red',marker='s')
        plt.grid()
        plt.xlabel("Time (seconds)")
        plt.ylabel("State")
        plt.title("Convergence of consensus algorithm")
        plt.legend()
