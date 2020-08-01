import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import time
import sys

class ConsensusSimulation:
    """Class to model a general consensus problem
        see DOI: 10.1109/JPROC.2006.887293"""
    def __init__(self,
                 topology,
                 dynamics,
                 dynamics_args,
                 time_step=0.01,
                 x_init=None,
                 convergence_warning=True,
                 delay=0):
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
            if(len(dynamics_args)==1):
                self.f_arg = (dynamics_args,1)
            self.f_arg = dynamics_args
        else:
            print("Argument Error: dynamics must be a function")
        self.dt = time_step
        self.tau = delay
        # set up initial vector to
        # 1,2,3,...,n
        if(not isinstance(x_init, type(np.ones(1))) and x_init==None):
            self.x = np.linspace(1,self.size,self.size)
            self.x = self.x.reshape(self.size,1)
        else:
            self.x = x_init.copy().reshape(self.size,1)
        # The Laplacian matrix, quite the building block
        # for the algorithms
        self.L = nx.laplacian_matrix(self.graph).todense()
        self.X = list()
        self.T = list()
        # connected graph won't converge
        # maybe there's some algorithm that will
        # though...
        self.warn = convergence_warning

        self.d_max = max(np.array(self.graph.degree)[:,1])
        self.tau_max = (np.pi)/(4*self.d_max)

    def disagreement(self):
        """Returns the 'error'/inhomogeneity in the
           decision vector"""
        return 0.5*(np.dot(np.dot(np.transpose(self.x),self.L),self.x)).item(0)

    def agreement(self,tol=1e-6):
        """Test for convergence"""
        if(self.disagreement()<tol):
            return True
        else:
            return False

    def run_sim(self,record_all=False,update_every=1.0):
        """run the core simulation"""
        t=0
        self.x_init = self.x
        self.X = list()
        self.T = list()
        flag = False

        self.X.append(self.x)
        self.T.append(0)
        start = time.time()
        time_since_last_update = 0.0
        progress = 1
        while self.agreement() == False:
            start_it = time.time()
            if(t==0 and self.warn and not nx.is_connected(self.graph)):
                print("Graph not connected, consensus algorithm will probably not converge!")
                print("Simulating to 5 seconds...")
                flag = True
            if(flag and time.time()-start>5):
                break
            # core simulation done here
            # very simple discretisation...
            self.x = self.x+self.dt*self.f(self.x,*self.f_arg)
            # odd way to test for 1,2,3,etc
            # when arg is float
            if (record_all):
                self.X.append(self.x)
                self.T.append(time.time()-start)
            else:
                if (t-np.floor(t)<1e-2):
                    self.X.append(self.x)
                    self.T.append(time.time()-start)
            t = t+self.dt
            end = time.time()-start_it
            time_since_last_update += end
            if time_since_last_update >= update_every:
                sys.stdout.write("\r" + "Iteration: {}, disagreement: {}, time: {}".format(progress,self.disagreement(),time.time()-start))
                sys.stdout.flush()
                time_since_last_update = 0.0
            progress += 1

        print("")
        end = time.time()
        return self.T[-1]

    def run_sim_delay(self,delay=1,runtime=100,update_every=1.0):
        t=0
        self.tau=delay
        self.x_init = self.x
        self.X = list()
        self.T = list()
        flag = False
        for i in range(0,delay+1):
            self.X.append(self.x)
            self.T.append(0)
        start = time.time()
        time_since_last_update = 0.0
        progress = 1
        while self.agreement() == False:
            start_it = time.time()
            if (self.T[-1] > runtime):
                break
            if (t==0 and self.warn and not nx.is_connected(self.graph)):
                print("Graph not connected, consensus algorithm will probably not converge!")
                print("Simulating to 5 seconds...")
                flag = True
            if(flag and time.time()-start>5):
                break
            # core simulation done here
            # very simple discretisation...
            self.x = self.X[-1]
            if (len(self.X)-delay<0):
                pass
            else:
                index = len(self.X)-delay
                self.x = self.X[-1]+self.dt*self.f(self.X[index],*self.f_arg)
            # odd way to test for 1,2,3,etc
            # when arg is float
            self.X.append(self.x)
            self.T.append(time.time()-start)
            t = t+self.dt
            end = time.time()-start_it
            time_since_last_update += end
            if time_since_last_update >= update_every:
                sys.stdout.write("\r" + "Iteration: {}, disagreement: {}, time: {}".format(progress,self.disagreement(),time.time()-start))
                sys.stdout.flush()
                time_since_last_update = 0.0
            progress += 1

        end = time.time()
        return self.T[-1]

    def plot(self, weight_average=False):
        """Show the convergence analysis"""
        if(len(self.X)==0 or len(self.T)==0):
            print("Nothing to plot...")
        x = np.array(self.X)
        for i in range(0,x.shape[1]):
            plt.plot(self.T,x[:,i,0])
        if(weight_average):
            w_i = np.zeros(self.size)
            s = sum(np.array(self.graph.degree)[:,1])
            x = self.x_init
            for i in nx.nodes(self.graph):
                w_i[i] = self.graph.degree(i)/s
                x[i] = x[i]*w_i[i]
            plt.plot(np.linspace(0,self.T[-1],10),np.zeros(10)+sum(x), label="Connected graph consensus: "+str(sum(x)),color='red',marker='s')
        else:
            plt.plot(np.linspace(0,self.T[-1],10),np.zeros(10)+np.mean(self.x_init), label="Connected graph consensus: "+str(round(np.mean(self.x_init),3)),color='red',marker='s')
        plt.grid()
        plt.xlabel("Time (seconds)")
        plt.ylabel("State")
        plt.title("Convergence of consensus algorithm")
        plt.legend()

    def print_delay(self):
        print("Delay in seconds")
        return self.dt*self.tau

    def delay_stable_max(self):
        d = maximum_degree(self.graph)
        return (np.pi)/(4*d[1])
