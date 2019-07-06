import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import animation
import random
# from graph_utils import *
import time
# %matplotlib inline
import time
random.seed(8888)

fig_size = 8
p,ax = plt.subplots(figsize=(fig_size,fig_size))

def rho_h(z,h=0.2):
    if (0 <= z and z < h):
        return 1
    if (h <= z and z <= 1):
        return (1/2)*(1+np.cos(np.pi*(z-h)/(1-h)))
    else:
        return 0

class Flock:
    """Implements the flocking framework as proposed in
    [3] R. Olfati-Saber, "Flocking for multi-agent dynamic systems:
    algorithms and theory," in IEEE Transactions on Automatic Control,
    vol. 51, no. 3, pp. 401-420, March 2006. doi: 10.1109/TAC.2005.864190"""
    def __init__(self,
                 acceleration=None,
                 acceleration_args=None,
                 inter_agent_distance=7,
                 communication_range=None,
                 number_of_agents=50,
                 initial_position=None,
                 initial_velocity=np.zeros((50,2)),
                 time_step=0.03,
                 sigma_norm_epsilon = 0.1,
                 phi_a=5,
                 phi_b=5,
                 bump_function=None,
                 gamma_agent=False,
                 gamma_agent_Cq=4.5,
                 # gamma_agent_Cp=1,
                 # add new three factors
                 arfa_agent_Cq = 2.5,
                 # for arfa-agent-num = 50 set beda_agent_Cq : 16 or it will have some arfa-agent fly into obstacle
                 beda_agent_Cq = 8,
                 beda_agent=True,
                 obstacles = np.array([[100,20,10],[110,60,4],[120,40,2],[130,-20,5],[150,40,5],[160,0,3]])
                 ):
        self.N = number_of_agents
        self.M_s = obstacles
        if(isinstance(initial_position,type(np.ones(1)))):
            self.Q = initial_position
        elif(initial_position==None):
            self.Q = np.sqrt(25)*np.random.randn(self.N,2)
        if(isinstance(initial_velocity,type(np.ones(1)))):
            self.P = initial_velocity
        elif(initial_velocity==None):
            self.P = (10)*np.random.rand(self.N,2)-1
        self.gamma_agent=gamma_agent
        self.beda_agent = beda_agent
        # set gamma-agent
        if(gamma_agent):
            self.p = np.array([0.5,0])
            self.q = np.array([200,30])
            self.gamma_C_q = gamma_agent_Cq
            self.gamma_C_p = 2 * np.sqrt(gamma_agent_Cq)

        if(beda_agent):
            self.arfa_C_q = arfa_agent_Cq
            self.arfa_C_p = 2 * np.sqrt(arfa_agent_Cq)
            self.beda_C_q = beda_agent_Cq
            self.beda_C_p = 2 * np.sqrt(beda_agent_Cq)
            self.Q_beda = np.array([])
            self.P_beda = np.array([])

        if(callable(acceleration)):
            self.P_dot = acceleration
            if(acceleration_args != None):
                if(len(acceleration_args)==1):
                    self.args = (acceleration_args,1)
                else:
                    self.args = acceleration_args

        self.dt = time_step
        # gets the neighbour graph: edge if ||q_j-q_i||<r
        self.e = sigma_norm_epsilon
        self.d = inter_agent_distance
        self.d_obstacle = 0.6*self.d
        self.a = phi_a
        self.b = phi_b
        if(communication_range==None):
            self.r = 1.2*self.d
            self.r_obstacle = 0.6*self.r
        else:
            self.r = communication_range

        if(bump_function==None):
            self.rho_h = rho_h
        elif(callable(bump_function)):
            self.rho_h = rho_h
        else:
            print("Argument Error: bump_function must be a function")

        self.G = self.get_net(self.Q)
        self.get_beda_position(self.Q,self.P)
        self.get_net_obstacle(self.Q)

    def run_sim(self,T=10,save_data=False,to_agreement=False):
        if(save_data):
            self.Q_sim = []
            self.P_sim = []
            self.sim_time = []
            self.p_sim = []
            self.q_sim = []
            self.Q_beda_sim = []
            self.P_beda_sim = []
        t=0
        start = time.time()
        while t<T:
            self.G = self.get_net(self.Q)
            self.get_net_obstacle(self.Q)
            Q = self.Q.copy()
            P = self.P.copy()
            # find position and velocity of beda-agent
            self.get_beda_position(Q,P)
            # in the short it is uniform linear motion
            # so it use this equation: s = vt ; v = at
            self.Q = self.Q+self.P*self.dt
            print(t)
            # if t > 16.91:
            #     print(t)
            self.P = self.P+self.P_dot(Q,P)*self.dt
            for i in range(self.N):
                self.P[i] = self.P[i]+self.f_gamma(Q[i],P[i])*self.dt
            self.P = self.P+self.P_beda_dot(Q,P)*self.dt
            self.q = self.q + self.p*self.dt
            if(save_data):
                self.sim_time.append(time.time()-start)
                self.Q_sim.append(self.Q)
                self.P_sim.append(self.P)
                self.p_sim.append(self.p)
                self.q_sim.append(self.q)
                self.Q_beda_sim.append(self.Q_beda)
                self.P_beda_sim.append(self.P_beda)
            if(to_agreement and self.velocity_angle_agreement(self.P)):
                t = T
            elif (to_agreement):
                t = t+self.dt
                T = T+self.dt
            else:
                t = t+self.dt

    def plot_time_series(self,
                         time_step_plot,
                         with_labels=False,
                         # fig_size=8,
                         node_size=25,
                         width=2.5,
                         arrow_width=.25):

        # p = plt.figure(figsize=(fig_size,fig_size))
        ax.clear()
        unit = np.zeros(self.P_sim[time_step_plot].shape)
        norms = np.zeros(self.N)
        # Velocity vector normalization
        for i in range(0,self.N):
            norms[i] = np.linalg.norm(self.P_sim[time_step_plot][i])
            unit[i] = self.P_sim[time_step_plot][i]/norms[i]
        rel = np.zeros(self.P_sim[time_step_plot].shape)
        # Velocity vector arrow normalization to show the length of arrow
        for i in range(0,self.N):
            rel[i] = unit[i]*(np.linalg.norm(self.P_sim[time_step_plot][i])/max(norms))

        for i in range(0,self.N):
            plt.arrow(self.Q_sim[time_step_plot][i,0],self.Q_sim[time_step_plot][i,1],rel[i,0],rel[i,1],
                      width=arrow_width,
                      edgecolor='green',
                      facecolor='green')
        self.G = self.get_net(self.Q_sim[time_step_plot])
        # print(self.G.neighbors(4))
        G = self.G.copy()
        Q = self.Q_sim[time_step_plot].copy()
        node_colors = ['red']
        self.plot_obstacles()
        q_sim = self.q_sim
        # print(q_sim[time_step_plot],type(q_sim[time_step_plot]))
        if(self.gamma_agent):
            rel_gamma = self.p_sim[time_step_plot]/max(norms)
            plt.arrow(self.q_sim[time_step_plot][0],self.q_sim[time_step_plot][1],rel_gamma[0],rel_gamma[1],
                      width=arrow_width,
                      edgecolor='blue',
                      facecolor='blue')
            G.add_node(self.N)
            node_colors = []
            for node in G:
                if node < self.N:
                    node_colors.append('red')
                if node == self.N:
                    node_colors.append('orange')
            Q = np.append(Q,self.q_sim[time_step_plot]).reshape(self.N+1,2)
        # add beda-agent position and velocity
        self.get_beda_position(self.Q, self.P)
        if(self.beda_agent):
            rel_beda = self.P_beda_sim[time_step_plot]/max(norms)
            for i in range(self.N*len(self.M_s)):
                plt.arrow(self.Q_beda_sim[time_step_plot][i,0], self.Q_beda_sim[time_step_plot][i,1], rel_beda[i,0], rel_beda[i,1],
                          width=arrow_width,
                          edgecolor='yellow',
                          facecolor='yellow')
            for i in range(self.N * len(self.M_s)):
                G.add_node(i+self.N+1)
            for node in G:
                if node > self.N:
                    node_colors.append('black')
            Q = np.append(Q,self.Q_beda_sim[time_step_plot]).reshape(self.N * (len(self.M_s) + 1) + 1, 2)

        nx.draw_networkx(G,
                         pos=Q,
                         node_color=node_colors,
                         edge_color='black',
                         width=width,
                         node_size=node_size,
                         with_labels=with_labels)
        # plt.autoscale(enable=True)
        # plt.xlim(plt.xlim()[0] - np.abs(unit[0, 0]), plt.xlim()[1] + np.abs(unit[0, 0]))
        # plt.ylim(plt.ylim()[0] - np.abs(unit[0, 1]), plt.ylim()[1] + np.abs(unit[0, 1]))
        plt.tick_params(axis='both', which='both', bottom=True, left=True, labelbottom=True, labelleft=True)
        min_axes = min(min(self.Q_sim[time_step_plot][:,0]), min(self.Q_sim[time_step_plot][:,1]))
        max_axes = max(max(self.Q_sim[time_step_plot][:,0]), max(self.Q_sim[time_step_plot][:,1]))
        plt.xlim(min_axes - 4, max_axes + 4)
        plt.ylim(min_axes - 4, max_axes + 4)
        plt.title("t = %ss" %(time_step_plot/100), fontsize=25)
        # return p

    def plot_obstacles(self):
        if(self.beda_agent):
            for i in range(len(self.M_s)):
                c = plt.Circle((self.M_s[i,0], self.M_s[i,1]), self.M_s[i,2], facecolor="red",edgecolor="green")
                plt.gca().add_patch(c)
        # plt.show()

    def get_beda_position(self,Q,P):
        len_obstacle = len(self.M_s)
        self.Q_beda = np.array([])
        self.P_beda = np.array([])
        for i in range(0, self.N):
            for k in range(len(self.M_s)):
                u_beda = self.M_s[k,2] / np.linalg.norm(Q[i,:] - self.M_s[k,0:2])
                a_k = (Q[i,:] - self.M_s[k,0:2]) / np.linalg.norm(Q[i,:] - self.M_s[k,0:2])
                P_k = np.eye(2) - a_k.reshape((2,1))*a_k
                Q_beda_temp = u_beda * Q[i,:] + (1-u_beda) * self.M_s[k,0:2]
                P_beda_temp = u_beda * P_k @ P[i,:]
                self.Q_beda = np.append(self.Q_beda, Q_beda_temp)
                self.P_beda = np.append(self.P_beda, P_beda_temp)
        self.Q_beda = self.Q_beda.reshape(len_obstacle*self.N, 2)
        self.P_beda = self.P_beda.reshape(len_obstacle * self.N, 2)

    def get_net(self,Q):
        """Draw a graph based on a position matrix with dimension Nxd"""
        G = nx.Graph()
        for i in range(0,self.N):
            for j in range(0,self.N):
                if (np.linalg.norm(Q[i,:]-Q[j,:])<self.r):
                    if(not G.has_edge(i,j)):
                        G.add_edge(i,j)
        return(G)

    def get_net_obstacle(self,Q):
        len_obstacle = len(self.M_s)
        for i in range(0, self.N):
            for k in range(len_obstacle):
                if (np.linalg.norm(self.Q_beda[(i*len_obstacle + k),:] - Q[i,:])<self.r_obstacle):
                    if (not self.G.has_edge(i,(i*len_obstacle + k + self.N + 1))):
                        self.G.add_edge(i,(i*len_obstacle + k + self.N + 1))

    def P_dot(self,Q,P):
        """For Flock class mostly"""
        u = np.zeros((self.N,2))
        A = self.get_spatial_adjacency(Q)
        for i in self.G.nodes():
            for j in self.G.neighbors(i):
                if (j < self.N) & (i < self.N):
                    u[i] = u[i] + self.arfa_C_q * self.phi_alpha(self.sigma_norm(Q[j,:]-Q[i,:]))*(self.sigma_grad(Q[j,:]-Q[i,:]))
                    u[i] = u[i] + self.arfa_C_p * A[i,j]*(P[j,:]-P[i,:])
        return u

    def P_beda_dot(self,Q,P):
        """caculate the position and velocity of beda-agent"""
        u = np.zeros((self.N,2))
        B = self.get_spatial_adjacency_beda(Q)
        len_obstacle = len(self.M_s)
        for i in self.G.nodes():
            if i < self.N:
                for k in self.G.neighbors(i):
                    if (k >= (self.N + 1 + i * len_obstacle)) & (k < (self.N + 1 + (i+1) * len_obstacle)):
                        print(k,i)
                        u[i] = u[i] + self.beda_C_q * self.phi_alpha_beda(self.sigma_norm(self.Q_beda[k-self.N-1,:] - Q[i, :])) * \
                            (self.sigma_grad(self.Q_beda[k-self.N-1,:] - Q[i, :]))
                        u[i] = u[i] + self.beda_C_p * B[(k-self.N-1)//len_obstacle, (k-self.N-1)%len_obstacle] * (self.P_beda[k-self.N-1,:] - P[i, :])
        return u

    def get_spatial_adjacency_beda(self,Q):
        d_beda = self.sigma_norm(self.d_obstacle)
        B = np.zeros((self.N, len(self.M_s)))
        len_obstacle = len(self.M_s)
        for i in range(0,self.N):
            for k in range(len_obstacle):
                B[i,k] = self.rho_h(self.sigma_norm(self.Q_beda[(i*len_obstacle + k)] - Q[i])/d_beda)
        return B

    def get_spatial_adjacency(self,Q):
        r_alpha = self.sigma_norm(self.r)
        A = np.zeros((self.N,self.N))
        for i in range(0,self.N):
            for j in range(0,self.N):
                if (i != j):
                    A[i,j] = self.rho_h(self.sigma_norm(Q[j]-Q[i])/r_alpha)
        return A

    def sigma_norm(self,z):
        return (1/self.e)*(np.sqrt(1+self.e*(np.linalg.norm(z))**2)-1)

    def sigma_grad(self,z):
        return z/(1+self.e*self.sigma_norm(z))

    def sigma_1(self,z):
        return z/(np.sqrt(1+z**2))

    def sigma_1_all(self,z):
        for i in range(len(z)):
            z[i] = z[i] / (np.sqrt(1 + np.linalg.norm(z[i]) ** 2))
        return z

    def phi(self,z):
        c = np.abs(self.a-self.b)/(np.sqrt(4*self.a*self.b))
        return (1/2)*((self.a+self.b)*self.sigma_1(z+c)+(self.a-self.b))

    def phi_alpha(self,z):
        r_alpha = self.sigma_norm(self.r)
        d_alpha = self.sigma_norm(self.d)
        return self.rho_h(z/r_alpha)*self.phi(z-d_alpha)

    def phi_alpha_beda(self,z):
        d_beda = self.sigma_norm(self.d_obstacle)
        return self.rho_h(z/d_beda,h=0.9)*(self.sigma_1(z-d_beda)-1)

    def f_gamma(self,Q,P):
        return -self.gamma_C_q*self.sigma_1_all(Q-self.q)-self.gamma_C_p*(P-self.p)

    def velocity_angle_agreement(self,v,tol=1e-6):
        unit = np.zeros(v.shape)
        for i in range(0,len(v)):
            unit[i] = v[i]/np.linalg.norm(v[i])
        ref = np.angle(complex(unit[1,0],unit[1,1]))
        for i in range(0,len(unit)):
            if (np.abs(ref - np.angle(complex(unit[i,0],unit[i,1])))>tol):
                return False
        return True

    def speed_agreement(self,v,tol=1e-6):
        for i in range(0,len(v)):
            if np.linalg.norm(v[1]-v[i])>tol:
                return False
        return True

def main():
    # could also leave Q and P for a default random initialisation
    N = 20
    Q = np.random.rand(N, 2)
    for i in range(N):
        Q[i] = random.sample(range(-40,80),1) * Q[i]
    # P = np.zeros((N,2))
    P = (5) * np.random.rand(N, 2) - 1
    # ser obstacle matrix M_s
    M_s = np.array([[100,20,10],[110,60,4],[120,40,2],[130,-20,5],[150,40,5],[160,0,3]])

    FS = Flock(number_of_agents=N,
               initial_position=Q,
               initial_velocity=P,
               inter_agent_distance=7,
               obstacles=M_s,
               gamma_agent=True)

    # p = FS.plot(arrow_width=0.25)
    # plt.title("Initial state for flock following F2", fontsize=25)
    time_period = 150
    time_frames = time_period / FS.dt
    FS.run_sim(T=time_period, save_data=True)
    ani = animation.FuncAnimation(p, FS.plot_time_series, int(time_frames),interval=0.1)
    # ani.save("flock-avoid.gif", writer=animation.PillowWriter(fps=128))
    # ani.save('flock-avoid.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
    plt.rcParams['animation.ffmpeg_path'] = 'D:\\python\\ffmpeg\\bin\\ffmpeg'
    writer = animation.FFMpegWriter(fps=60, metadata=dict(artist='Me'), bitrate=1800)
    ani.save('../results/Flocking-with-obstacle-avoiding.mp4',writer = writer)
    plt.show()
    print('simlate for T=5 ends!')



if __name__ == "__main__":
    main()