from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import time

import consensus

def always_aggree(N=10):
    Q = np.sqrt(2500)*np.random.randn(N,2)
    P = (5)*np.random.rand(N,2)-1

    FS = consensus.Flock(number_of_agents=N,
               initial_position=Q,
               initial_velocity=P,
               inter_agent_distance=7,
               gamma_agent=True)

    FS.run_sim(T=10, save_data=False)
    assert FS.agreement()
