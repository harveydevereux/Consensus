from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import time

import consensus

def test_agree(N=5):
    Q = np.ones((N,2))
    P = (5)*np.ones((N,2))-1

    FS = consensus.Flock(number_of_agents=N,
               initial_position=Q,
               initial_velocity=P,
               inter_agent_distance=7,
               gamma_agent=True)

    FS.run_sim(T=10, save_data=False)
    assert FS.velocity_angle_agreement(FS.P)
