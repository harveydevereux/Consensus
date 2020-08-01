from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import time

import consensus

def test_agree(N=10):
    G = nx.from_numpy_matrix(np.ones((N,N)))

    CS = consensus.ConsensusSimulation(G,
                              consensus.distributed,
                              [nx.laplacian_matrix(G).todense()])

    CS.run_sim()
    assert CS.agreement()

def test_agree_delay(N=10):
    G = nx.from_numpy_matrix(np.ones((N,N)))

    CS = consensus.ConsensusSimulation(G,
                              consensus.distributed,
                              [nx.laplacian_matrix(G).todense()])

    CS.run_sim()
    assert CS.agreement()
