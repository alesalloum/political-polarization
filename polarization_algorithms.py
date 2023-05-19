'''
    File name: polarization_algorithms.py
    Author: Ali Salloum
    Date created: 01/09/2020
    Date last modified: 11/01/2021
    Python Version: 3.6
'''
import random

import numpy as np
import networkx as nx
import scipy.stats

import heapq
import logging

def get_influencer_nodes(G, cluster1_nodes, cluster2_nodes, n_influencers):
    """
    Returns the top n_influencers number of nodes with highest degree in two given clusters, c1 and c2.

    Args:
        G (networkx.Graph): The graph containing the nodes and edges.
        cluster1_nodes (list): List of nodes in cluster1.
        cluster2_nodes (list): List of nodes in cluster2.
        n_influencers (int): Number of nodes to be returned as influencers from each cluster.

    Returns:
        tuple: Returns two lists of top n_influencers influencers from cluster1 and cluster2 
        respectively based on their degree centrality. If the n_influencers is less than 1, then
        it will be considered as a percentage of the total number of nodes in the respective clusters.
    """ 

    c1_nodes_degrees = ((node, G.degree(node)) for node in cluster1_nodes)
    c2_nodes_degrees = ((node, G.degree(node)) for node in cluster2_nodes)
    
    k_c1 = min(n_influencers, len(cluster1_nodes)) if n_influencers >= 1 else max(1, int(n_influencers * len(cluster1_nodes)))
    k_c2 = min(n_influencers, len(cluster2_nodes)) if n_influencers >= 1 else max(1, int(n_influencers * len(cluster2_nodes)))
    
    c1_influencers = heapq.nlargest(k_c1, c1_nodes_degrees, key=lambda x: x[1])
    c2_influencers = heapq.nlargest(k_c2, c2_nodes_degrees, key=lambda x: x[1])
    
    return [x[0] for x in c1_influencers], [x[0] for x in c2_influencers]

def perform_random_walk(G, left_influencers, right_influencers, starting_node):
    """Performs the random walk and returns the ending side of the walk"""
    
    current_node = starting_node
    ending_side = 0
    absorbed = 0
    
    while not absorbed:
        
        next_node = random.choice([n for n in G.neighbors(current_node)])
        
        if next_node in left_influencers:
            ending_side = "cluster1"
            absorbed = 1
        elif next_node in right_influencers:
            ending_side = "cluster2"
            absorbed = 1
        else:
            current_node = next_node
    
    return ending_side

def random_walk_pol(G, ms, n_influencers, n_sim, n_walks):
    """
    Computes Random Walk Controversy Polarization.

    Parameters
    ----------
    G : networkx graph
        The graph on which to perform the random walks.
    ms : dict
        A dictionary mapping nodes to their political affiliation (0 for left, 1 for right).
    n_influencers : int
        The number of influencer nodes to select on each side.
    n_sim : int
        The number of simulations to run.
    n_walks : int
        The number of random walks to perform in each simulation.

    Returns
    -------
    float
        The average RWC value across all simulations.
    """

    cluster1_nodes = [node for node in ms if ms[node] == 0]
    cluster2_nodes = [node for node in ms if ms[node] == 1]
    
    cluster1_influencers, cluster2_influencers = get_influencer_nodes(G, cluster1_nodes, cluster2_nodes, n_influencers)
    
    rwc_dist = []
    
    for sim in range(n_sim):

        logging.info(f'Running simulation number {sim}')
        #print(f'Running simulation number {sim}')

        c1_c1 = 0
        c2_c1 = 0
        c1_c2 = 0
        c2_c2 = 0

        for _ in range(n_walks):

            starting_side = random.choice(["cluster1", "cluster2"])

            if starting_side == "cluster1":
                which_random_starting_node = random.choice(cluster1_nodes)
            else:
                which_random_starting_node = random.choice(cluster2_nodes)

            ending_side = perform_random_walk(G, cluster1_influencers, cluster2_influencers, which_random_starting_node)

            if (starting_side == "cluster1") and (ending_side == "cluster1"):
                c1_c1 += 1

            elif (starting_side == "cluster2") and (ending_side == "cluster1"):
                c2_c1 += 1

            elif (starting_side == "cluster1") and (ending_side == "cluster2"):
                c1_c2 += 1

            elif (starting_side == "cluster2") and (ending_side == "cluster2"):
                c2_c2 += 1

            else:
                print("Error!")

        e1 = (c1_c1)/(c1_c1 + c2_c1)
        e2 = (c2_c1)/(c1_c1 + c2_c1)
        e3 = (c1_c2)/(c2_c2 + c1_c2)
        e4 = (c2_c2)/(c2_c2 + c1_c2)
        
        rwc = e1*e4 - e2*e3
        rwc_dist.append(rwc)

    rwc_ave = sum(rwc_dist)/len(rwc_dist) 
    
    return rwc_ave    

def krackhardt_ratio_pol(G, ms):
    """Computes EI-Index Polarization"""
    EL = 0
    IL = 0
    
    for e in G.edges:
        s, t = e

        if ms[s] != ms[t]:
            EL += 1
        else:
            IL += 1
            
    return (EL-IL)/(EL+IL)

def extended_krackhardt_ratio_pol(G, ms):
    """Computes Extended EI-Index Polarization"""
    
    block_a = [k for k in ms if ms[k] == 0]
    block_b = [k for k in ms if ms[k] == 1]
    
    n_a = len(block_a)
    n_b = len(block_b)

    c_a = len(G.subgraph(block_a).edges)
    c_b = len(G.subgraph(block_b).edges)
    c_ab = 0
    
    for e in G.edges:
        s, t = e

        if ms[s] != ms[t]:
            c_ab += 1

    B_aa = (c_a)/(n_a*(n_a-1)*0.5)
    B_bb = (c_b)/(n_b*(n_b-1)*0.5)
    B_ab = (c_ab)/(n_a*n_b)
    B_ba = B_ab
        
    return -(B_aa+B_bb-B_ab-B_ba)/(B_aa+B_bb+B_ab+B_ba)

def betweenness_pol(G, ms):
    """Computes Betweenness Centrality Controversy Polarization"""
    dict_eb = nx.edge_betweenness_centrality(G, k = int(0.75*len(G)))
    #n_pivots = min(1000, len(G))
    #dict_eb = nx.edge_betweenness_centrality(G, k=n_pivots)
    
    cut_edges = []
    rest_edges = []
    
    BCC_dist = []

    for e in G.edges():
        s, t = e

        if ms[s] != ms[t]:
            cut_edges += [e]
        else:
            rest_edges += [e]

    cut_ebc = [dict_eb[e] for e in cut_edges]
    rest_ebc = [dict_eb[e] for e in rest_edges]
    
    if len(cut_ebc) <= 1:
        print("Error in the gap!")
        return -1
    
    kernel_for_cut = scipy.stats.gaussian_kde(cut_ebc, "silverman")
    kernel_for_rest = scipy.stats.gaussian_kde(rest_ebc, "silverman")

    BCC = []
    
    for _ in range(10):
        cut_dist = kernel_for_cut.resample(10000)[0]
        rest_dist = kernel_for_rest.resample(10000)[0]

        cut_dist = [max(0.00001, value) for value in cut_dist]
        rest_dist = [max(0.00001, value) for value in rest_dist]

        kl_divergence = scipy.stats.entropy(rest_dist, cut_dist)

        BCCval = 1-2.71828**(-kl_divergence)
        
        BCC.append(BCCval)
        
    return sum(BCC)/len(BCC)

def gmck_pol(G, ms):
    """Computes Boundary Polarization"""
    
    X = set([node for node in ms if ms[node] == 0])
    Y = set([node for node in ms if ms[node] == 1])

    B = []

    for e in G.edges:
        s, t = e

        if ms[s] != ms[t]:
            B.extend([s,t])

    Bset = set(B)
    
    Bx = Bset & X
    By = Bset & Y    

    Ix = X.difference(Bx)
    Iy = Y.difference(By)
    
    I = Ix.union(Iy)

    Bxf = Bx.copy()
    Byf = By.copy()
    
    for u in Bxf:
        connections_of_u = set([n for n in G.neighbors(u)])
        if connections_of_u.issubset(B):
            Bx.remove(u)

    for v in Byf:
        connections_of_v = set([n for n in G.neighbors(v)])
        if connections_of_v.issubset(B):
            By.remove(v)

    B = Bx.union(By)

    summand = []
    for node in B:

        di = nx.cut_size(G, [node], I)
        
        if node in Bx:
            db = nx.cut_size(G, [node], Byf)
            
        else:
            db = nx.cut_size(G, [node], Bxf)
            
        summand.append((di/(di+db)) - 0.5)

    GMCK = (1/(len(B)+0.0001)) * sum(summand)

    return GMCK

def dipole_pol(G, ms):
    """Computes Dipole Polarization"""
    
    left_nodes = [node for node in ms if ms[node] == 0]
    right_nodes = [node for node in ms if ms[node] == 1]
    
    X_top, Y_top = get_influencer_nodes(G, left_nodes, right_nodes, 0.05)
    
    X_top = set(X_top)
    Y_top = set(Y_top)

    dict_polarity = dict.fromkeys(list(G), 0)

    dict_polarity.update(zip(X_top, [-1] * len(X_top)))
    dict_polarity.update(zip(Y_top, [1] * len(Y_top)))

    listeners = set(G.nodes) - X_top - Y_top
    
    polarity = np.asarray(list(dict_polarity.values()))
    polarity_new = np.zeros(len(polarity))

    roundcount = 0
    tol=10**-5
    
    notconverged = len(polarity_new)
    max_rounds = 500

    while notconverged > 0 :

        for node in listeners:

            polarity_new[node] = np.mean([polarity[n] for n in G.neighbors(node)])

        if roundcount < 1:
            polarity_new[list(X_top)] = -1
            polarity_new[list(Y_top)] = 1

        diff = np.abs(polarity - polarity_new)
        notconverged = len(diff[diff>tol])

        polarity = polarity_new.copy()

        if roundcount > max_rounds:

            #print("Maximum rounds achieved")
            break

        roundcount += 1

    #print("Rounds needed: ", roundcount)
    
    n_nodes = len(G)
    n_plus = len(polarity[polarity > 0]) 
    n_minus = n_nodes - n_plus

    delta_A = np.abs((n_plus - n_minus) * (1/(n_nodes)))

    gc_plus = np.mean(polarity[polarity > 0])
    gc_minus = np.mean(polarity[polarity < 0])

    pole_D = np.abs(gc_plus-gc_minus) * (1/2)
    MBLB = (1-delta_A) * pole_D
    
    return MBLB