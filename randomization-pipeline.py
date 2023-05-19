import logging
import os
import json

import networkx as nx

import pymetis
import numpy as np

import polarization_algorithms as pol

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("network_name")
#parser.add_argument("year")
args = parser.parse_args()

network_name = args.network_name
year = "2023"

randomization_strategies = {"zerok": 0, "onek": 1, "twok": 0}
n_samples = 5

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def prepare_network(filename):

    # Load the graph from the GraphML file
    G = nx.read_graphml(filename, node_type=int)

    # Remove self-loops
    G.remove_edges_from(nx.selfloop_edges(G))

    # Get the giant component
    giant_component = max(nx.connected_components(G), key=len)

    # Create a new graph with only the giant component
    GC = G.subgraph(giant_component).copy()

    return GC

def partition_metis(G):

    def get_adjacency_dict(G):
        adj_list = {}
        for node in G.nodes:
            neighbors = list(G.neighbors(node))
            adj_list[node] = neighbors
        return adj_list

    def get_partition(G):
        adj_dict = get_adjacency_dict(G)
        adj_list = [np.asarray(neighs) for neighs in adj_dict.values()]
        n_cuts, membership = pymetis.part_graph(nparts = 2, adjacency = adj_list, options = pymetis.Options(ufactor=400, niter=100, contig=True))
        membership = dict(zip(adj_dict.keys(), membership))
        return n_cuts, membership
    
    return get_partition(G)

def compute_polarization(R, network_name):

    def get_giant_component(G):
        # Get the giant component
        giant_component = max(nx.connected_components(G), key=len)

        # Create a new graph with only the giant component
        GC = G.subgraph(giant_component).copy()
        GC_int = nx.convert_node_labels_to_integers(GC)
        return GC_int

    #logging.info(f"Starting randomization pipeline for {network_name}...")

    #R = prepare_network(R)
    R = get_giant_component(R)

    _, ms = partition_metis(R)
    logging.info("Network has been partitioned with METIS")

    logging.info("Measuring RWC polarization.")
    n_sim, n_walks = 10, int(1e4)
    rwc_metis = pol.random_walk_pol(R, ms, 10, n_sim, n_walks)

    logging.info("Measuring ARWC polarization.")
    arwc_metis = pol.random_walk_pol(R, ms, 0.01, n_sim, n_walks)

    logging.info("Measuring EI polarization.")
    ei_metis = -1*pol.krackhardt_ratio_pol(R, ms)

    logging.info("Measuring AEI polarization.")
    extei_metis = -1*pol.extended_krackhardt_ratio_pol(R, ms)

    logging.info("Measuring MOD polarization.")
    c1_metis = [node for node in ms if ms[node] == 0]
    c2_metis = [node for node in ms if ms[node] == 1]
    mod_metis = nx.community.modularity(R, [c1_metis, c2_metis])
 
    logging.info("Measuring EBC polarization.")
    ebc_metis = pol.betweenness_pol(R, ms)
   
    logging.info("Measuring GMCK polarization.")
    gmck_metis = pol.gmck_pol(R, ms)

    logging.info("Measuring MBLB polarization.")
    mblb_metis = pol.dipole_pol(R, ms)

    logging.info(f"Polarization pipeline for {network_name} has ended.")

    infopack = [rwc_metis, arwc_metis, ebc_metis, gmck_metis, mblb_metis, mod_metis, ei_metis, extei_metis]

    return infopack

def zerok(G, n_samples):
    
    n, m = len(G.nodes), len(G.edges)
    
    buffer = []
    for i in range(n_samples):
        logging.info(f"Processing sample {i}")
        R = nx.gnm_random_graph(n, m)
        buffer.append(compute_polarization(R, network_name=network_name))
        
    results_averaged = np.mean(buffer, axis=0)
    results_errors = np.std(buffer, axis=0)
    
    return [results_averaged, results_errors]

def onek(G, n_samples):
    
    degree_sequence = [d for v, d in G.degree()]
    
    buffer = []
    for i in range(n_samples):
        logging.info(f"Processing sample {i}")
        R = nx.Graph(nx.configuration_model(degree_sequence))
        R.remove_edges_from(nx.selfloop_edges(R))
        buffer.append(compute_polarization(R, network_name=network_name))
        
    results_averaged = np.mean(buffer, axis=0)
    results_errors = np.std(buffer, axis=0)
    
    return [results_averaged, results_errors]

def twok(G, n_samples):
    
    G.remove_edges_from(nx.selfloop_edges(G))
    degree_sequence = [d for v, d in G.degree()]
    deg_dict = dict(zip(G.nodes(), degree_sequence))

    nx.set_node_attributes(R, deg_dict, "degree")
    joint_degrees = nx.attribute_mixing_dict(G, "degree")
    
    buffer = []
    for i in range(n_samples):
        print("Processing sample number: ", i+1)
        R = nx.joint_degree_graph(joint_degrees)
        buffer.append(compute_polarization(R))
        
    results_averaged = np.mean(buffer, axis=0)
    results_errors = np.std(buffer, axis=0)
    
    return [results_averaged, results_errors]

def run_pipeline():

    logging.info(f"Starting randomization pipeline for {network_name}...")

    filename = f"./rich-networks/{year}/RICH_{network_name}_{year}_NET.graphml"
    observed_G = prepare_network(filename=filename)

    randomized_pol_dict = dict()

    if randomization_strategies["zerok"]:
        ave, std = zerok(observed_G, n_samples)
        randomized_pol_dict["zerok"] = {"ave": list(ave), "std": list(std)}

    if randomization_strategies["onek"]:
        ave, std = onek(observed_G, n_samples)
        randomized_pol_dict["onek"] = {"ave": list(ave), "std": list(std)}
    
    if randomization_strategies["twok"]:
        ave, std = twok(observed_G, n_samples)
        randomized_pol_dict["twok"] = {"ave": list(ave), "std": list(std)}

    randomized_pol_dict["mapping"] = "rwc, arwc, ebc, gmck, mblb, mod, ei, extei"

    os.makedirs("randomized_polarization_scores", exist_ok = True) 
    with open(f'./randomized_polarization_scores/{network_name}_{year}_randomized_pol.json', 'w') as fp:
        json.dump(randomized_pol_dict, fp, indent=2)

if __name__ == "__main__":
    run_pipeline()
