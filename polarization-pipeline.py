import logging
import argparse
import json
import os

import numpy as np
import networkx as nx
import pymetis

import polarization_algorithms as pol

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("network_name")
args = parser.parse_args()

network_name = args.network_name
year = "2023"

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

def partition_metis(graph):

    def get_adjacency_dict(graph):
        adj_list = {}
        for node in graph.nodes:
            neighbors = list(graph.neighbors(node))
            adj_list[node] = neighbors
        return adj_list

    def get_partition(graph):
        adj_dict = get_adjacency_dict(graph)
        adj_list = [np.asarray(neighs) for neighs in adj_dict.values()]
        n_cuts, membership = pymetis.part_graph(nparts = 2, adjacency = adj_list, options = pymetis.Options(ufactor=400, niter=100, contig=True))
        membership = dict(zip(adj_dict.keys(), membership))
        return n_cuts, membership
    
    return get_partition(graph)

def run_pipeline():

    logging.info(f"Starting polarization pipeline for {network_name}...")

    filename = f"./rich-networks/{year}/RICH_{network_name}_{year}_NET.graphml"
    G = prepare_network(filename=filename)
    pol_metrics = dict()

    _, ms = partition_metis(G)
    logging.info("Network has been partitioned with METIS")

    logging.info("Measuring RWC polarization.")
    n_sim, n_walks = 10, int(1e4)
    rwc_metis = pol.random_walk_pol(G, ms, 10, n_sim, n_walks)

    logging.info("Measuring ARWC polarization.")
    arwc_metis = pol.random_walk_pol(G, ms, 0.01, n_sim, n_walks)

    logging.info("Measuring EI polarization.")
    ei_metis = -1*pol.krackhardt_ratio_pol(G, ms)

    logging.info("Measuring AEI polarization.")
    extei_metis = -1*pol.extended_krackhardt_ratio_pol(G, ms)

    logging.info("Measuring MOD polarization.")
    c1_metis = [node for node in ms if ms[node] == 0]
    c2_metis = [node for node in ms if ms[node] == 1]
    mod_metis = nx.community.modularity(G, [c1_metis, c2_metis])
 
    logging.info("Measuring EBC polarization.")
    ebc_metis = pol.betweenness_pol(G, ms)
   
    logging.info("Measuring GMCK polarization.")
    gmck_metis = pol.gmck_pol(G, ms)

    logging.info("Measuring MBLB polarization.")
    mblb_metis = pol.dipole_pol(G, ms)

    logging.info(f"Polarization pipeline for {network_name} has ended.")

    pol_metrics["rwc_metis"] = rwc_metis
    pol_metrics["arwc_metis"] = arwc_metis
    pol_metrics["ei_metis"] = ei_metis
    pol_metrics["extei_metis"] = extei_metis
    pol_metrics["mod_metis"] = mod_metis
    pol_metrics["ebc_metis"] = ebc_metis
    pol_metrics["gmck_metis"] = gmck_metis
    pol_metrics["mblb_metis"] = mblb_metis

    print(pol_metrics)
    
    os.makedirs("polarization_scores", exist_ok = True) 
    with open(f'./polarization_scores/{network_name}_{year}_pol.json', 'w') as fp:
        json.dump(pol_metrics, fp, indent=2)

if __name__ == "__main__":
    run_pipeline()
