import networkx as nx
import numpy as np
import pymetis
import copy
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("netname")
args = parser.parse_args()
netname = str(args.netname)
year = "2023"

CANDIDATES_INFORMATION = 1

class SocialNetwork():

    def __init__(self, name, filename):
        self.name = name
        self.directed_graph = read_graphml_file(filename)
        self.undirected_graph = self.directed_graph.to_undirected()

        largest_cc = max(nx.connected_components(self.undirected_graph), key=len)
        self.giant_component = self.undirected_graph.subgraph(largest_cc).copy()

        self.giant_component_int = nx.convert_node_labels_to_integers(self.giant_component, first_label = 0, ordering = 'default', label_attribute = 'user_id')

    def get_giant_component_fraction(self):
        return len(self.giant_component)/len(self.undirected_graph)

    def get_adjacency_dict(self):
        adj_list = {}
        for node in self.giant_component_int.nodes:
            neighbors = list(self.giant_component_int.neighbors(node))
            adj_list[node] = neighbors
        return adj_list

def read_graphml_file(filename):
    """Reads a graph from a GraphML file and returns a NetworkX graph object."""
    try:
        # Read the graph from the file
        graph = nx.read_graphml(filename)

        # Return the graph object
        return graph
    except Exception as e:
        print(f"Error: {e}")
        return None
    
def get_partition(net):
    adj_dict = net.get_adjacency_dict()
    adj_list = [np.asarray(neighs) for neighs in adj_dict.values()]
    n_cuts, membership = pymetis.part_graph(nparts = 2, adjacency = adj_list, options = pymetis.Options(ufactor=400, niter=100, contig=True))
    membership = dict(zip(adj_dict.keys(), membership))
    return n_cuts, membership

def finetune_partition(net, membership):

    potential_bridge_nodes = []
    loner_nodes = []

    for node in net.giant_component_int.nodes:
        neighbors = net.giant_component_int.neighbors(node)
        neighbors_cluster = set([net.giant_component_int.nodes[n]["cluster"] for n in neighbors])
        if membership[node] not in neighbors_cluster:
            loner_nodes.append(node)

    membership_finetuned = copy.deepcopy(membership)

    c0 = {k for k, v in membership.items() if v == 0}
    c1 = {k for k, v in membership.items() if v == 1}

    q_best = nx.community.modularity(net.giant_component_int, [c0, c1])
    print(f"Before finetuning modularity is {q_best}")

    for node in loner_nodes:

        if membership[node] == 0:
            membership_finetuned[node] = 1
            new_label = 1
        else:
            membership_finetuned[node] = 0
            new_label = 0
        
        c0_candidate = {k for k, v in membership_finetuned.items() if v == 0}
        c1_candidate = {k for k, v in membership_finetuned.items() if v == 1}

        new_q = nx.community.modularity(net.giant_component_int, [c0_candidate, c1_candidate])
        
        if new_q > q_best:
            print(f"Improvement {new_q - q_best} by swapping node {node}")
            membership[node] = new_label
            q_best = new_q

        else:
            print(f"No improvement by swapping node {node}")
            membership_finetuned[node] = 1-new_label
            
    print(f"After finetuning modularity is {q_best}")
    return membership

def get_candidate_mappings():
    candidates = pd.read_csv("candidates-2023.csv")
    candidates_full = pd.read_csv("candidates2023-complete.csv")

    id_2_candidate = dict(zip(candidates.id.astype(str), candidates.screen_name))
    candidate_2_id = dict(zip(candidates.screen_name, candidates.id.astype(str)))

    candidates_full['twitter_id'] = candidates_full['screen_name'].map(candidate_2_id)

    id_2_party = dict(zip(candidates_full.twitter_id, candidates_full.puolue))
    id_2_age = dict(zip(candidates_full.twitter_id, candidates_full.ikä))
    id_2_sex = dict(zip(candidates_full.twitter_id, candidates_full.sukupuoli))
    id_2_hometown = dict(zip(candidates_full.twitter_id, candidates_full.kotikunta))
    id_2_lang = dict(zip(candidates_full.twitter_id, candidates_full.kieli))

    return id_2_candidate, id_2_party, id_2_age, id_2_sex, id_2_hometown, id_2_lang

def run_pipeline():

    filename = f"./pure-networks/{year}/{netname}_{year}_net.graphml"
    net = SocialNetwork(name = f"{netname}{year}", filename = filename)
    n_cuts, membership = get_partition(net)

    # ATTRIBUTE 1: Original partition
    nx.set_node_attributes(net.giant_component_int, membership, name="cluster")

    #membership_original = copy.deepcopy(membership)
    membership = finetune_partition(net, membership)

    # ATTRIBUTE 2: Finetuned partition
    nx.set_node_attributes(net.giant_component_int, membership, name="finetuned_cluster")

    if CANDIDATES_INFORMATION:
        # ATTRIBUTE 3: Candidate information
        id_2_candidate, id_2_party, id_2_age, id_2_sex, id_2_hometown, id_2_lang = get_candidate_mappings()

        screen_name_attributes = dict()
        party_attributes = dict()
        sex_attributes = dict()
        language_attributes = dict()

        for node in net.giant_component_int.nodes():
            node_user_id = net.giant_component_int.nodes[node]["user_id"]
            try:
                if node_user_id in id_2_candidate.keys():
                    screen_name_attributes[node] = id_2_candidate[node_user_id].rstrip()
                    party_attributes[node] = id_2_party[node_user_id].rstrip()
                    sex_attributes[node] = id_2_sex[node_user_id]
                    language_attributes[node] = id_2_lang[node_user_id]
                else:
                    screen_name_attributes[node] = "NA"
                    party_attributes[node] = "NA"
                    sex_attributes[node] = "NA"
                    language_attributes[node] = "NA"
            except:
                screen_name_attributes[node] = "NA"
                party_attributes[node] = "NA"
                sex_attributes[node] = "NA"
                language_attributes[node] = "NA"
                print(f"Error with node {node_user_id}")

        nx.set_node_attributes(net.giant_component_int, screen_name_attributes, "screen_name")
        nx.set_node_attributes(net.giant_component_int, party_attributes, "party")
        nx.set_node_attributes(net.giant_component_int, sex_attributes, "sex")
        nx.set_node_attributes(net.giant_component_int, language_attributes, "language")

    nx.write_graphml_lxml(net.giant_component_int, f"./rich-networks/{year}/RICH_{netname}_{year}_NET.graphml")


if __name__ == "__main__":
    run_pipeline()