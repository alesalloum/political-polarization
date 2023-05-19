import os
import gzip
import json
import logging

from tqdm import tqdm
import pandas as pd

import networkx as nx


# Helper functions
def check_retweet_status(obj):
    """
    Checks whether an object is a retweet.

    Args:
        obj (dict): A dictionary representing the object to check.

    Returns:
        bool: True if the object is a retweet, False otherwise.
    """
    return "referenced_tweets" in obj and obj["referenced_tweets"][0]["type"] == "retweeted"

def remove_duplicates(raw_data):
    """
    Remove duplicate objects from a list of JSON objects.

    Args:
        raw_data (list): A list of JSON objects, each represented as a dictionary.

    Returns:
        list: A new list of JSON objects with duplicates removed.
    """

    logging.info(f"Number of json objects found is {len(raw_data)}.")
    
    no_duplicates = []

    for obj in tqdm(raw_data):
        if obj not in no_duplicates:
            no_duplicates.append(obj)

    logging.info(f"After removing the duplicates, we are left with {len(no_duplicates)} objects.")

    return no_duplicates

# Process functions

def load_data(tweets_file):
    
    logging.info(f"Loading file {tweets_file} now.")
    
    raw_data = []
    
    with open(os.path.join(input_data_dir, tweets_file), mode = 'r') as f_tweets:
        for line in f_tweets:
            tweet = json.loads(line)
            raw_data.append(tweet)

    return raw_data

def filter_data(data):

    data_preprocessed_1 = remove_duplicates(data)
    data_preprocessed_2 = [obj for obj in data_preprocessed_1 if check_retweet_status(obj)]

    return data_preprocessed_2

def save_network_data(edge_data, path, network_context):
    """Saves network data to files in the specified directory.

    Args:
        edge_data (list): List of tuples containing edge data.
        path (str): Path to the directory where files will be saved.
        network_context (str): Name of the network to be saved.

    Returns:
        None

    Saves two files in the specified directory:
        - A comma-separated text file containing edge data (source, target, timestamp),
          named <network_context>_edgelist.txt
        - A GraphML file containing the network graph, named <network_context>_net.graphml
    """
    
    df = pd.DataFrame(edge_data, columns=["source", "target", "timestamp"])
   
    full_path_edgelist = os.path.join(path, network_context + "_edgelist.txt")
    df.to_csv(full_path_edgelist, index=False, header=False)

    full_path_graphml = os.path.join(path, network_context + "_net.graphml")
    G = nx.from_pandas_edgelist(df, create_using = nx.DiGraph())
    nx.write_graphml_lxml(G, full_path_graphml)

# :------------------: #

# :------------------: #

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define input and output directories and network name
#input_data_dir = "../keywords_non_universal_stream"
#output_data_dir = "./keywords_non_universal_stream_processed"
input_data_dir = "../keywords_nonpolitical"
output_data_dir = "./pure-networks"
network_context = "WILMAMURTO_2023"
network_context_str = "WILMAMURTO_2023"
twitter_filename = "wilmamurto-historical.jsonl"

def run_pipeline():

    logging.info(f"Starting data pipeline for {network_context}...")

    # Load data from input directory

    EDGE_DATA = []

    data = load_data(twitter_filename)
    logging.info("Data loaded successfully")


    data = filter_data(data)
    logging.info("Data filtered successfully")
        
    for retweet in data:

        #tweet_text = retweet["referenced_tweets"][0]["tweet"]["text"]

        retweeter_node = retweet["author_id"]
        retweeted_node = retweet["referenced_tweets"][0]["tweet"]["author_id"]
        timestamp = retweet["created_at"]

        EDGE_DATA.append((retweeter_node, retweeted_node, timestamp))
    
    logging.info(f"Edge formation done successfully. {len(EDGE_DATA)} relevant edges were formed.")
    save_network_data(EDGE_DATA, output_data_dir, network_context_str)
    logging.info(f"Network {network_context_str} data saved successfully")

if __name__ == "__main__":
    run_pipeline()
