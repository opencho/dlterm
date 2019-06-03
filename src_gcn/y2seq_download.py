import requests
import numpy as np
import time
import datetime
import networkx as nx

def download_yeast_protein_seq(yid = 'YAL034W-A'):
    def return_empty_with_msg(msg):
        print(msg)
        return (0, "")
    
    protein_length = 0
    protein_sequence = ""
    
    # 1. Request Sequence Data
    try:
        r = requests.get("http://yeastgenome.org/backend/locus/{}/sequence_details".format(yid))
    except:
        return_empty_with_msg("[ERROR] not request REST-API, id = {}".format(yid))
     
    # 2. Parsing the data
    try:
        ret = r.json()
    except:
        return_empty_with_msg("[ERROR] cannot parsing the requested json , id = {}".format(yid))
    
    # 3. Find Protein sequences
    try:
        protein_length = ret['protein'][0]['protein_length']
        protein_sequence = ret['protein'][0]['residues']
    except:
        return_empty_with_msg("[ERROR] not exist protein sequence, id = {}".format(yid))
    
    return (protein_length, protein_sequence)


def load_node_list(fname='data/yeast.edgelist'):
    g = nx.read_edgelist(fname)
    adj = nx.adjacency_matrix(g)
    nodelist = nx.nodes(g)
    nodes = list(set(nodelist.keys()))
    return nodes
 
    
    
if __name__ == "__main__":
    nodelist = load_node_list()
    start_time = time.time()
    protein_seqs = {}
    temp_time = start_time + 1e-10
    
    # Remove for all data download
    nodelist = nodelist[:10]
    
    for i, yid in enumerate(nodelist):
        cur_time = time.time() - start_time
        cur_perc = (i /len(nodelist)) * 100
        if cur_perc > 0 :
            wait_time = ((100-cur_perc) / cur_perc) * cur_time
        else:
            wait_time = 9999999.9999999
        print("sequence downloading... [{:.5f} (%)] / time - [{:.5f}(s)] / expected waiting time is {}(s)".format(cur_perc, cur_time, str(datetime.timedelta(seconds=wait_time))))
        pseq = download_yeast_protein_seq(yid)
        protein_seqs[yid] = pseq

    yseq_path = "npz/yeast.sequences.npz"
    np.savez(yseq_path, protein_seqs=protein_seqs)
    print("saved the yeast protein sequences at {}".format(yseq_path))