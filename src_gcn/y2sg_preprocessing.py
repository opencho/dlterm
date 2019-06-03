from __future__ import division
from __future__ import print_function
import time
import numpy as np
import scipy.sparse as sp
import networkx as nx
from matplotlib import pylab
import matplotlib.pyplot as plt
#from Graph_Sampling import ForestFire
from copy import deepcopy
import random
import time
from collections import defaultdict
from utils import *
from copy import deepcopy

def load_data():
    G = nx.read_edgelist('data/yeast.edgelist')
    adj = nx.adjacency_matrix(G)
    return G, adj

# https://github.com/Ashish7129/Graph_Sampling/blob/master/Graph_Sampling/ForestFire.py
class ForestFire():
    def __init__(self):
        self.G1 = nx.Graph()
        self.emptyG = nx.Graph()

    def forestfire(self,G,size):
        list_nodes=list(G.nodes())
        #print(len(G))
        dictt = set()
        visits = defaultdict(lambda: False)
        random_node = random.sample(set(list_nodes),1)[0]
        #print(random_node)
        q = set() #q = set contains the distinct values
        q.add(random_node)
        while(len(self.G1.nodes())<size):
            if(len(q)>0):
                initial_node = q.pop()
                if(initial_node not in dictt):
                    dictt.add(initial_node)
                    neighbours = list(G.neighbors(initial_node))
                    
                    if len(neighbours) <= 1:
                        continue
                    
                    np = random.randint(1,len(neighbours))
                    for x in neighbours[:np]:
                        if(len(self.G1.nodes())<size):
                            self.G1.add_edge(initial_node,x)
                            q.add(x)
                        else:
                            break
                else:
                    continue
            else:
                random_node = random.sample(set(list_nodes) and dictt,1)[0]
                if visits[random_node] == True:
                    #print("redundant node {}".format(random_node))
                    return self.emptyG
                else:
                    q.add(random_node)
                    visits[random_node] = True
                    #print("add {}".format(random_node))
                
        q.clear()
        return self.G1
    
def min_max_norm(a):
    a_min = np.min(a)
    a_max = np.max(a)
    return (a - a_min)/(a_max - a_min)

def sampling_forestfire(G, y2seqs, nodes_to_sample, number_of_samples):
    print("Start SubGraph Sampling with ForestFire algorithm")
    target_graph = deepcopy(G)
    
    fsamples = []
    gsamples = []
    asamples = []
    
    # Sequence encoding
    p_mer = 3
    p_CTF = improvedCTF(letters=["A","B","C","D","E","F","G"],length=p_mer)
    rpdict = get_reduced_protein_letter_dict()
    
    for i in range(number_of_samples):
        gsampler = ForestFire()
        sampled_graph = gsampler.forestfire(target_graph, nodes_to_sample)
        if len(sampled_graph.nodes) == 0:
            print("loading... {}/{} <--- PASS".format(i+1,number_of_samples))
            continue
        
        
        sample_adj = nx.adjacency_matrix(sampled_graph)
        asamples.append(sample_adj)
        
        sg_nodes = sampled_graph.nodes
        sg_edges = sampled_graph.edges
        sg_node_seqs = [y2seqs[n][1].replace("*","") for n in sg_nodes]
        
        # Make reduced protein sequence letters
        sg_node_rseqs = []
        for seq in sg_node_seqs:
            rseq = ""
            for s in seq:
                rseq += rpdict[s]
            sg_node_rseqs.append(rseq)
        
        # CTF
        sg_node_features = []
        for seq in sg_node_rseqs:
            p_feature_dict = p_CTF.get_feature_dict()
            for mer in range(1,p_mer+1):
                for idx in range(0, len(seq)-mer):
                    pattern = seq[idx:idx+mer]
                    p_feature_dict[pattern] += 1
             
            p_feature = np.array(list(p_feature_dict.values()))
            p_feature = min_max_norm(p_feature)   
            sg_node_features.append(p_feature)
            
        gsamples.append((sg_nodes, sg_edges))
        fsamples.append(np.array(sg_node_features))
        
        if i % 1 == 0:
            print("loading... {}/{} | sample adj shape : {} | sample graph nodes : {}".format(i+1,number_of_samples, sample_adj.shape, len(sampled_graph.nodes)))
            

    asamples = np.array(asamples)
    gsamples = np.array(gsamples)
    fsamples = np.array(fsamples)
    
    print("Finish")
    print("feature vector shape : {}".format(fsamples.shape))
    
    return gsamples, asamples, fsamples

def load_yeast_seqs():
    return np.load("npz/yeast.sequences.npz")['protein_seqs'][()]

def save_yeast_subnets_preprocessed(num_nodes, num_samples,npz_path="npz/y2sg.npz"):
    print("Loading Yeast PPI network...")
    G, adj = load_data()
    
    print("Before: Graph Nodes (N): {}".format(len(G.nodes)))
    print("Before: Graph Edges (E): {}".format(len(G.edges)))
    print("Remove not exist nodes")
    y2seqs = load_yeast_seqs()
    for k, v in y2seqs.items():
        if v[0] == 0: 
            G.remove_node(k)
    
    print("After : Graph Nodes (N): {}".format(len(G.nodes)))
    print("After : Graph Edges (E): {}".format(len(G.edges)))
    gsamples, asamples, fsamples = sampling_forestfire(G, y2seqs, nodes_to_sample=num_nodes, number_of_samples=num_samples)
    
    print("---------------------------------")
    print("Sampled Graphs                   : {}".format(gsamples.shape))
    print("Sampled Graph Adjacency Matrices : {}".format(asamples.shape))
    print("Sampled Feature vectors          : {}".format(fsamples.shape))
    
    np.savez(npz_path, G=gsamples, A=asamples, F=fsamples)
    print("Saved at {}".format(npz_path))
    
    
if __name__ == "__main__":
    num_nodes = 200
    num_samples = 100
    
    save_yeast_subnets_preprocessed(num_nodes=num_nodes, 
                                    num_samples=num_samples, 
                                    npz_path="npz/y2sg_n{}_s{}.npz".format(num_nodes, num_samples))
    
    
    
    
    