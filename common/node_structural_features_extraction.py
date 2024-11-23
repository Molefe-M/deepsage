import numpy as np
import networkx as nx
import osmnx as ox
import json
import geopandas as gpd
import os
import sys


def node_degree_centrality(l=None, verbose=1):
    '''
    Add node degree centrality
    '''
    degree = {}
    for n in l.nodes:
        degree[n] = l.degree[n]
    nx.set_node_attributes(l, degree, 'degree_centrality')
    if verbose>0:
        print('n\Degree centrality added')
    
    return l

def eigenvector_centrality(l=None, verbose=1):
    '''
    Add eigenvector centrality
    '''
    centrality = nx.eigenvector_centrality(l, max_iter = 1000)
    nx.set_node_attributes(l, centrality, 'eigenvector_centrality')
    if verbose>0:
        print('n\Eigenvector centrality added')
    
    return l

def betweeness_centrality(l=None, verbose=1):
    '''
    Add betweeness centrality
    '''
    
    betweenness =nx.betweenness_centrality(l, k =5000)
    nx.set_node_attributes(l, betweenness, 'betweenness_centrality')
    if verbose>0:
        print('n\Betweenness centrality added')
    
    return l


def closeness_centrality(l=None, verbose=1):
    '''
    Add closeness centrality
    '''
    closeness = nx.closeness_centrality(l)
    nx.set_node_attributes(l, closeness, 'closeness_centrality')
    if verbose>0:
        print('n\Closeness centrality added')
    
    return l

def execute(L):
    '''
    Main function that runs the above functions
    '''
   
    L = node_degree_centrality(L)
    L = eigenvector_centrality(L)
#     L = betweeness_centrality(L)
    L = closeness_centrality(L)
    
    return L