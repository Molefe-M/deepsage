import networkx as nx
import osmnx as ox
from datetime import datetime
import json
import geopandas as gpd
import numpy as np
import os
import sys


# ################# Create line graph (L) transformed applying line-graph-transformation on original graph (G)
def copy_edge_attributes_to_nodes(g, l, verbose=0):
    """
    This function copies the edge (road segments) features of the original graph G to nodes of the trandformed graph L(G)
    """
    if verbose > 0:
        print('Copying old edge attributes new node attributes...')
    node_attr = {}
    for u, v, d in g.edges(data=True):
        node_attr[(u, v)] = d
    nx.set_node_attributes(l, node_attr)

def convert_attributes_to_lists(g):
    for u, d in g.nodes(data=True):
        # print(d)
        for key, val in d.items():
            # print('here')
            # print(key, type(d[key]))
            if isinstance(val, np.ndarray):
                d[key] = val.flatten().tolist()

    for u, v, d in g.edges(data=True):
        # print(d)
        for key, val in d.items():
            # print('here')
            # print(key, type(d[key]))
            if isinstance(val, np.ndarray):
                d[key] = val.flatten().tolist()
    pass
    
def convert_to_line_graph(g, verbose=1):
    
    """
    This function convert G to L(G)
    1. Edges in G are set as nodes in L(G).
    2. Edges are created where common nodes exist.
    3. Edge attributes of G are copy to L(G) as nodes attributes.
    """
    # print input graph summary
    if verbose > 0:
        print('\n---Original Graph---')
        print(nx.info(g))

    # make edges to nodes, create edges where common nodes existed
    if verbose > 0:
        print('\nConverting to line graph...')
    l = nx.line_graph(g)

    # copy graph attributes
    l.graph['name'] = g.graph['name'] + '_line'
    l.graph['osm_query_date'] = g.graph['osm_query_date']
    l.graph['name'] = g.graph['name']

    # copy edge attributes to new nodes
    copy_edge_attributes_to_nodes(g, l, verbose=verbose)

    # relabel new nodes, storing old id in attribute
    mapping = {}
    for n in l:
        mapping[n] = n
    nx.set_node_attributes(l, mapping, 'original_id')
    l = nx.relabel.convert_node_labels_to_integers(l, first_label=0, ordering='default')

    # print output graph summary
    if verbose > 0:
        print('\n---Converted Graph---')
        print(nx.info(l))
        print('Done.')

    return l