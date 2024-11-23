import networkx as nx
import osmnx as ox
from datetime import datetime
import json
import geopandas as gpd
import os
import sys

def extract_osm_network_transductive(name=None, poi=None, buffer=None, verbose=1):
    '''
    extract the graph from OSM and convert to undirected graph
    '''
    timestamp = datetime.now().astimezone().strftime("%Y-%m-%dT%H:%M:%S %z")
   
    # Retrieve osm data by center coordinate and spatial buffer
    g = ox.graph_from_point(poi, dist = buffer, network_type='drive', simplify=True)
    g = ox.project_graph(g, to_crs="EPSG:32633")

    g.graph['osm_query_date'] = timestamp
    g.graph['name'] = name
    g.graph['poi'] = poi
    g.graph['buffer'] = buffer
    
    # Add speed attribute
    g = ox.speed.add_edge_speeds(g) 
    
    # Add travel time attribute
    g=ox.speed.add_edge_travel_times(g)
    
    # create incremental node ids
    g = nx.relabel.convert_node_labels_to_integers(g, first_label=0, ordering='default')

    # convert to undirected graph (i.e. directions and parallel edges are removed)
    g = nx.Graph(g.to_undirected())
    
    if verbose > 0:
        print('\nDone extracting the graph from OSM...')

    return g