import numpy as np
import networkx as nx
import osmnx as ox
import matplotlib.pyplot as plt
from datetime import datetime
import random
import collections
from sklearn.preprocessing import OneHotEncoder
from shapely.geometry import LineString
import json
import geopandas as gpd
import os
import sys

def get_params_transductive():
    
    '''
    This function defines two parameters:
    1. Label_lookup: Edge labels parameters of original road names, these are replaced by integer values. 
    2. Label lookup: grouped edge labels to reduce imbalanced dataset.
    '''
    
    PARAMS = {
        
        'geom_vector_len': 20,
        
        'exclude_edge_attributes': ['osmid', 'bridge', 'tunnel', 'ref', 'junction',
                                    'access', 'name', 'key'],
        'exclude_node_attributes': ['ref', 'osmid'],

        # Original labels
        'label_lookup_': {'motorway': 0,
                          'trunk': 1,
                          'primary': 2,
                          'secondary': 3,
                          'tertiary': 4,
                          'unclassified': 5,
                          'residential': 6,
                          'motorway_link': 7,
                          'trunk_link': 8,
                          'primary_link': 9,
                          'secondary_link': 10,
                          'tertiary_link': 11,
                          'living_street': 12,
                          'road': 13,
                          'yes': 14,
                          'planned': 15
                          },
        # Merged labels
        'label_lookup': {'motorway': 0,
                         'trunk': 0,  # merge for class balance
                         'primary': 0,  # merge for class balance
                         'secondary': 0,  # merge for class balance
                         'tertiary': 1,
                         'unclassified': 2,
                         'residential': 3,
                         'motorway_link': 0,  # merge for class balance
                         'trunk_link': 0,  # merge for class balance
                         'primary_link': 0,  # merge for class balance
                         'secondary_link': 0,  # merge for class balance
                         'tertiary_link': 1,  # merge for class balance
                         'living_street': 4,
                         'road': 5,
                         'yes': 0,
                         'planned': 5
                         },
        
        'oneway_lookup':{'False': 0,
                         'True': 1
                        }
        
    }

    return PARAMS

def plot_original_graph():
    '''
    Function to plot the graph
    '''
    plt.figure(figsize=(80, 40))
    nx.draw(G,with_labels = True, font_weight ='bold')
    
    pass

def convert_class_labels(g, PARAMS):
    """
    This function does the following:
    1. Assign a default label 'road' to roads (edges) that doesn not have a highway label
    2. Takes the first element on the road (edges) segments that have more than one highway label
    3. Adds road (edges) attributes not included to a look up table. 
    """
    cnt = 0
    labels = nx.get_edge_attributes(g, 'highway')
    labels_int = {}
    for edge in g.edges:
        # If an edge is unlabelled,assign default label 'road'
        if not edge in labels:
            labels[edge] = 'road'

        # some edges have two attributes, take only their first
        if type(labels[edge]) == list:
            labels[edge] = labels[edge][0]

        # some edges have attributes, not listed in our label lookup
        # these could be added to the label lookup if increases significantly
        if not labels[edge] in PARAMS['label_lookup']:
            cnt += 1
            labels[edge] = 'road'

        #print('Number of newly added road labels by OSM:', cnt)
        labels_int[edge] = PARAMS['label_lookup'][labels[edge]]

    nx.set_edge_attributes(g, labels_int, 'label')
    pass

def remove_unwanted_attributes(g, PARAMS):
    """
    This function removes some nodes and edge attributes defined in function 'get_params_transductive' 
    """
    
    # deleting some node attributes
    for n in g:
        for att in PARAMS['exclude_node_attributes']:
            g.nodes[n].pop(att, None)
    # deleting some edge attributes
    for n1, n2, d in g.edges(data=True):
        for att in PARAMS['exclude_edge_attributes']:
            d.pop(att, None)
    pass

def standardize_geometries(g, PARAMS, attr_name='geom', verbose=0):

    steps = PARAMS['geom_vector_len']

    if verbose > 0:
        print('\nGenerating fixed length (%d) geometry vectors...' % (steps))
    geoms = nx.get_edge_attributes(g, 'geometry')
    xs = nx.get_node_attributes(g, 'x')
    ys = nx.get_node_attributes(g, 'y')
    np_same_length_geoms = {}
    count_no = 0
    count_yes = 0 
    for e in g.edges():
        points = []

        if e not in geoms:  # edges that don't have a geometry
            line = LineString([(xs[e[0]], ys[e[0]]), (xs[e[1]], ys[e[1]])])
            for step in np.linspace(0, 1, steps):
                point = line.interpolate(step, normalized=True)
                points.append([point.x, point.y])
            count_no += 1

        else:  # all other edges
            for step in np.linspace(0, 1, steps):
                point = geoms[e].interpolate(step, normalized=True)
                points.append([point.x, point.y])
            count_yes += 1
        np_same_length_geoms[e] = np.array([np.array((p[0], p[1])) for p in points])

    if verbose > 0:
        print('- Geometry inserted from intersection coordinates for', count_no, 'nodes.')
        print('- Standardized geometry created for', count_no + count_yes, 'nodes.')

    nx.set_edge_attributes(g, np_same_length_geoms, attr_name)
    if verbose > 0:
        print('Done.')
    pass

def standardize_speed(g):
    '''
    This function standardizes max speed limit on each road segment to the nearest 10.
    It then adds a new attribute called 'max_speed_rounded'
    '''
    max_speed = nx.get_edge_attributes(g, 'speed_kph')
    # initializing K
    K = 0
    # loop to iterate for values
    res = dict()
    for key in max_speed:
        # rounding to K using round()
        res[key] = round(max_speed[key], K)
        res[key] = round(max_speed[key], -1)
        res[key] = int(res[key])
    nx.set_edge_attributes(g, res, 'max_speed_rounded')
    pass

# def travel_time(g):
#     '''
#     This function calculates the travel time based on the newly added max_speed_eounded attributes.
#     '''
#     # loop to iterate for values
#     res = dict()
#     for edge in g.edges:
#         max_speed = nx.get_edge_attributes(g, 'max_speed')
#         distance = nx.get_edge_attributes(g, 'length')
#         res[edge] = ((distance[edge])/(max_speed[edge])*3.6) 
#     nx.set_edge_attributes(g, res, 'travel_time')
#     pass

def convert_oneway_labels(g, PARAMS):
    oneway = nx.get_edge_attributes(g, 'oneway')
    oneway_int = {}
    for k, v in oneway.items():
        oneway[k] = str(v)
        
    for edge in g.edges:
        oneway_int[edge] = PARAMS['oneway_lookup'][oneway[edge]]
        
    nx.set_edge_attributes(g, oneway_int, 'oneway_label')
    pass

def convert_lane_count(g):
    """
    This function does the following:
    1. Assign a default value 'lane' to lanes not having count
    2. Takes the first element of the lane count that have more than one lane count
    """
    lanes = nx.get_edge_attributes(g, 'lanes')
    lanes_int = {}
    for edge in g.edges:
        # If an edge is unlabelled,assign default label 'road'
        if not edge in lanes:
            lanes[edge] = '0'
        # some edges have two attributes, take only their first
        if type(lanes[edge]) == list:
            lanes[edge] = lanes[edge][0]
        lanes_int = lanes[edge]
        
    nx.set_edge_attributes(g, lanes, 'lanes')
    pass



def midpoint(p1, p2):
    return ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2) # function to calculate midpoint between two nodes (p1 and p2) (x:0, y:1) 


def midpoint_generation(g):
    pos = {}
    for u, d in g.nodes(data=True):
        pos[u] = (d['x'], d['y'])
    new_pos = {}
    for u, v, d in g.edges(data=True): # Calculate midpoint between two nodes
        e = (u, v)
        new_pos[e] = {'midpoint': np.array(midpoint(pos[u], pos[v]))}
    nx.set_edge_attributes(g, new_pos)
    pass

def midpoint_subtraction(g): # substract geom points by midpoint of two nodes
    for u, v, d in g.edges(data=True):
        e = (u, v)
        d['geom'] = d['geom'] - d['midpoint']
    pass

def one_hot_encode_maxspeeds(g, verbose=0):
    if verbose > 0:
        print('\nGenerating one-hot encoding maxspeed limits...')

    maxspeeds_standard = [5, 7, 10, 20, 30, 40, 50, 60,\
                          70, 80, 90, 100, 110, 120]

    maxspeeds = nx.get_edge_attributes(g, 'max_speed_rounded')
    

    maxspeeds_single_val = {}
    for e in g.edges():
        if e not in maxspeeds:
            maxspeeds[e] = 'unknown'
            
        if type(maxspeeds[e]) == list:
            maxspeeds_single_val[e] = maxspeeds[e][0]
        else:
            maxspeeds_single_val[e] = maxspeeds[e]

    for e in maxspeeds_single_val:
        if maxspeeds_single_val[e] not in maxspeeds_standard:
            if maxspeeds_single_val[e].isdigit():
                maxspeeds_standard.append(maxspeeds_single_val[e])
            else:
                maxspeeds_single_val[e] = 'unknown'

    enc = OneHotEncoder(handle_unknown='ignore')
    # enc.fit(np.array(list(maxspeeds_single_val.values())).reshape(-1, 1))

    enc.fit(np.array(maxspeeds_standard).reshape(-1, 1))

    if verbose > 0:
        print('- One-hot encoder fitted to data with following categories:')
        print('-', np.array(enc.categories_).flatten().tolist())

    maxspeeds_one_hot = {k: enc.transform(np.array(v).reshape(1, -1)).toarray().flatten().tolist() for k, v in
                         maxspeeds_single_val.items()}
    if verbose > 0:
        print('- One-hot encoded maxspeed limits generated.')

    nx.set_edge_attributes(g, maxspeeds_one_hot, 'maxspeed_one_hot')
    if verbose > 0:
        print('Done.')
    pass

def one_hot_lanes (g, verbose=0):
    if verbose > 0:
        print('\nGenerating one-hot encoding maxspeed limits...')

    standard_lanes = ['0', '1', '2', '3', '4','5','6']

    lane_count = nx.get_edge_attributes(g, 'lanes')
    lane_count_single_val = {}
    for e in g.edges():
        if e not in lane_count:
            lane_count[e] = '0'
    
        if type(lane_count[e]) == list:
            lane_count_single_val[e] = lane_count[e][0]
        else:
            lane_count_single_val[e] = lane_count[e]

    for e in lane_count_single_val:
        if lane_count_single_val[e] not in standard_lanes:
            if lane_count_single_val[e].isdigit():
                standard_lanes.append(lane_count_single_val[e])
            else:
                lane_count_single_val[e] = '0'

    enc = OneHotEncoder(handle_unknown='ignore')
    # enc.fit(np.array(list(maxspeeds_single_val.values())).reshape(-1, 1))

    enc.fit(np.array(standard_lanes).reshape(-1, 1))

    if verbose > 0:
        print('- One-hot encoder fitted to data with following categories:')
        print('-', np.array(enc.categories_).flatten().tolist())

    lane_count_one_hot = {k: enc.transform(np.array(v).reshape(1, -1)).toarray().flatten().tolist() for k, v in
                         lane_count_single_val.items()}
    if verbose > 0:
        print('- One-hot encoded lane count generated.')

    nx.set_edge_attributes(g, lane_count_one_hot, 'lane_count_one_hot')
    if verbose > 0:
        print('Done.')
    pass

def one_hot_oneway (g, verbose=0):
    if verbose > 0:
        print('\nGenerating one-hot encoding maxspeed limits...')

    standard_oneway = [0, 1, 'unknown']
    
    oneway = nx.get_edge_attributes(g, 'oneway_label')
    oneway_single_val = {}
    
    for k, v in oneway.items():
        oneway[k] = str(v)
    for e in g.edges():
        if e not in oneway:
            oneway[e] = 'unknown'
    
        if type(oneway[e]) == list:
            oneway_single_val[e] = oneway[e][0]
        else:
            oneway_single_val[e] = oneway[e]

    for e in oneway_single_val:
        if oneway_single_val[e] not in standard_oneway:
            if oneway_single_val[e].isdigit():
                standard_oneway.append(oneway_single_val[e])
            else:
                oneway_single_val[e] = 'unknown'

    enc = OneHotEncoder(handle_unknown='ignore')
    # enc.fit(np.array(list(maxspeeds_single_val.values())).reshape(-1, 1))

    enc.fit(np.array(standard_oneway).reshape(-1, 1))

    if verbose > 0:
        print('- One-hot encoder fitted to data with following categories:')
        print('-', np.array(enc.categories_).flatten().tolist())

    lane_count_one_hot = {k: enc.transform(np.array(v).reshape(1, -1)).toarray().flatten().tolist() for k, v in
                         oneway_single_val.items()}
    if verbose > 0:
        print('- One-hot encoded lane count generated.')

    nx.set_edge_attributes(g, lane_count_one_hot, 'oneway_one_hot')
    if verbose > 0:
        print('Done.')
    pass

def extract_raw_features(g, verbose=1):
    PARAMS = get_params_transductive()
    standardize_speed(g)
#     travel_time(g)
    convert_class_labels(g, PARAMS)
    convert_oneway_labels(g,PARAMS)
    convert_lane_count(g)
    remove_unwanted_attributes(g, PARAMS)
    standardize_geometries(g, PARAMS, verbose=verbose)
    midpoint_generation(g)
    midpoint_subtraction(g)
    one_hot_encode_maxspeeds(g, verbose=verbose)
    one_hot_lanes(g, verbose=verbose)
    one_hot_oneway(g, verbose=verbose)
    
    return g