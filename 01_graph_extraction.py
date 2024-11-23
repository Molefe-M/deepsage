import os
import json
import logging
import pickle as pkl
from common import osm_graph_extraction, raw_feature_extraction,\
line_graph_transformation, node_structural_features_extraction,\
load_save_data

base_path = "E:\\School_Work\\PhD_Experiments\\road_type_classification\\"

input_path= f'{base_path}/input_graph/'
if not os.path.exists(input_path):
    os.makedirs(input_path)

if __name__ == '__main__':
    '''
    - Define parameters for network extraction from OSM.

    '''
    poi = (-26.2050000, 28.0497220)
    city = 'Johannesburg'
    name = f'{city}-OSM'
    buffer = 7000
    prefix = f'{city}-OSM-One-Hot'
    format_ = 'pkl'
    file_name = f'Road_network_graph-{prefix}.{format_}'
    
    '''Extract original input graph.'''
    G = osm_graph_extraction.extract_osm_network_transductive(name, poi, buffer, verbose = 1)
    
    '''Extract Raw features.'''
    G = raw_feature_extraction.extract_raw_features(G)
    
    '''Convert to Line graph.'''
    L = line_graph_transformation.convert_to_line_graph(G, verbose=1)
    
    '''Extract Structural features'''
#     L = node_structural_features_extraction.execute(L)
    
    '''Save graph and features'''
    load_save_data.save_graph(L, input_path, format_, file_name)        
    