import json
import pickle as pkl
import networkx as nx

def save_graph(graph, file_path, format_, file_name):
    '''
    Save a NetworkX graph to a file in the specified format.

    Args:
        graph (nx.Graph): The graph to save.
        file_path (str): The file path where the graph will be saved.
        format (str): The format to save the graph ('gml' or 'json').
    '''
    data = nx.json_graph.node_link_data(graph)
    if format_ == 'gml':
        data = nx.write_gml(graph, file_path+file_name)
        with open(file_path+file_name, 'w') as fp:
            json.dump(data, fp)
        print(f'Graph saved in GML format at: {file_path+file_name}') 
    elif format_ == 'json':
        data = nx.write_gml(graph, file_path+file_name)
        with open(file_path+file_name, 'w') as fp:
            json.dump(data, fp)
        print(f'Graph saved in JSON format at: {file_path+file_name}')
    elif format_ == 'pkl':
        pkl.dump(graph, open(file_path+file_name, 'wb'))
        print(f'Graph saved in PKL format at: {file_path+file_name}')
    else:
        raise ValueError("Unsupported format. Use 'gml', 'json' or 'pkl'.")


def load_graph(file_path, format_, file_name):
    '''
    Load a NetworkX graph from a file in the specified format.

    Args:
        file_path (str): The file path from which to load the graph.
        format (str): The format to load the graph ('gml' or 'json').

    Returns:
        nx.Graph: The loaded graph.
    '''
    if format_ == 'gml':
        graph = nx.read_gml(file_path+file_name)
        print(f'Graph loaded from GML format at: {file_path+file_name}')
        return graph
    elif format_ == 'json':
        with open(file_path+file_name, 'r') as fp:
            data = json.load(fp)
        graph = nx.json_graph.node_link_graph(data)
        print(f'Graph loaded from JSON format at: {file_path+file_name}')
        return graph
    elif format_ == 'pkl':
        graph = pkl.load(open(file_path+file_name, 'rb'))
        print(f'Graph loaded from PKL format at: {file_path+file_name}')
        return graph
    else:
        raise ValueError("Unsupported format. Use 'gml', 'json' or 'pkl'.")
        
