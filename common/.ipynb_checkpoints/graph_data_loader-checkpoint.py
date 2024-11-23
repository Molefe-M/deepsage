import torch
import numpy as np
import networkx as nx
from sklearn.preprocessing import MinMaxScaler

class GraphLoader:
    def __init__(self, path=None, features=None, labels=None, edge_list=None, batch_size=None):
        self.path = path
        self.features = features
        self.labels = labels
        self.edge_list = edge_list
        self.batch_size = batch_size
 
    def read_all_sources(self, path, features, labels, edge_list):
        node_features = np.load(path+features)
        scaler = MinMaxScaler(feature_range=(0, 1))
        np_node_features = scaler.fit_transform(node_features)

        dict_labels = open(path+labels)
        dict_labels = json.load(dict_labels)
        list = []
        for key in dict_labels:
            list.append(dict_labels[key])
        np_labels = np.array(list)

        edge_list = open(path+edge_list)
        edge_list = json.load(edge_list)
        np_edge_list = np.array(edge_list)
        return np_node_features, np_labels, np_edge_list

    def numpy_to_torch (self, np_node_features, np_labels, np_edge_list):
        node_features = torch.from_numpy(np_node_features).to(torch.float32)
        labels = torch.from_numpy(np_labels).to(torch.long)
        edge_list = torch.from_numpy(np_edge_list).to(torch.long).T
        return node_features, labels, edge_list
    
    def construct_graph (self, node_features, labels, edge_list):
        graph = Data(x = node_features, edge_index = edge_list, y = labels)
        return graph

    def graph_splits (self, graph):
        split = T.RandomNodeSplit(num_val=0.15, num_test=0.15)
        graph_split = split(graph)
        return graph_split

    def data_loader(self, graph_split, batch_size):
        train_loader = NeighborLoader(graph_split, num_neighbors=[5, 10], batch_size=batch_size, input_nodes=graph_split.train_mask)
        return train_loader

    def convert_to_networkx(self, graph, n_sample=None):
        g = to_networkx(graph)
        y = graph.y.numpy()
        if n_sample is not None:
            sampled_nodes = random.sample(g.nodes, n_sample)
            g = g.subgraph(sampled_nodes)
            y = y[sampled_nodes]
        return g, y

    def plot_graph(self, g, y):
        plt.figure(figsize=(70, 35))
        nx.draw(g, with_labels = True, font_weight = 'bold', node_size=70, arrows=False, node_color=y)

    def main():
        np_node_features, np_labels, np_edge_list = self.read_all_sources(path, features, labels, edge_list)
        node_features, labels, edge_list = self.numpy_to_torch(np_node_features, np_labels, np_edge_list)
        graph = self.construct_graph (node_features, labels, edge_list)
        graph_split = self.graph_splits(graph)
        #g, y = convert_to_networkx(graph, n_sample=6764)
        #plot_graph(g, y)
        train_loader = data_loader(graph_split, batch_size)
        return np_node_features, np_labels, np_edge_list, node_features, labels, edge_list, graph, graph_split, train_loader