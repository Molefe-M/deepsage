import torch
import numpy as np
import torch.nn as nn
import tensorflow as tf
from tensorflow import keras
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import degree
from tensorflow.keras import layers
from keras.models import Sequential,Model
from tensorflow.keras.layers import Embedding, Dense, Dropout, Input, Flatten, Concatenate
from torch_geometric.nn import SAGEConv, GCNConv, GATConv

class MLP(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.layers = torch.nn.Sequential(
        nn.Linear(input_dim, 20),
        nn.ReLU(),
        nn.Linear(20, 10),
        nn.ReLU(),
        nn.Linear(10, output_dim)
        )

    def forward(self, data):
        x = data.x  
        output = self.layers(x)
        return output

    
class DNNE(keras.Model):
    def __init__(self, num_speed, num_oneway, num_lanecount, num_label, embedding_size, df, input_length,\
                 units, num_numerical_features, **kwargs):
        super(DNNE, self).__init__(**kwargs)
        self.num_speed = num_speed
        self.num_oneway = num_oneway
        self.num_lanecount = num_lanecount
        self.num_label = num_label
        self.embedding_size = embedding_size
        self.units = units
        self.input_length = input_length 
        self.inputs = df
        self.num_numerical_features = num_numerical_features
        
        self.maxspeed_embedding = Sequential([Embedding(
                                           num_speed,
                                           embedding_size,
                                           embeddings_initializer="he_normal",
                                           embeddings_regularizer=keras.regularizers.l2(1e-6),
                                           input_length = input_length),
                                         Flatten()]
                                       )
        self.maxspeed_bias = Embedding(num_speed, 1)
    
        self.oneway_embedding = Sequential([Embedding(
                                           num_oneway,
                                           embedding_size,
                                           embeddings_initializer="he_normal",
                                           embeddings_regularizer=keras.regularizers.l2(1e-6),
                                           input_length = input_length),
                                         Flatten()]
                                       )
        self.oneway_bias = Embedding(num_oneway, 1)
    
        self.lanecount_embedding = Sequential([Embedding(
                                           num_lanecount,
                                           embedding_size,
                                           embeddings_initializer="he_normal",
                                           embeddings_regularizer=keras.regularizers.l2(1e-6),
                                           input_length = input_length),
                                         Flatten()]
                                       )
        self.lanecount_bias = Embedding(num_lanecount, 1)
        
        self.input_layer = Sequential([Input(shape=(embedding_size * 3 + num_numerical_features,))])
        
        self.dense_comp = Sequential([Dense(units, activation = tf.nn.relu),
                                         Dropout(0.2),
                                         Dense((units/4), activation = tf.nn.relu),
                                         Dropout(0.2),
                                         Dense((units/8), activation = tf.nn.relu),
                                         Dropout(0.2),
                                         Dense((units/16), activation = tf.nn.relu),
                                         Dense(num_label, activation = tf.nn.softmax)]
                                       )

    def call(self, inputs):
        maxspeed_vector = self.maxspeed_embedding(inputs[:, 0])
        maxspeed_bias = self.maxspeed_bias(inputs[:, 0])
        oneway_vector = self.oneway_embedding(inputs[:, 1])
        oneway_bias = self.oneway_bias(inputs[:, 1])
        lanecount_vector = self.lanecount_embedding(inputs[:, 2])
        lanecount_bias = self.lanecount_bias(inputs[:, 2])
        combined_vector = tf.concat([maxspeed_vector, oneway_vector, lanecount_vector], 1) + maxspeed_bias + oneway_bias + lanecount_bias 
        numerical_features = inputs[:, 3:]
        full_input_vector = tf.concat([combined_vector, numerical_features], axis=1)
        input_layer_features = self.input_layer(full_input_vector)
        y = self.dense_comp(input_layer_features)
        return y


class DimensionReductionTwo(Model):
    def __init__(self, input_shape_):
        super(DimensionReductionTwo, self).__init__()
        self.input_shape_ = input_shape_
        self.encoder = Sequential([
            Dense(input_shape_, activation = "relu"),
            Dense(48, activation = "relu"),
            Dense(38, activation = "relu"),
            Dense(28, activation = "relu"),
            Dense(18, activation = 'relu'),
            Dense(8, activation = 'relu')])

        self.decoder = Sequential([
            Dense(18, activation = "relu"),
            Dense(28, activation = "relu"),
            Dense(38, activation = "relu"),
            Dense(48, activation = "relu"),
            Dense(input_shape_, activation="sigmoid")])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class DimensionReductionOne(Model):
    def __init__(self, input_shape_):
        super(DimensionReductionOne, self).__init__()
        '''
            Parameters of the enocoder component
        '''
        self.input_shape_ = input_shape_
        self.encoder = Sequential([
            Dense(input_shape_, activation = "relu"),
            Dense(49, activation="relu"),
            Dense(40, activation="relu"),
            Dense(31, activation="relu"),
            Dense(22, activation="relu"),
            Dense(13, activation="relu"),
            Dense(4, activation = 'relu')])
        '''
            Parameters of the decoder component
        '''
        self.decoder = Sequential([
            Dense(13, activation="relu"),
            Dense(22, activation="relu"),
            Dense(31, activation="relu"),
            Dense(40, activation="relu"),
            Dense(49, activation="relu"),
            Dense(input_shape_, activation="sigmoid")])

    def call(self, x):
        '''
            forward function that executes the encoder and decoder component
        '''
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    

class DimensionReductionThree(Model):
    def __init__(self, input_shape_):
        super(DimensionReductionThree, self).__init__()
        '''
            Parameters of the enocoder component
        '''
        self.input_shape_ = input_shape_
        self.encoder = Sequential([
            Dense(input_shape_, activation = "relu"),
            Dense(46, activation="relu"),
            Dense(34, activation="relu"),
            Dense(22, activation="relu"),
            Dense(10, activation = 'relu')])
        '''
            Parameters of the decoder component
        '''
        self.decoder = Sequential([
            Dense(22, activation = "relu"),
            Dense(34, activation = "relu"),
            Dense(46, activation = "relu"),
            Dense(input_shape_, activation="sigmoid")])

    def call(self, x):
        '''
            forward function that executes the encoder and decoder component
        '''
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channel1, hidden_channel2, hidden_out, out_channels, aggregator_class):
        super(GCN, self).__init__()
        '''
        Parameter setup:
        in_channels: The size of the input features.
        hidden_channel1: The size of the hidden features after the first layer.
        hidden_channel2: The size of the hidden features after the second layer.
        hidden_out: The size of the hidden features before the final output.
        out_channels: The size of the output features.
        aggregator_class: The aggregator class to be used for aggregation.
        
        '''
        self.in_channels = in_channels
        self.hidden_channel1 = hidden_channel1
        self.hidden_channel2 = hidden_channel2
        self.hidden_out = hidden_out
        self.out_channels = out_channels
        
        self.conv1 = GCNConv(in_channels, hidden_channel1, aggr = aggregator_class)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channel1)
        
        self.conv2 = GCNConv(hidden_channel1, hidden_channel2, aggr = aggregator_class)
        self.bn2 = torch.nn.BatchNorm1d(hidden_channel2)

        self.hidden_fc = torch.nn.Linear(hidden_channel2, hidden_out)
        self.bn3 = torch.nn.BatchNorm1d(hidden_out)
        self.fc = torch.nn.Linear(hidden_out, out_channels)
        
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # First layer
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        # x = self.aggregator1(x, batch)
        
        # Second layer
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        # x = self.aggregator2(x, batch)
        
        # Hidden layer
        x = F.relu(self.hidden_fc(x))
        x = self.bn3(x)

        # Final output layer
        x = self.fc(x)
        return F.log_softmax(x, dim=1)

    
class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channel1, hidden_channel2, hidden_out, out_channels, aggregator_class):
        super(GraphSAGE, self).__init__()
        '''
        Parameter setup:
        in_channels: The size of the input features.
        hidden_channel1: The size of the hidden features after the first layer.
        hidden_channel2: The size of the hidden features after the second layer.
        hidden_out: The size of the hidden features before the final output.
        out_channels: The size of the output features.
        aggregator_class: The aggregator class to be used for aggregation.
        
        '''
        self.in_channels = in_channels
        self.hidden_channel1 = hidden_channel1
        self.hidden_channel2 = hidden_channel2
        self.hidden_out = hidden_out
        self.out_channels = out_channels
        self.aggregator_class = aggregator_class
        
        self.sage1 = SAGEConv(in_channels, hidden_channel1, aggr = aggregator_class, cached = False)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channel1)
        
        self.sage2 = SAGEConv(hidden_channel1, hidden_channel2, aggr = aggregator_class, cached = False)
        self.bn2 = torch.nn.BatchNorm1d(hidden_channel2)
        
        self.hidden_fc = torch.nn.Linear(hidden_channel2, hidden_out)
        self.bn3 = torch.nn.BatchNorm1d(hidden_out)
        self.fc = torch.nn.Linear(hidden_out, out_channels)
        
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # First layer
        x = self.sage1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        # x = self.aggregator1(x, batch)
        
        # Second layer
        x = self.sage2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        # x = self.aggregator2(x, batch)
        
        # Hidden layer
        x = F.relu(self.hidden_fc(x))
        x = self.bn3(x)

        # Final output layer
        x = self.fc(x)
        return F.log_softmax(x, dim=1) 

       
class RankSAGE(torch.nn.Module):
    
    def __init__(self, in_channels, hidden_channel1, hidden_channel2, hidden_out, out_channels, \
                 aggregator_class, top_neighbours):
        super(RankSAGE, self).__init__()
        
        '''
        Parameter setup:
        in_channels: The size of the input features.
        hidden_channel1: The size of the hidden features after the first layer.
        hidden_channel2: The size of the hidden features after the second layer.
        hidden_out: The size of the hidden features before the final output.
        out_channels: The size of the output features.
        aggregator_class: The aggregator class to be used for aggregation.
        top_neighbours: number of neighbours for sampling
        
        '''
        self.in_channels = in_channels
        self.hidden_channel1 = hidden_channel1
        self.hidden_channel2 = hidden_channel2
        self.hidden_out = hidden_out
        self.out_channels = out_channels
        self.aggregator_class = aggregator_class
        self.top_neighbours = top_neighbours
        
        self.sage1 = SAGEConv(in_channels, hidden_channel1, aggregator_class, cached = False)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channel1)
        
        self.sage2 = SAGEConv(hidden_channel1, hidden_channel2, aggregator_class, cached = False)
        self.bn2 = torch.nn.BatchNorm1d(hidden_channel2)

        self.hidden_fc = torch.nn.Linear(hidden_channel2, hidden_out)
        self.bn3 = torch.nn.BatchNorm1d(hidden_out)
        self.fc = torch.nn.Linear(hidden_out, out_channels)
        
        self.top_neighbours = top_neighbours
        
    def neighbourhood_sampling(self, edge_index, node_features, top_neighbours):
        # Convert edge_index to list of neighbors for each node
        neighbors = {}
        for edge in edge_index.t().tolist():
            src, dst = edge
            if src not in neighbors:
                neighbors[src] = []
            neighbors[src].append(dst)
        
        # Rank neighbors based on degree importance measure
        sampled_edge_index = []
        num_nodes = edge_index.max().item() + 1
        degrees = degree(edge_index[1], num_nodes=num_nodes)
        
        for node, nbrs in neighbors.items():
            neighbor_degrees = degrees[nbrs]
            top_indices = np.argsort(neighbor_degrees.numpy())[-top_neighbours:]  # Select top 10
            sampled_neighbors = [nbrs[i] for i in top_indices]
            for nbr in sampled_neighbors:
                sampled_edge_index.append((node, nbr))
        
        return torch.tensor(sampled_edge_index, dtype=torch.long).t().contiguous()
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Importance based sampling strategy
        edge_index = self.neighbourhood_sampling(edge_index, x, self.top_neighbours)
        
        # First layer
        x = self.sage1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        # x = self.aggregator1(x, batch)
        
        # Second layer
        x = self.sage2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        # x = self.aggregator2(x, batch)
        
        # Hidden layer
        x = F.relu(self.hidden_fc(x))
        x = self.bn3(x)

        # Final output layer
        x = self.fc(x)
        return F.log_softmax(x, dim=1)

    
class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channel1, hidden_channel2, hidden_out, out_channels):
        super(GAT, self).__init__()
        '''
        Parameter setup:
        in_channels: The size of the input features.
        hidden_channel1: The size of the hidden features after the first layer.
        hidden_channel2: The size of the hidden features after the second layer.
        hidden_out: The size of the hidden features before the final output.
        out_channels: The size of the output features.
        
        '''
        self.in_channels = in_channels
        self.hidden_channel1 = hidden_channel1
        self.hidden_channel2 = hidden_channel2
        self.hidden_out = hidden_out
        self.out_channels = out_channels
        
        self.gat1 = GATConv(in_channels, hidden_channel1)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channel1)
        
        self.gat2 = GATConv(hidden_channel1, hidden_channel2)
        self.bn2 = torch.nn.BatchNorm1d(hidden_channel2)
        
        self.hidden_fc = torch.nn.Linear(hidden_channel2, hidden_out)
        self.bn3 = torch.nn.BatchNorm1d(hidden_out)
        self.fc = torch.nn.Linear(hidden_out, out_channels)
        
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # First layer
        x = self.gat1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        
        # Second layer
        x = self.gat2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
   
        # Hidden layer
        x = F.relu(self.hidden_fc(x))
        x = self.bn3(x)

        # Final output layer
        x = self.fc(x)
        return F.log_softmax(x, dim=1)
    
if __name__ == "__main__":
    print("Model classes defined successfully.")