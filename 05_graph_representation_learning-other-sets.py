import os
import time
import torch
import torch.nn as nn
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from common.models import DimensionReductionThree
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid, Reddit
from torch_geometric.loader import NeighborLoader
import tensorflow as tf
from tensorflow import keras
from common import load_save_data
from common.aggregators import MeanAggregator
from common.models import MLP, GCN, GraphSAGE, RankSAGE, GAT
from common.graph_dae_embedding import DaeFeaturePreparation, trainDAE, RetrieveDaeEmbeddings
from common.categorical_feature_representation import trainDNNE
from common.graph_representation_learning import GrlFeaturePreparation, trainGRL
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

base_path = "E:\\School_Work\\PhD_Experiments\\other_datasets\\"

dataset = 'reddit'

output_path= f'{base_path}/{dataset}/'
if not os.path.exists(output_path):
    os.makedirs(output_path)
    
path_performance= f'{output_path}/performance_report/'
if not os.path.exists(path_performance):
    os.makedirs(path_performance)

if __name__ == '__main__':
    '''
    - Load datasets
    '''
    if dataset == 'cora':
        dataset = Planetoid(root=output_path, name='Cora')
        data = dataset[0]
    
    elif dataset == 'reddit':
        dataset = Reddit(root=output_path)
        data = dataset[0]
    
    elif dataset == 'CiteSeer':
        dataset = Planetoid(root=output_path, name='CiteSeer')
        data = dataset[0]
    
    else:
        print('Invalid dataset provided. Please provide "Cora", "Reddit", "CiteSteer"')
        
    '''
    - Prepare data to train GRL.
    '''
    batch_size = 1024
    split = T.RandomNodeSplit(num_val=0.15, num_test=0.15)
    graph_split = split(data) 
    train_loader = NeighborLoader(data, num_neighbors=[5, 10], batch_size=batch_size, input_nodes=graph_split.train_mask)
    
    
    '''Train GRL.'''
    
    performance_results = []
    
    # MLP Parameters
    mlp_input_dim = data.num_node_features
    input_graph_ = mlp_input_dim
    mlp_output_dim = dataset.num_classes

    # GRL parameters
    n_channels = input_graph_
    hidden_channel1 = 32
    hidden_channel2 = 64
    hidden_out = 32
    out_channels = mlp_output_dim
    
    # GRL layer and training parameters
    aggregators = ['mean']
    top_neighbours = [5] #ranksage
    learning_rates = [0.01]
    loss = torch.nn.CrossEntropyLoss()
    epochs = 160
    measure = 'micro'

    # DAE layer and training parameters
    learning_rate_dae = [0.0001]
    loss = 'mean_squared_error'
    batch_size = 64
    epochs = 400
    callback = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 10)
    input_shape_ = mlp_input_dim
    model_dae_three = DimensionReductionThree(input_shape_)
    models = [model_dae_three]
    model_names = ['dae_architecture_three']
    
    # Choose GRL Model to train
    grl_model = 'gat'
    
    if grl_model == 'ranksage':
        model_name = grl_model
        for aggregator in aggregators:
            for neighbour in top_neighbours:
                model = RankSAGE(n_channels, hidden_channel1, hidden_channel2, hidden_out, out_channels, aggregator, neighbour).to(device)
                for lr_ in learning_rates:
                    optimiser = torch.optim.Adam(model.parameters(), lr=lr_, weight_decay=5e-4)
                    start_time = time.time() 
                    print(f'training {model_name} at agg funct: {aggregator} and top neighb: {neighbour} and lr: {lr_}')
                    model_train = trainGRL(graph_split, model, train_loader, optimiser, loss, epochs, measure, out_channels)
                    model_, val_micro_f1, test_micro_f1 = model_train.main()
                    end_time = time.time()
                    training_time = (end_time - start_time) / 60
                    print(f'Done training {model_name} at agg funct: {aggregator} and top neighb: {neighbour} and lr: {lr_} after {training_time} minutes')
                    performance_results = trainGRL.performance_report(performance_results, model_name, aggregator, lr_,\
                                                                      neighbour, val_micro_f1, test_micro_f1, training_time)
            
        results_df = pd.DataFrame(performance_results)
        # Save the results to a CSV file
        results_df.to_csv(path_performance+f'performance_report_{model_name}.csv', index=False)
    
    elif grl_model == 'graphsage':
        model_name = grl_model
        for aggregator in aggregators:
            model = GraphSAGE(n_channels, hidden_channel1, hidden_channel2, hidden_out, out_channels, aggregator).to(device)
            for lr_ in learning_rates:
                optimiser = torch.optim.Adam(model.parameters(), lr=lr_, weight_decay=5e-4)
                start_time = time.time() 
                print(f'training {model_name} at agg funct: {aggregator} and lr: {lr_}')
                model_train = trainGRL(graph_split, model, train_loader, optimiser, loss, epochs, measure, out_channels)
                model_, val_micro_f1, test_micro_f1 = model_train.main()
                end_time = time.time()
                training_time = (end_time - start_time) / 60
                print(f'Done training {model_name} at agg funct: {aggregator} and lr: {lr_} after {training_time} minutes')
                neighbour = 'nan'
                performance_results = trainGRL.performance_report(performance_results, model_name, aggregator, lr_,\
                                                                  neighbour, val_micro_f1, test_micro_f1, training_time)
        
        results_df = pd.DataFrame(performance_results)
        # Save the results to a CSV file
        results_df.to_csv(path_performance+f'performance_report_{model_name}.csv', index=False)
    
    elif grl_model == 'gcn':
        model_name = grl_model
        aggregator = 'sum'
        model = GCN(n_channels, hidden_channel1, hidden_channel2, hidden_out, out_channels, aggregator).to(device)
        for lr_ in learning_rates:
            optimiser = torch.optim.Adam(model.parameters(), lr=lr_, weight_decay=5e-4)
            start_time = time.time() 
            print(f'training {model_name} at agg funct: {aggregator} and lr: {lr_}')
            model_train = trainGRL(graph_split, model, train_loader, optimiser, loss, epochs, measure, out_channels)
            model_, val_micro_f1, test_micro_f1 = model_train.main()
            end_time = time.time()
            training_time = (end_time - start_time) / 60
            print(f'Done training {model_name} at agg funct: {aggregator} and lr: {lr_} after {training_time} minutes')
            neighbour = 'nan'
            performance_results = trainGRL.performance_report(performance_results, model_name, aggregator, lr_,\
                                                              neighbour, val_micro_f1, test_micro_f1, training_time)
        
        results_df = pd.DataFrame(performance_results)
        # Save the results to a CSV file
        results_df.to_csv(path_performance+f'performance_report_{model_name}.csv', index=False)
    
    elif grl_model == 'gat':
        model_name = grl_model
        model = GAT(n_channels, hidden_channel1, hidden_channel2, hidden_out, out_channels).to(device)
        for lr_ in learning_rates:
            optimiser = torch.optim.Adam(model.parameters(), lr=lr_, weight_decay=5e-4)
            start_time = time.time() 
            model_train = trainGRL(graph_split, model, train_loader, optimiser, loss, epochs, measure, out_channels)
            model_, val_micro_f1, test_micro_f1 = model_train.main()
            end_time = time.time()
            training_time = (end_time - start_time) / 60
            neighbour, aggregator = 'nan', 'nan'
            performance_results = trainGRL.performance_report(performance_results, model_name, aggregator, lr_,\
                                                              neighbour, val_micro_f1, test_micro_f1, training_time)
            
        
        results_df = pd.DataFrame(performance_results)
        # Save the results to a CSV file
        results_df.to_csv(path_performance+f'performance_report_{model_name}.csv', index=False)
    
    elif grl_model == 'mlp':
        model_name = grl_model
        model = MLP(n_channels, out_channels).to(device)
        for lr_ in learning_rates:
            optimiser = torch.optim.Adam(model.parameters(), lr=lr_, weight_decay=5e-4)
            start_time = time.time() 
            model_train = trainGRL(graph_split, model, train_loader, optimiser, loss, epochs, measure, out_channels)
            model_, val_micro_f1, test_micro_f1 = model_train.main()
            end_time = time.time()
            training_time = (end_time - start_time) / 60
            neighbour, aggregator = 'nan', 'nan'
            performance_results = trainGRL.performance_report(performance_results, model_name, aggregator, lr_,\
                                                              neighbour, val_micro_f1, test_micro_f1, training_time)
            
        results_df = pd.DataFrame(performance_results)
        # Save the results to a CSV file
        results_df.to_csv(path_performance+f'performance_report_{model_name}.csv', index=False)
        
    else:
        print('Invalid grl model provided. Please provide "ranksage", "graphsage", "gcn", "gat" or "mlp"')
