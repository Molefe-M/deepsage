import os
import time
import torch
import torch.nn as nn
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
import tensorflow as tf
from tensorflow import keras
from common import load_save_data
from common.aggregators import MeanAggregator
from common.models import MLP, GCN, GraphSAGE, RankSAGE, GAT
from common.graph_dae_embedding import DaeFeaturePreparation, trainDAE, RetrieveDaeEmbeddings
from common.categorical_feature_representation import trainDNNE
from common.graph_representation_learning import GrlFeaturePreparation, trainGRL
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

base_path = "E:\\School_Work\\PhD_Experiments\\road_type_classification\\"

city = 'Johannesburg'
prefix_in = f'{city}-OSM-DNNE'
prefix_out = f'{city}-OSM-GRL'
format_ = 'pkl'
prefix = prefix_in
feature_type = 'dae'

input_path= f'{base_path}/output_graph_dae/graph_dae_features/'
if not os.path.exists(input_path):
    os.makedirs(input_path)

output_path= f'{base_path}/output_graph_grl/'
if not os.path.exists(output_path):
    os.makedirs(output_path)
    
path_performance= f'{output_path}/performance_report/{feature_type}/'
if not os.path.exists(path_performance):
    os.makedirs(path_performance)

path_models= f'{output_path}/models/'
if not os.path.exists(path_models):
    os.makedirs(path_models)

if __name__ == '__main__':
    '''
    - Load graph data
    '''
    prefix = prefix_in
    file_name = f'Road_network_graph-{prefix}.{format_}'
    L = load_save_data.load_graph(input_path, format_, file_name)
    
    '''
    - Prepare data to train GRL.
    '''
    batch_size = 1024
    feature_prep = GrlFeaturePreparation(L, feature_type, batch_size)
    torch_features, torch_class_labels, torch_edge_list, graph, graph_split, train_loader = feature_prep.main()
    
    print(torch_edge_list.shape)
    print(torch_edge_list.dtype)
    
    '''Train GRL.'''
    
    performance_results = []
    
    # MLP Parameters
    mlp_input_dim = torch_features.shape[1]
    input_graph_ = graph_split.x.shape[1]
    mlp_output_dim = len(torch_class_labels.unique())

    # GRL parameters
    n_channels = input_graph_
    hidden_channel1 = 32
    hidden_channel2 = 64
    hidden_out = 32
    out_channels = mlp_output_dim
    
    # GRL layer parameters
    aggregators = ['mean','max','sum']
    top_neighbours = [2, 4, 5, 10, 15, 20]
    learning_rates = [0.0001, 0.001, 0.01]
    
    #GRL training parameters
    loss = torch.nn.CrossEntropyLoss()
    epochs = 160
    measure = 'micro'
    
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
