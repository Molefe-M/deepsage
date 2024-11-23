import os
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from common import load_save_data
from common.models import DimensionReductionOne, DimensionReductionTwo, DimensionReductionThree
from common.graph_dae_embedding import DaeFeaturePreparation, trainDAE, RetrieveDaeEmbeddings
from common.categorical_feature_representation import trainDNNE

base_path = "E:\\School_Work\\PhD_Experiments\\road_type_classification\\"

city = 'Johannesburg'
prefix_in = f'{city}-OSM-One-Hot'
prefix_out = f'{city}-OSM-DNNE'
format_ = 'pkl'
prefix = prefix_in
feature_type = 'dnne'

input_path= f'{base_path}/output_graph_dnne/graph_dnne_features/'
if not os.path.exists(input_path):
    os.makedirs(input_path)

output_path= f'{base_path}/output_graph_dae/'
if not os.path.exists(output_path):
    os.makedirs(output_path)
    
path_performance= f'{output_path}/performance_report/'
if not os.path.exists(path_performance):
    os.makedirs(path_performance)

path_models= f'{output_path}/models/'
if not os.path.exists(path_models):
    os.makedirs(path_models)

path_plots= f'{output_path}/dae_plots/'
if not os.path.exists(path_plots):
    os.makedirs(path_plots)
    
graph_path= f'{output_path}/graph_dae_features/'
if not os.path.exists(graph_path):
    os.makedirs(graph_path)

if __name__ == '__main__':
    '''
    - Load graph data
    '''
    prefix = prefix_out
    file_name = f'Road_network_graph-{prefix}.{format_}'
    L = load_save_data.load_graph(input_path, format_, file_name)
    
    '''
    - Prepare data to train DAE.
    '''
    p = 0.7
    feature_prep = DaeFeaturePreparation(L, p, feature_type)
    train_features, test_features = feature_prep.main()
    
    print(train_features.shape)
    
    '''Train DAE.'''
    
    performance_results = []

    # dae parameters
    learning_rates = [0.0001, 0.001, 0.01, 0.1]
    loss = 'mean_squared_error'
    batch_size = 64
    epochs = 400
    callback = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 10)
    input_shape_ = train_features.shape[1]
    
    model_dae_one = DimensionReductionOne(input_shape_) 
    model_dae_two = DimensionReductionTwo(input_shape_)
    model_dae_three = DimensionReductionThree(input_shape_)
    models = [model_dae_one, model_dae_two, model_dae_three]
    model_names = ['dae_architecture_one','dae_architecture_two','dae_architecture_three']
    
    path = path_models
    path_ = path_plots
    
    for model, model_name in zip(models, model_names):
        for lr in learning_rates:
            optimiser = keras.optimizers.Adam(learning_rate=lr)
            model_name_ = f'{model_name}_{lr}'
            model_ = model
            model_train = trainDAE(model_, train_features, test_features, loss,\
                                   optimiser, lr, batch_size, epochs, callback,\
                                   path, path_, model_name_)
            
            model_,history, test_loss_mean = model_train.main()

            performance_results = model_train.performance_report(test_loss_mean,performance_results, model_name_,lr)
    
    results_df = pd.DataFrame(performance_results)
    # Save the results to a CSV file
    results_df.to_csv(path_performance+'performance_report.csv', index=False)
                                    
    '''
    - Load best model and retrieve it's embeddings
    '''
    prefix = prefix_out
    file_name = f'Road_network_graph-{prefix}.{format_}'
    report = pd.read_csv(path_performance+'performance_report.csv', index_col=False)
    min_ = report['Reconstruction error'].min()
    dd = report[report['Reconstruction error'] == min_]
    model_name = dd.model_name.values[0]
    model = trainDNNE.load_model_artifacts(model_name+'.pkl', path_models)
    embeddings = RetrieveDaeEmbeddings(L, model, feature_type)  
    L = embeddings.main()
    load_save_data.save_graph(L, graph_path, format_, file_name)