import os
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from common import load_save_data
from common.models import DNNE
from common.categorical_feature_representation import FeaturePreparation, trainDNNE, RetrieveEmbeddings

base_path = "E:\\School_Work\\PhD_Experiments\\road_type_classification\\"

city = 'Johannesburg'
prefix_in = f'{city}-OSM-One-Hot'
prefix_out = f'{city}-OSM-DNNE'
format_ = 'pkl'

input_path= f'{base_path}/input_graph/'
if not os.path.exists(input_path):
    os.makedirs(input_path)

output_path= f'{base_path}/output_graph_dnne/'
if not os.path.exists(output_path):
    os.makedirs(output_path)
    
path_performance= f'{output_path}/performance_report/'
if not os.path.exists(path_performance):
    os.makedirs(path_performance)

path_models= f'{output_path}/models/'
if not os.path.exists(path_models):
    os.makedirs(path_models)
    
path_labels= f'{output_path}/labels/'
if not os.path.exists(path_labels):
    os.makedirs(path_labels)
    
graph_path= f'{output_path}/graph_dnne_features/'
if not os.path.exists(graph_path):
    os.makedirs(graph_path)

if __name__ == '__main__':
    '''
    - Load graph data
    '''
    prefix = prefix_in
    file_name = f'Road_network_graph-{prefix}.{format_}'
    L = load_save_data.load_graph(input_path, format_, file_name)
    
    '''
    - Prepare data to train DNNE.
    '''
    p = 0.8
    feature_prep = FeaturePreparation(L, p,path_labels)
    num_speed, num_oneway, num_lanecount, num_label, df_categorical, df_combined, x_train, x_val, y_train, y_val = feature_prep.main()
    input_ = df_combined.drop('label', axis=1).values
    
    
    '''Train DNNE.'''
    
    performance_results = []

    # Classifier parameters
    learning_rates = [0.001, 0.01, 0.1]

    loss = tf.keras.losses.CategoricalCrossentropy()
    batch_size = 64
    epochs = 400
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience = 20)
    path = output_path

    # DNNE paramneters
    input_length = 1
    num_numerical_features = input_.shape[1]-3
    
    embedding_sizes = [10, 15, 20, 25, 30, 40]
    units = [128, 256, 512]
    
    for lr in learning_rates:
        optimiser = keras.optimizers.Adam(learning_rate=lr)
        for embedding_size in embedding_sizes:
            for unit in units:
                model_name = f'model_{lr}_{embedding_size}_{unit}'
                model = DNNE(num_speed, num_oneway, num_lanecount,\
                             num_label, embedding_size, input_,\
                             input_length, unit, num_numerical_features)

                model_train = trainDNNE(model, x_train, y_train, x_val,\
                                        y_val, loss, optimiser, lr, batch_size,\
                                        epochs, callback, path_models, model_name)

                model_,history= model_train.main()

                performance_results = model_train.performance_report(model_,x_val, y_val, performance_results,\
                                                                     model_name, lr, embedding_size, unit,history)


    results_df = pd.DataFrame(performance_results)
    # Save the results to a CSV file
    results_df.to_csv(path_performance+'performance_report.csv', index=False)
    
    '''
    - Load best model and retrieve it's embeddings
    '''
    prefix = prefix_out
    file_name = f'Road_network_graph-{prefix}.{format_}'
    report = pd.read_csv(path_performance+'performance_report.csv', index_col=False)
    min_ = report['Micro-Averaged F1 Score'].min()
    dd = report[report['Micro-Averaged F1 Score'] == min_]
    model_name = dd.model_name.values[0]
    model = model_train.load_model_artifacts(model_name+'.pkl', path_models)
    embeddings = RetrieveEmbeddings(model, df_categorical, L, path_labels)  
    L = embeddings.main()
    load_save_data.save_graph(L, graph_path, format_, file_name)