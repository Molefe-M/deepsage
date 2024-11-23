import math
import joblib
import pandas as pd
import numpy as np
import networkx as nx
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from keras.utils import np_utils
from sklearn.metrics import classification_report
from common import line_graph_transformation, load_save_data

from common.categorical_feature_representation import FeaturePreparation, trainDNNE

class DaeFeaturePreparation:
    def __init__ (self, l, p, feature_type):
        self.l = l
        self.p = p
        self.feature_type = feature_type 
    
    @staticmethod
    def one_hot_features(l):
        '''
        This function raw features and one hot encoded features
        
        Args:
            l:input graph
        
        Returns:
            one_hot_arr: numerical and one-hot features numpy array
        '''
        one_hot_arr = []
        line_graph_transformation.convert_attributes_to_lists(l)
        for n, d in l.nodes(data=True):
            one_hot_arr.append(np.hstack(
                ([
                    d['midpoint'],
                    np.array(d['maxspeed_one_hot']),
                    np.array(d['lane_count_one_hot']),
                    np.array(d['oneway_one_hot']),
                    np.array(d['geom']),
                    d['length'],
                    d['travel_time']
                ])))
        df_one_hot = pd.DataFrame(np.array(one_hot_arr)).fillna(0)
        # df_one_hot = FeaturePreparation.standardise_features(df_one_hot, df_one_hot.columns) 
        scaler = MinMaxScaler(feature_range=(0, 1))
        features = scaler.fit_transform(df_one_hot.values)
        return features
    
    @staticmethod
    def dnne_emb_features(l):
        '''
        This function raw features and dnne features

        Args:
            l:input graph

        Returns:
            dnne_emb_arr: numerical and dnne features numpy array
        '''
        dnne_emb_arr = []
        line_graph_transformation.convert_attributes_to_lists(l)
        for n, d in l.nodes(data=True):
            dnne_emb_arr.append(np.hstack(
                ([
                    d['midpoint'],
                    np.array(d['emd_maxspeed']),
                    np.array(d['emd_lanecount']),
                    np.array(d['emd_oneway']),
                    np.array(d['geom']),
                    d['length'],
                    d['travel_time']
                ])))
        df_dnne = pd.DataFrame(np.array(dnne_emb_arr)).fillna(0)
        scaler = MinMaxScaler(feature_range=(0, 1))
        features = scaler.fit_transform(df_dnne.values)
        # df_dnne = FeaturePreparation.standardise_features(df_dnne, df_dnne.columns) 
        return features
    
    @staticmethod
    def dae_features(l):
        '''
        This function extracts dae features

        Args:
            l:input graph

        Returns:
            dae_emb_arr: dae features
        '''
        dae_emb_arr = []
        line_graph_transformation.convert_attributes_to_lists(l)
        for n, d in l.nodes(data=True):
            dae_emb_arr.append(np.hstack(
                ([
                    np.array(d['dae_features'])
                ])))
        df_dae = pd.DataFrame(np.array(dae_emb_arr)).fillna(0)
        scaler = MinMaxScaler(feature_range=(0, 1))
        features = scaler.fit_transform(df_dae.values)
        # df_dae = FeaturePreparation.standardise_features(df_dae, df_dae.columns) 
        return features
    
    
    def trian_test_split(self, features, p):
        '''
        This function splits data into train and test set

        Args:
            features: scaled features
            p: percentage split

        Returns:
            train_features: training features
            test_features: testing features
        '''
        train_length = math.ceil(len(features)*p)
        train = features[0:train_length,:]
        test = features[train_length:,:]
        return train, test
    
    def main(self):
        '''
        main function that executes the above functions
        '''
        if self.feature_type == 'one-hot':
            features = self.one_hot_features(self.l)
        else:
            features = self.dnne_emb_features(self.l)
        train_features, test_features = self.trian_test_split(features, self.p)
        return train_features, test_features
    

class trainDAE:
    def __init__ (self, model, train_features, test_features, loss, optimiser,\
                  learning_rate, batch_size, epochs, callback, path, path_, model_name):
        self.model = model
        self.train_features = train_features
        self.test_features = test_features
        self.loss = loss
        self.optimiser = optimiser
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.callback = callback
        self.path = path
        self.path_ = path_
        self.model_name = model_name
        
    def model_compile(self, model, loss, optimizer, learning_rate):
        '''
        Function to compile the model
        
        Args:
            model: dnne model
            loss: loss function 
            optimiser: model optimiser
            learning_rate: model learning rate
        '''
        model.compile(
            loss = loss,
            optimizer = optimizer,
            metrics = ['accuracy'])
        pass

    def train_model(self, model, train_features, test_features, batch_size, epochs, callback):
        '''
        Function to train the model
        
        Args:
            model: dnne model
            train_features: training features 
            test_features: test features
            batch_size: number of batch size
            epochs: iteration
            callback: callback for model monitoring
            
        Returns:
            history: model history
        '''
        history = model.fit(train_features, train_features, 
                            epochs=epochs, 
                            batch_size=batch_size, 
                            callbacks = [callback],
                            validation_data=(test_features, test_features))

        return history

    def plot_loss(self, history):
        '''
        plot losses
        '''
        plt.plot(history.history["loss"])
        plt.plot(history.history["val_loss"])
        plt.title("model loss")
        plt.ylabel("loss")
        plt.xlabel("epoch")
        plt.legend(["train", "validate"], loc="upper left")
        plt.show()
        pass
    
    def plot_accuracy(self, history):
        '''
        Plot accuracy
        '''
        plt.figure(figsize=(10,5))
        plt.plot(history.history['loss'], label = 'train_loss')
        plt.plot(history.history['val_loss'], label = 'validation_loss')
        plt.title('Mean Squared Error loss')
        plt.xlabel('Epochs')
        plt.ylabel('loss')
        plt.grid(True)
        plt.legend()
        plt.show()
        pass
    
    def plot_reconstructions(self, test_features, model, model_name, path_):
        '''
        This function plost reconstructions based on the test set
        '''
        test_reconstr = model.encoder(test_features).numpy()
        test_reconstr_dec = model.decoder(test_reconstr).numpy()
        for i in range(1,50):
            plt.plot(test_features[i], 'b')
            plt.plot(test_reconstr_dec[i],'r')
            plt.legend(labels = ['input','reconstrcutions, error'])
            plt.savefig(path_+f'reconstruction_{model_name}_{i}.png')
            plt.close()
        pass

    def mean_reconstruction_error(self, model, test_features):
        '''
          This function computes the mean reconstructions error 
        '''
        test_recons = model.predict(test_features)
        test_loss = tf.keras.losses.mse(test_recons,test_features)
        test_loss_mean = np.mean(test_loss)
        
        return test_loss_mean

    def performance_report(self, test_loss_mean, performance_results, model_name, learning_rate):
        '''
        '''
        performance_results.append({
            "model_name": model_name,
            "learning_rate": learning_rate,
            "Reconstruction error": test_loss_mean
        })
        return performance_results
    
    def main(self):
        '''
        Main function that runs the above functions
        '''
        self.model_compile(self.model, self.loss, self.optimiser, self.learning_rate)
        history = self.train_model(self.model, self.train_features, self.test_features,\
                                   self.batch_size, self.epochs, self.callback)
        # self.plot_loss(history)
        # self.plot_accuracy(history)
        self.plot_reconstructions(self.test_features, self.model, self.model_name, self.path_)
        test_loss_mean = self.mean_reconstruction_error(self.model, self.test_features)
        trainDNNE.save_model_artifacts(history, self.model, self.model_name, self.path)
        return self.model, history, test_loss_mean 
    
class RetrieveDaeEmbeddings:
    def __init__ (self, l, model, feature_type):
        self.l = l
        self.model= model
        self.feature_type = feature_type
        
    def extract_latent_features(self, model, features):
        '''
        This function extracts dae features

        Args:
            model: trained DAE model
            features: scaled features

        Returns:
            dae_features: features produced by the dae model
        '''
        
        dae_features = model.encoder(features)
        
        return dae_features
    
    def append_dae_features(self, l, dae_features):
        '''
        This function appends dae features to graph L

        Args:
            l: Transformed graph
            dae_features: features produced by dae model
        
        Returns:
            l: Graph with dae features added
        '''
        features = {}
        for n in l.nodes:
            features[n] = dae_features[n]
        nx.set_node_attributes(l, features, 'dae_features')
        
        return l
    
    def main(self):
        '''
        main function that executes the above functions
        '''
        # extract features
        if self.feature_type == 'one-hot':
            features = DaeFeaturePreparation.one_hot_features(self.l)
        else:
            features = DaeFeaturePreparation.dnne_emb_features(self.l)
        
        dae_features = self.extract_latent_features(self.model, features)
        L = self.append_dae_features(self.l, dae_features)
        return L