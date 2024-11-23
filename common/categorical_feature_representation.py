import joblib
import pandas as pd
import numpy as np
import networkx as nx
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.utils import np_utils
from sklearn.metrics import classification_report
from common import line_graph_transformation

class FeaturePreparation:
    def __init__ (self, l, p, path):
        self.l = l
        self.p = p
        self.path = path

    def numerical_features(self, l):
        '''
        This function extracts numerical features from l
        
        Args:
            l:input graph
        
        Returns:
            data_arr: numerical features numpy array
            df_num: numerical features Pandas DataFrame
        '''
        data_arr = []
        line_graph_transformation.convert_attributes_to_lists(l)
        for n, d in l.nodes(data=True):
            data_arr.append(np.hstack(
                ([
                    d['midpoint'],
                    np.array(d['geom']),
                    d['length'],
                    d['travel_time']
                ])))
        df_num = pd.DataFrame(np.array(data_arr))
        return np.array(data_arr), df_num


    def graph_cat_features(self, l):
        '''
        This function extracts categorical features from l
        
        Args:
            l: input graph
        
        Returns:
            labels: road tyep class labels
            maspeed_rounded: maximum speed categorical feature
            oneway_label: one way categorical feature
            lane_count: lane count categorical feature
        '''
        labels = nx.get_node_attributes(l, 'label')
        maxspeed_rounded = nx.get_node_attributes(l, 'max_speed_rounded')
        oneway_label = nx.get_node_attributes(l, 'oneway_label')
        lane_count = nx.get_node_attributes(l, 'lanes')
        return labels, maxspeed_rounded, oneway_label, lane_count

    def cat_features_dataframe(self, labels, maxspeed_rounded, oneway_label, lane_count):
        '''
        This function created categorical features dataframe
        
        Args:
            labels: road tyep class labels
            maspeed_rounded: maximum speed categorical feature
            oneway_label: one way categorical feature
            lane_count: lane count categorical feature
        
        Returns:
            df_cat: categorical features Pandas Dataframe
            
        '''
        df_label = pd.DataFrame(labels.items(), columns = ['road_segment','label'])
        df_maxspeed = pd.DataFrame(maxspeed_rounded.items(), columns = ['road_segment','maxspeed'])
        df_oneway = pd.DataFrame(oneway_label.items(), columns = ['road_segment','oneway'])
        df_lanes = pd.DataFrame(lane_count.items(), columns = ['road_segment','lane_count'])
        df_cat = df_maxspeed.merge(df_oneway, on ='road_segment')
        df_cat = df_cat.merge(df_lanes, on ='road_segment')
        df_cat = df_cat.merge(df_label, on ='road_segment')
        return df_cat

    def features_encode(self, df_cat):
        '''
        This function Encode categorical variables using label encoding 
        
        Args:
            df_cat: categorical features Pandas Dataframe
        
        Returns:
            maxspeed_label: labels for maxspeeds variables
            oneway_label: labels for oneway variables
            lane_count_label: labels for oneway variables
        '''
        maxspeed_ids = df_cat['maxspeed'].unique().tolist()
        maxspeed_label = {x: i for i, x in enumerate(maxspeed_ids)}
        maxspeed_encoded = {i: x for i, x in enumerate(maxspeed_ids)}

        oneway_ids = df_cat['oneway'].unique().tolist()
        oneway_label = {x: i for i, x in enumerate(oneway_ids)}
        oneway_encoded = {i: x for i, x in enumerate(oneway_ids)}

        lane_count_ids = df_cat['lane_count'].unique().tolist()
        lane_count_label = {x: i for i, x in enumerate(lane_count_ids)}
        lane_count_encoded = {i: x for i, x in enumerate(lane_count_ids)}
        return maxspeed_label, oneway_label, lane_count_label

    def save_labels(self, maxspeed_label, oneway_label, lane_count_label, path):
        '''
        This function save labels as pkl files to the given path
        
        Args:
            path: path where labels will be saved
        '''
        joblib.dump(maxspeed_label, self.path+'maxspeed_label.pkl')
        joblib.dump(oneway_label, self.path+'oneway_label.pkl')
        joblib.dump(lane_count_label, self.path+'lane_count_label.pkl')
        pass

    def map_features(self, df_cat, maxspeed_label, oneway_label, lane_count_label):
        '''
        This function map features to labels
        
        Args:
            df_cat: categorical features Pandas Dataframe
            labels: (maxspeed, oneway and lane_count_labels)
        
        Returns:
            df_cat: categorical features Pandas Dataframe

        '''
        df_cat['maxspeed_enc'] = df_cat['maxspeed'].map(maxspeed_label)
        df_cat['oneway_enc'] = df_cat['oneway'].map(oneway_label)
        df_cat['lanecount_enc'] = df_cat['lane_count'].map(lane_count_label)
        return df_cat

    def combined_features(self, df_num, df_cat):
        '''
        This function combines numerical and categorical feature DataFrames
        
        Args:
            df_num: Pandas DataFrame with numerical features
            df_cat: Pandas Dataframe with categorical features
        
        Returns:
            df_combined: Combined features Pandas DataFrame
        '''
        col_excl = ['road_segment', 'maxspeed', 'oneway', 'lane_count']
        data_frames = [df_cat.drop(col_excl, axis=1), df_num]
        df_combined = pd.concat(data_frames, axis=1)
        return df_combined

    def features_length(self, df_combined):
        '''
        This function obtains categorical features length
        
        Args:
            df_combined: Combined features Pandas DataFrame
        
        Returns:
            num_speed: length of speed feature
            num_oneway: length of oneway feature
            num_label: length of classes
        '''
        num_speed = len(df_combined.maxspeed_enc.unique())
        num_oneway = len(df_combined.oneway_enc.unique())
        num_lanecount = len(df_combined.lanecount_enc.unique())
        num_label = len(df_combined.label.unique())
        return num_speed, num_oneway, num_lanecount, num_label

    def input_to_numpy(self, df_combined, p):
        '''
        This function converts features into numpy array and split to train/test
        
        Args:
            df_combined: Combined features Pandas DataFrame
            p: percentage split
        
        Returns:
             x_train: training features
             x_val: validation features
             y_train: training labels
             y_val: validation labels
        '''
        label_col = ['label']
        df_combined = df_combined.sample(frac=1, random_state=42)
        df_combined = df_combined.dropna()
        x = df_combined.drop('label', axis=1).values
        y = np_utils.to_categorical(df_combined[label_col])
        train_indices = int(p*df_combined.shape[0])
        x_train, x_val = (x[: train_indices], x[train_indices:])
        y_train, y_val = (y[: train_indices], y[train_indices:])
        return x_train, x_val, y_train, y_val
    
    @staticmethod
    def standardise_features(df_combined,columns):
        '''
        This function standardises features using min-max scaler
        
        Args:
            df_combined: Combined features Pandas DataFrame
            columns: Columns to standardise
        
        Returns:
            df_combined: Combined features Pandas DataFrame 
        '''

        df_combined[columns] = df_combined[columns].fillna(0)
        for column in columns:
            min_ = df_combined[column].min()
            max_ = df_combined[column].max()
            df_combined[column] = (df_combined[column] - min_)/(max_ - min_)
        return df_combined

    def main(self):
        '''
        Main function that executes the above functions
        '''
        labels, maxspeed_rounded, oneway_label, lane_count = self.graph_cat_features(self.l)
        df_categorical = self.cat_features_dataframe(labels, maxspeed_rounded, oneway_label, lane_count)
        maxspeed_label, oneway_label, lane_count_label = self.features_encode(df_categorical) 
        self.save_labels(maxspeed_label, oneway_label, lane_count_label, self.path)
        df_categorical = self.map_features(df_categorical, maxspeed_label, oneway_label, lane_count_label)
        data_arr, df_numerical = self.numerical_features(self.l)
        df_numerical = self.standardise_features(df_numerical,df_numerical.columns)
        df_combined = self.combined_features(df_numerical, df_categorical)
        num_speed, num_oneway, num_lanecount, num_label = self.features_length(df_combined)
        x_train, x_val, y_train, y_val = self.input_to_numpy(df_combined, self.p)
        return num_speed, num_oneway, num_lanecount, num_label, df_categorical, df_combined, x_train, x_val, y_train, y_val


class trainDNNE:
    def __init__ (self, model, x_train, y_train, x_val, y_val, loss, optimiser,\
                  learning_rate, batch_size, epochs, callback, path, model_name):
        self.model = model
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.loss = loss
        self.optimiser = optimiser
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.callback = callback
        self.path = path
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

    def train_model(self, model, x_train, y_train, x_val, y_val, batch_size, epochs, callback):
        '''
        Function to train the model
        
        Args:
            model: dnne model
            x_train: training features array
            y_train: training labels
            x_val: validation features array
            y_val: validation labels
            batch_size: number of batch size
            epochs: iteration
            callback: callback for model monitoring
            
        Returns:
            history: model history
        '''
        history = model.fit(
            x=x_train,
            y=y_train,
            batch_size= batch_size,
            epochs=epochs,
            verbose=1,
            callbacks = [callback],
            validation_data=(x_val, y_val)
        )
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
        plt.plot(history.history["accuracy"])
        plt.plot(history.history["val_accuracy"])
        plt.title("model accuracy")
        plt.ylabel("accuracy")
        plt.xlabel("epoch")
        plt.legend(["train", "validate"], loc="upper left")
        plt.show()
        pass

    def performance_report(self, model, x_val, y_val, performance_results, model_name, learning_rate, \
                          embedding_size, unit,history):
        '''
        This function create model perfomance metrics results
        '''
        average_train_loss = sum(history.history['loss']) / len(history.history['loss'])
        average_val_loss = sum(history.history['val_loss']) / len(history.history['val_loss'])
        # Convert one-hot encoded y_test to class indices if needed
        if y_val.ndim > 1:  # Check if y_test is one-hot encoded
            y_test_classes = tf.argmax(y_val, axis=1).numpy()  # Convert to class indices
        else:
            y_test_classes = y_val  # Already in the correct format
        # Make predictions
        y_pred = model.predict(x_val)
        y_pred_classes = tf.argmax(y_pred, axis=1)
        print(y_pred_classes)
        # Generate classification report
        report = classification_report(y_test_classes, y_pred_classes, output_dict=True)
        # Extract micro-averaged F1 score
        macro_f1_score = report['macro avg']['f1-score']
        micro_f1_score = report['accuracy']
        # Create a report
        performance_results.append({
            "model_name": model_name,
            "learning_rate": learning_rate,
            "unit": unit,
            "Average Training Loss": average_train_loss,
            "Average Validation Loss": average_val_loss,
            "Macro-Averaged F1 Score": macro_f1_score,
            "Micro-Averaged F1 Score": micro_f1_score 
        })
        return performance_results
    
    @staticmethod
    def save_model_artifacts(history, model, model_name, path):
        '''
        This function saves model artifacts
        '''
        model.save(path+model_name+'.pkl')
        joblib.dump(history.history,path+model_name+'history.pkl')
        pass
    
    @staticmethod
    def load_model_artifacts(model_name, path):
        '''
        This function loads mdoel artifacts
        '''
        model = tf.keras.models.load_model(path+model_name)
        return model
    
    def main(self):
        '''
        Main function that runs the above functions
        '''
        self.model_compile(self.model, self.loss, self.optimiser, self.learning_rate)
        history = self.train_model(self.model, self.x_train, self.y_train, self.x_val,\
                                   self.y_val, self.batch_size, self.epochs, self.callback)
        # self.plot_loss(history)
        # self.plot_accuracy(history)
        self.save_model_artifacts(history, self.model, self.model_name, self.path)
        
        return self.model, history

    
class RetrieveEmbeddings:
    def __init__ (self, model, data_frame, l, path):
        self.model = model
        self.data_frame = data_frame
        self.l = l
        self.path = path
        
    def load_labels(self, path):
        maxspeed_label = joblib.load(path+'maxspeed_label.pkl')
        oneway_label = joblib.load(path+'oneway_label.pkl')
        lane_count_label = joblib.load(path+'lane_count_label.pkl')
        return maxspeed_label, oneway_label, lane_count_label

    def retrieve_embeddings(self, model):
        weights_maxspeed = np.array(tf.Variable(model.layers[0].get_weights()[0][0:]))
        weights_oneway = np.array(tf.Variable(model.layers[2].get_weights()[0][0:]))
        weights_lanecount = np.array(tf.Variable(model.layers[4].get_weights()[0][0:]))
        return weights_maxspeed, weights_oneway, weights_lanecount

    def update_labels(self, data_frame, weights_maxspeed, weights_oneway, weights_lanecount,\
                      maxspeed_label,oneway_label,lane_count_label):
        maxspeed_label.update({60:weights_maxspeed[0], 40:weights_maxspeed[1], 80:weights_maxspeed[2], 30:weights_maxspeed[3],
                          50:weights_maxspeed[4], 100:weights_maxspeed[5], 70:weights_maxspeed[6], 120:weights_maxspeed[7],
                          20:weights_maxspeed[8], 90:weights_maxspeed[9]})

        oneway_label.update({0:weights_oneway[0],1:weights_oneway[1]})

        lane_count_label.update({'2':weights_lanecount[0],'3':weights_lanecount[1], '0':weights_lanecount[2], '1':weights_lanecount[3],
                          '4':weights_lanecount[4], '5':weights_lanecount[5], '6':weights_lanecount[6]})

        data_frame['maxspeed_emb'] = data_frame['maxspeed'].map(maxspeed_label)
        data_frame['oneway_emb'] = data_frame['oneway'].map(oneway_label)
        data_frame['lanecount_emb'] = data_frame['lane_count'].map(lane_count_label)

        emd_maxspeed = data_frame[['road_segment','maxspeed_emb']].set_index('road_segment').T.to_dict('list')
        emd_oneway = data_frame[['road_segment','oneway_emb']].set_index('road_segment').T.to_dict('list')
        emd_lanecount = data_frame[['road_segment','lanecount_emb']].set_index('road_segment').T.to_dict('list')
        return emd_maxspeed, emd_oneway, emd_lanecount

    def update_nodes_attributes(self, l, emd_maxspeed, emd_oneway, emd_lanecount):
        nx.set_node_attributes(l, emd_maxspeed, 'emd_maxspeed')
        nx.set_node_attributes(l, emd_oneway, 'emd_oneway')
        nx.set_node_attributes(l, emd_lanecount, 'emd_lanecount')
        return l

    def maxspeed_to_numpy(self, l):
        emd_maxspeed = nx.get_node_attributes(l, 'emd_maxspeed')
        emd_oneway = nx.get_node_attributes(l, 'emd_oneway')
        emd_lanecount = nx.get_node_attributes(l, 'emd_lanecount')
        res = dict()
        for key in emd_maxspeed: 
            res[key] = np.array(emd_maxspeed[key])
        nx.set_node_attributes(l, res, 'emd_maxspeed')
        return l

    def oneway_to_numpy(self, l):
        emd_oneway = nx.get_node_attributes(l, 'emd_oneway')
        res = dict()
        for key in emd_oneway: 
            res[key] = np.array(emd_oneway[key])
        nx.set_node_attributes(l, res, 'emd_oneway')
        return l

    def lanecount_to_numpy(self, l):
        emd_lanecount = nx.get_node_attributes(l, 'emd_lanecount')
        res = dict()
        for key in emd_lanecount: 
            res[key] = np.array(emd_lanecount[key])
        nx.set_node_attributes(l, res, 'emd_lanecount')
        return l
    
    def main(self):
        maxspeed_label, oneway_label, lane_count_label = self.load_labels(self.path)
        weights_maxspeed, weights_oneway, weights_lanecount = self.retrieve_embeddings(self.model)
        emd_maxspeed, emd_oneway, emd_lanecount = self.update_labels(self.data_frame, weights_maxspeed,weights_oneway,\
                                                        weights_lanecount,maxspeed_label,oneway_label,lane_count_label)
        L = self.update_nodes_attributes(self.l, emd_maxspeed, emd_oneway, emd_lanecount)
        L = self.maxspeed_to_numpy(L)
        L = self.oneway_to_numpy(L)
        L = self.lanecount_to_numpy(L)
        return L
