import torch
import numpy as np
import torch.nn as nn
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.data import Data, DataLoader
from torch_geometric.loader import NeighborLoader
from torch_geometric.utils.convert import to_networkx
from torch_geometric.utils.convert import from_networkx
from torchmetrics.classification import MulticlassF1Score

from common.graph_dae_embedding import DaeFeaturePreparation


class GrlFeaturePreparation:
    def __init__(self, l, feature_type, batch_size):
        self.l = l
        self.feature_type = feature_type
        self.batch_size = batch_size
        
    def load_features(self, l, feature_type):
        '''
        This function loads input features
        
        Args:
            l: input graph
            feature: type of features to load
        
        Returns:
            features
        '''
        if feature_type == 'one-hot':
            features = DaeFeaturePreparation.one_hot_features(l)
        elif feature_type == 'dnne':
            features = DaeFeaturePreparation.dnne_emb_features(l)
        elif feature_type == 'dae':
            features = DaeFeaturePreparation.dae_features(l)
        else:
            print(f"Invalid feature type. Please provide 'one-hot', 'dnne'  or 'dae'")
            
        return features
    
    def load_class_labels(self, l):
        '''
        This function extracts road type labels from l
        
        Args:
            l: input graph
        
        Returns:
            class_labels: road type labels
        '''
        class_map = {}
        for n in l.nodes:
            class_map[str(n)] = np.array(l.nodes[n]['label']).astype(int).tolist()
        
        list_labels = []
        for key in class_map:
            list_labels.append(class_map[key])
        class_labels = np.array(list_labels)
        
        return class_labels
    
    def load_edge_list(self, l):
        '''
        This function extracts edge list from l
        
        Args:
            l: input graph
        
        Returns:
            edge_list: edge_list
        
        '''
        b=list(l.edges)
        edge_list = np.array(b)
        return edge_list
    
    def numpy_to_torch (self, features, class_labels, edge_list):
        '''
        This function converts graph attributes torch
        
        Args:
            features: numpy array of features
            class_labels: numpy array of class labels
            edge_list: numpy array of edge list
        
        Returns:
            torch_features: pytorch array features
            torch_class_labels: pytorch class labels array
            torch_edge_list: pytorch edge_list
        '''
        torch_features = torch.from_numpy(features).to(torch.float32)
        torch_class_labels = torch.from_numpy(class_labels).to(torch.long)
        torch_edge_list = torch.from_numpy(edge_list).to(torch.long).T
        # torch_edge_list = torch.tensor(edge_list).T
        return torch_features, torch_class_labels, torch_edge_list
    
    def construct_graph (self, torch_features, torch_class_labels, torch_edge_list):
        '''
        This function constructs graph for use with pytorch pygeometry
        
        Args:
            features: numpy array of features
            class_labels: numpy array of class labels
            edge_list: numpy array of edge list
        
        Returns:
            graph: pytorch graph
        '''
        graph = Data(x = torch_features, edge_index = torch_edge_list, y = torch_class_labels)
        
        return graph
    
    def graph_splits(self, graph):
        '''
        Args:
            graph: pytorch graph
        
        Returns: 
            graph_split
            
        '''
        split = T.RandomNodeSplit(num_val=0.15, num_test=0.15)
        graph_split = split(graph)
        return graph_split
    
    def data_loader(self, graph_split, batch_size):
        '''
        Function to perform data loader
        
        Args:
            graph_split:
            batch_size:
        
        Returns:
            train_loader
        '''
        train_loader = NeighborLoader(graph_split, num_neighbors=[5, 10], batch_size=batch_size, input_nodes=graph_split.train_mask)
        return train_loader
    
    def main(self):
        features = self.load_features(self.l, self.feature_type)
        class_labels = self.load_class_labels(self.l)
        edge_list = self.load_edge_list(self.l)
        torch_features, torch_class_labels, torch_edge_list = self.numpy_to_torch(features, class_labels, edge_list)
        graph = self.construct_graph(torch_features, torch_class_labels, torch_edge_list)
        graph_split = self.graph_splits(graph)
        train_loader = self.data_loader(graph_split, self.batch_size)
        return torch_features, torch_class_labels, torch_edge_list, graph, graph_split, train_loader 

    
class trainGRL:
    def __init__(self, graph, model, train_loader, optimiser, loss_fn, n_epochs, measure, num_classes):
        self.graph = graph
        self.model = model
        self.train_loader = train_loader
        self.optimiser = optimiser
        self.loss_fn = loss_fn
        self.n_epochs = n_epochs
        self.measure = measure
        self.num_classes = num_classes
        
    def train_node_classifier(self, model, train_loader, optimiser, loss_fn, n_epochs):
        train_losses = []
        train_accuracies = []
        val_losses = []
        val_accuracies = []
        
        model.train()
        for epoch in range(n_epochs+1):
            train_loss = 0
            train_acc = 0
            val_loss = 0
            val_acc = 0
            # Train on batches
            for batch in train_loader:
                 #===================Foward pass =============
                out = model(batch)
                loss = loss_fn(out[batch.train_mask], batch.y[batch.train_mask])
                train_loss += loss 
                train_acc += self.eval_node_classifier(model, batch, batch.train_mask)
                 #===================backward pass =============
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()
                #===================Validation =============
                v_loss = loss_fn(out[batch.val_mask], batch.y[batch.val_mask])
                val_loss += v_loss
                val_acc += self.eval_node_classifier(model, batch, batch.val_mask)

            #===================Print metrics every 10 epochs ============= 
            if(epoch % 10 == 0):
                print(f'Epoch {epoch:>3} | Train Loss: {train_loss/len(train_loader):.3f} '
                      f'| Train Acc: {train_acc/len(train_loader)*100:>6.2f}% | Val Loss: '
                      f'{val_loss/len(train_loader):.2f} | Val Acc: '
                      f'{val_acc/len(train_loader)*100:.2f}%')

                #===================Save metrics for plotting =============
                train_losses.append(train_loss.item()/len(train_loader))
                train_accuracies.append(train_acc/len(train_loader)*100)
                val_losses.append(val_loss.item()/len(train_loader))
                val_accuracies.append(val_acc/len(train_loader)*100)

        return model, train_losses, train_accuracies, val_losses, val_accuracies

    def eval_node_classifier(self, model, graph, mask):
        model.eval()
        pred_all_graph = model(graph).argmax(dim=1)
        pred_test_label = pred_all_graph[mask]
        true_test_label = graph.y[mask]
        correct_predictions = (pred_test_label == true_test_label).sum()
        accuracy = correct_predictions/mask.sum()
        return accuracy

    def test_node_classifier (self, model, graph, mask, measure, num_classes):
        model.eval()
        pred_all_graph = model(graph).argmax(dim=1)
        pred_test_label = pred_all_graph[mask]
        true_test_label = graph.y[mask]
        correct_predictions = (pred_test_label == true_test_label).sum()
        accuracy = correct_predictions/mask.sum()
        metric = MulticlassF1Score(num_classes = num_classes, average = measure)
        f1_score = metric(pred_test_label, true_test_label)
        return accuracy, f1_score
    
    @staticmethod
    def performance_report(performance_results, model_name, aggregator, learning_rate, neighbours, val_f1_score, \
                           test_f1_score, training_time):
        '''
        '''
        performance_results.append({
            "model_name": model_name,
            "aggregator": aggregator,
            "top_neighbours": neighbours,
            "learning_rate": learning_rate,
            "val_micro_f1": val_f1_score,
            "test_micro_f1": test_f1_score,
            "training_time": training_time
        })
        return performance_results
    
    def main(self):
        '''
        This is good
        '''
        model_ = self.train_node_classifier(self.model, self.train_loader, self.optimiser, self.loss_fn, self.n_epochs)
        test_accuracy, test_f1_score = self.test_node_classifier (self.model, self.graph, self.graph.test_mask, self.measure, self.num_classes)
        val_accuracy, val_f1_score = self.test_node_classifier (self.model, self.graph, self.graph.val_mask, self.measure, self.num_classes)
        print(f'Val f1_score: {val_f1_score:.3f}')
        print(f'Test f1_score: {test_f1_score:.3f}')
        
        return model_, val_f1_score, test_f1_score