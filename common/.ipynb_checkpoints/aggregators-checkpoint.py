import torch
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import global_max_pool
from torch.nn import LSTM


class MeanAggregator(MessagePassing):
    def __init__(self):
        super(MeanAggregator, self).__init__(aggr='mean')

    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=x)

    def message(self, x_j):
        return x_j

    def update(self, aggr_out):
        return aggr_out


class SumAggregator(MessagePassing):
    def __init__(self):
        super(SumAggregator, self).__init__(aggr='add')

    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=x)

    def message(self, x_j):
        return x_j

    def update(self, aggr_out):
        return aggr_out


class MaxAggregator(MessagePassing):
    def __init__(self):
        super(MaxAggregator, self).__init__(aggr='max')

    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=x)

    def message(self, x_j):
        return x_j

    def update(self, aggr_out):
        return aggr_out


class LearnableAggregator(MessagePassing):
    def __init__(self, in_channels):
        super(LearnableAggregator, self).__init__()
        self.linear = torch.nn.Linear(in_channels, in_channels)
        
    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=x)

    def message(self, x_j):
        return self.linear(x_j)

    def update(self, aggr_out):
        return aggr_out


class MaxPoolAggregator(torch.nn.Module):
    def forward(self, x, batch):
        return global_max_pool(x, batch)


class LSTMAggregator(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(LSTMAggregator, self).__init__()
        self.lstm = LSTM(input_size=in_channels, hidden_size=hidden_channels, batch_first=True)

    def forward(self, x, batch):
        # Prepare input for LSTM: (batch_size, sequence_length, input_size)
        # Assuming that each node has a sequence length of 1
        x = x.unsqueeze(1)  # Add sequence length dimension
        lstm_out, (h_n, _) = self.lstm(x)  # Get the hidden state of the last time step
        return h_n[-1]  # Return the last hidden state

if __name__ == "__main__":
    print("Aggregator classes defined successfully.")
