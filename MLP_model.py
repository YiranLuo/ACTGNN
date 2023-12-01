import torch
import torch.nn as nn

# A simple 3-layer MLP (input-hidden-output)
# class MLP(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size):
#         super(MLP, self).__init__()
#         self.input_layer = nn.Linear(input_size, hidden_size)
#         self.relu = nn.ReLU()
#         self.hidden_layer = nn.Linear(hidden_size, output_size)
#         # self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         x = self.relu(self.input_layer(x))
#         # x = self.hidden_layer(x)
#         # x = self.sigmoid(x)
#         x = self.sigmoid(self.hidden_layer(x))
#         return x

# A 4-layer MLP
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(MLP, self).__init__()
        self.input_layer = nn.Linear(input_size, hidden_size1)
        self.relu1 = nn.ReLU()
        self.hidden_layer1 = nn.Linear(hidden_size1, hidden_size2)
        self.relu2 = nn.ReLU()
        self.hidden_layer2 = nn.Linear(hidden_size2, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu1(self.input_layer(x))
        x = self.relu2(self.hidden_layer1(x))
        # x = self.hidden_layer(x)
        # x = self.sigmoid(x)
        x = self.sigmoid(self.hidden_layer2(x))
        # x = self.hidden_layer2(x)
        return x