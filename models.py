import torch 
import torch.nn as nn


class MLP(torch.nn.Module):
    #single task classification
    def __init__(self, input_size, hidden_size, out_classes):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.hidden_size  = hidden_size
        self.out_classes = out_classes
        self.relu = torch.nn.ReLU()
       
        self.fc2 = torch.nn.Linear(self.input_size, self.out_classes)

    def forward(self, x):
        
        x = self.fc2(x)
        return x


class MLP2(torch.nn.Module):
    #single task classification
    def __init__(self, input_size, hidden_size, out_classes):
        super(MLP2, self).__init__()
        self.input_size = input_size
        self.hidden_size  = hidden_size
        self.out_classes = out_classes
        self.relu = torch.nn.ReLU()
        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
        self.inter = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.fc2 = torch.nn.Linear(self.hidden_size, self.out_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        
        return x