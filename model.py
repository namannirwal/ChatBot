import torch
import torch.nn as nn
# PyTorch is an open source machine learning framework.
# PyTorch provides the torch.nn module to help us in creating and training of the neural network.

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()     # super is used to access parent class functionalities.

        # nn.Linear(n,m) is a module that creates single layer feed forward network with n inputs and m output.
        self.l1 = nn.Linear(input_size, hidden_size) 
        self.l2 = nn.Linear(hidden_size, hidden_size) 
        self.l3 = nn.Linear(hidden_size, num_classes)
       
        self.relu = nn.ReLU()  # Relu is a activation function which converts negative values to 0,and positive as it is.
    
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)  # Activation function

        out = self.l2(out)
        out = self.relu(out)   # Activation Function

        out = self.l3(out)
        
        return out
