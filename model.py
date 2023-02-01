import torch
import torch.nn as nn

GPU = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
CPU = "cpu"
        
class NeuralNet(nn.Module):
    
    def __init__(self, input_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        
        self.flatten = nn.Flatten()
        self.leaky_relu = nn.LeakyReLU() # https://en.wikipedia.org/wiki/Rectifier_(neural_networks)
        self.drop_out = nn.Dropout(0.25)
        
        self.layer1 = nn.Sequential(
            nn.Linear(self.input_size, 2560),
            nn.BatchNorm1d(2560),
            nn.MaxPool1d(2), # 1280
            self.leaky_relu
        )
        
        self.layer2 = nn.Sequential(
            nn.Linear(1280, 256),
            nn.BatchNorm1d(256),
            nn.MaxPool1d(2), # 128
            self.leaky_relu
        )
        
        self.layer3 = nn.Sequential(
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            self.leaky_relu,
            self.drop_out
        )
        
        self.layer4 = nn.Sequential(
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            self.leaky_relu,
            self.drop_out
        )
        
        self.out_layer = nn.Linear(128, output_size)
        
    def forward(self, x):
        x = x.to(GPU)
        x = self.flatten(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        self.embedding = x
        
        x = self.out_layer(x)
        
        return x