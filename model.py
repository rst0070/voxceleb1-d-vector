import torch
import torch.nn as nn

GPU = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
CPU = "cpu"
        
class NeuralNet(nn.Module):
    
    def __init__(self, input_size, output_size):
        super().__init__()
        self.input_size = input_size # 25664
        self.output_size = output_size
        
        self.flatten = nn.Flatten()
        self.leaky_relu = nn.LeakyReLU() # https://en.wikipedia.org/wiki/Rectifier_(neural_networks)
        self.max_pool_
        
        self.layer1 = nn.Sequential(
            nn.Linear(self.input_size, 1024), 
            nn.BatchNorm1d(1024),
            self.leaky_relu,
            nn.MaxPool1d(2),
            nn.Dropout(0.25)
        )
        
        self.layer2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            self.leaky_relu,
            nn.MaxPool1d(2),
            nn.Dropout(0.25)
        )
        
        self.layer3 = nn.Sequential(
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            self.leaky_relu,
            nn.Dropout(0.25)
        )
        
        
        self.out_layer = nn.Linear(128, output_size)
        
    def forward(self, x):
        x = x.to(GPU)
        x = self.flatten(x)
        
        y1 = self.layer1(x)
        y2 = self.layer2(y1)
        y3 = self.layer3(y2)
        
        self.embedding = y3.detach()
        
        y4 = self.out_layer(y3)
        
        return x
    
    def getEmbedding(self):
        return self.embedding