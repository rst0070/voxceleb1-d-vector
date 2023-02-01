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
        self.drop_out = nn.Dropout(0.25)
        
        self.layer1 = nn.Sequential(
            nn.Linear(self.input_size, 6416), 
            nn.BatchNorm1d(6416),
            self.leaky_relu
            #nn.MaxPool1d(2, stride = 1), # 6415
        )
        
        self.layer2 = nn.Sequential(
            nn.Linear(6415, 1604),
            nn.BatchNorm1d(1604),
            #nn.MaxPool1d(2), # 128
            self.leaky_relu
        )
        
        self.layer3 = nn.Sequential(
            nn.Linear(1604, 401),
            nn.BatchNorm1d(401),
            self.leaky_relu,
            self.drop_out
        )
        
        self.layer4 = nn.Sequential(
            nn.Linear(401, 401),
            nn.BatchNorm1d(401),
            self.leaky_relu,
            self.drop_out
        )
        
        self.out_layer = nn.Linear(401, output_size)
        
    def forward(self, x):
        x = x.to(GPU)
        x = self.flatten(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        self.embedding = x.detach()
        
        x = self.out_layer(x)
        
        return x
    
    def getEmbedding(self):
        return self.embedding