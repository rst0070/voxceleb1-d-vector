import torch
import torch.nn as nn
from dataset.Database import NUM_ENROLLED_SPEAKER

class NeuralNetModel(nn.Module):
    
    def __init__(self, embedding_layer:int = -2):
        super().__init__()
        self.input_size = 128 * 121
        self.output_size = NUM_ENROLLED_SPEAKER
        self.hidden_size = (self.input_size + self.output_size) // 2 # https://stats.stackexchange.com/questions/181/how-to-choose-the-number-of-hidden-layers-and-nodes-in-a-feedforward-neural-netw
        self.embedding_layer = embedding_layer
        
        self.flatten = nn.Flatten() # 입력 tensor를 2차원으로 만드는 역할.
        
        # ReLU등의 활성화 함수는 매개변수학습을 하지 않는다. 
        # 따라서 활성화 함수 객체를 여러개 생성할 필요없이 같은걸 참조하게 하면된다.  
        # Sequential은 이를 자동으로 처리해준다.        
        self.network_sequence = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.BatchNorm1d(self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.BatchNorm1d(self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.BatchNorm1d(self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.BatchNorm1d(self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.output_size)
        )
        ## self.network_sequence[-2] : 마지막 은닉층의 활성화값
        
        #self.classification = nn.Softmax(dim = 1)
        
        root_2 = nn.init.calculate_gain('relu') # root(2)
        for layer in self.network_sequence:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight, root_2) # 가중치 초기값 설정: root(2 / (입력, 출력 사이즈 평균))
                layer.bias.data.fill_(0.01) # 편향 초기값 설정
        
        self.network_sequence[self.embedding_layer].register_forward_hook( self.embeddingHook )
        
        
    def embeddingHook(self, module, args, output):
        self.embedding = torch.clone(output.detach())
        #print(self.embedding)
        
    def forward(self, x):
        x = self.flatten(x)
        return self.network_sequence(x)
    
    def getEmbedding(self):
        #print(self.embedding)
        return self.embedding
    
    def getWeightsStd(self):
        weights = []
        for layer in self.network_sequence:
            if isinstance(layer, nn.Linear):
                weights.append(torch.std(layer.weight).item()) # 숫자로 변환
        return weights