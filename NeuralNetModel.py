import torch.nn as nn
from dataset.Database import NUM_ENROLLED_SPEAKER

class NeuralNetModel(nn.Module):
    
    def __init__(self, embedding_layer:int = -2):
        super().__init__()
        self.input_size = 128 * 121
        self.hidden_size = 148643 // (2*(self.input_size + NUM_ENROLLED_SPEAKER)) # https://stats.stackexchange.com/questions/181/how-to-choose-the-number-of-hidden-layers-and-nodes-in-a-feedforward-neural-netw
        self.flatten = nn.Flatten() # 입력 tensor를 2차원으로 만드는 역할.
        
        # ReLU등의 활성화 함수는 매개변수학습을 하지 않는다. 
        # 따라서 활성화 함수 객체를 여러개 생성할 필요없이 같은걸 참조하게 하면된다.  
        # Sequential은 이를 자동으로 처리해준다.        
        self.network_sequence = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, NUM_ENROLLED_SPEAKER)
        )
        
        self.network_sequence[embedding_layer].register_forward_hook( self.embeddingHook )
        
    def embeddingHook(self, module, args, output):
        self.embedding = output.detach()
    
    def forward(self, x):
        #print(f"shape1 : {x.shape}")
        x = self.flatten(x)
        #print(f"shape2 : {x.shape}")
        
        #print(f"shape3 : {128*41}, {self.hidden_size}")
        # logits는 softmax함수로 구해야하는게 아닌가 싶은데...
        # 일단 예시 코드대로 돌려봄
        logits = self.network_sequence(x)
        return logits
    
    def getEmbedding(self):
        return self.embedding