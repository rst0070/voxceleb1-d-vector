import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from NeuralNetModel import NeuralNetModel
from dataset.TrainDataset import TrainDataset
import wandb

GPU = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
CPU = "cpu"

class Trainer:
    """
    Train에 필요한 data, loss function, optimizer등을 정의한다.  
    """
    
    def __init__(self, model:NeuralNetModel, train_annotation_path:str, data_dir:str, batch_size:int):
        
        self.model = model
        
        self.train_data = TrainDataset(annotations_file = train_annotation_path, audio_dir = data_dir)
        self.dataloader = DataLoader(self.train_data, batch_size=batch_size, shuffle=True, num_workers=1)
        
        # loss function
        # CEE에 넣기 전에 softmax를 통과시키지 않아도 된다.
        # 혹시 모르니 일단 model.forward에 softmax를 구현해두었다.
        self.loss_fn = nn.CrossEntropyLoss().to(GPU)
        
        # Adam algorithm으로 optimizer 설정
        """
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr = 1e-3,
            weight_decay=1e-5
        )
        """
        self.optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        
        
        
        # learning rate가 epoch마다 0.95%씩 감소하도록 설정
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=1,
            gamma=0.95
        )
        
        self.weights_std_table = wandb.Table(columns = ["layer1", "layer2", "layer3", "layer4", "layer5", "layer6"])
    
    def getLearningRate(self):
        for parameters in self.optimizer.param_groups:
            if 'lr' in parameters:
                return parameters['lr']
        
        
    def train(self, epoch):
        # train mode
        self.model.train()
        loss = None
        for batch_idx, (x, ans) in enumerate(self.dataloader):
            
            self.optimizer.zero_grad()
            #x = x.squeeze(1)
            x, ans = x.to(GPU), ans.to(GPU)
        
            # 출력값, 오차
            output = self.model(x)
        
            #print(f"output: {output.device} ans: {ans.device}")
            # print(f"shape of output: {output.shape}, ans: {ans.shape}")
            loss = self.loss_fn(output, ans)
        
            # gradient값 새롭게 갱신
            # 해당 gradient로 parameter들 optimizer로 변경
            
            loss.backward()
            self.optimizer.step()
            
            
        # 가중치들의 표준편차 테이블 형식으로 log
        l1, l2, l3, l4, l5, l6 = self.model.getWeightsStd()
        self.weights_std_table.add_data(l1, l2, l3, l4, l5, l6)
        
        wandb.log({
            "learning rate by epoch" : self.getLearningRate(),
            "training_loss by epoch" : loss,
            "weights std table by epoch": self.weights_std_table
        })
        # learning rate 낮추기
        self.lr_scheduler.step()
        
        
        
        