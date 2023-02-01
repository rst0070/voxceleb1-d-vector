import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets import TrainDataset
import wandb
import transformer
#import wandb

GPU = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
CPU = "cpu"


class Trainer:
    """
    Trainer는 TrainDataset으로 부터 input size만큼의 waveform을 랜덤으로 받고,
    batch를 실행하기전에 mel spec을 통해 특징을 추출한다.
    이를 이용해 학습을 진행한다.
    """
    
    def __init__(self, model, dataset:TrainDataset, batch_size):
        
        self.model = model
        
        self.loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr = 0.001,
            weight_decay=1e-5
        )

        
        self.loss_fn = nn.CrossEntropyLoss().to(GPU)
        
        # learning rate가 epoch마다 0.95%씩 감소하도록 설정
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=1,
            gamma=0.95
        )

    
    def train(self, epoch):
        
        self.model.train()
        loss = None
        for utter, ans in tqdm(self.loader, desc="training"):
            
            feature = transformer.transform(utter.to(GPU))
            
            self.optimizer.zero_grad()
            output = self.model(feature)
            loss = self.loss_fn(output, ans.to(GPU))
            loss.backward()
            self.optimizer.step()
            #break
        
        print(f"epoch: {epoch}, loss: {loss}")    
        wandb.log({"loss by epoch" : loss})
        self.lr_scheduler.step()