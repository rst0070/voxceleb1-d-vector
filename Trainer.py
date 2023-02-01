import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformer import NUM_SEG_PER_UTTER
#import wandb

GPU = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
CPU = "cpu"


class Trainer:
    
    def __init__(self, model, dataset, batch_size):
        
        self.model = model
        
        self.loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        
        self.optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        
        self.loss_fn = nn.CrossEntropyLoss().to(GPU)
        
        # learning rate가 epoch마다 0.95%씩 감소하도록 설정
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=1,
            gamma=0.95
        )

    
    def train(self, epoch):
        
        self.model.train()
        
        for utter, ans in tqdm(self.loader):
            
            seg = random.randint(0, NUM_SEG_PER_UTTER - 1)
            feature = utter[seg]
            
            output = self.model(feature)
            self.optimizer.zero_grad()
            loss = self.loss_fn(output, ans)
            loss.backward()
            self.optimizer.step()
            break
            
        self.lr_scheduler.step()