import ftplib
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import torch.nn.functional as F

from tqdm import tqdm
import wandb

from sklearn import metrics
from scipy.optimize import brentq
from scipy.interpolate import interp1d


from datasets import TestDataset

GPU = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
CPU = "cpu"

class Tester:
    
    def __init__(self, model, dataset:TestDataset, batch_size):
        
        self.model = model
        self.dataset = dataset
        self.loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        
    def prepareEmbedding(self):
        data_dict = self.dataset.getAllFeature()
        self.embs = {}
        
        for audio_id, features in tqdm(data_dict.items(), desc="getting embeddings"):
            
            #ft = features.to(GPU)
            self.model(features)
            
            emb = self.model.getEmbedding() # 2차원 데이터
            emb = torch.sum(emb, dim=0)# 여러개의 feature들의 임베딩 누적값, 각 요소가 커도 38정도
            self.embs[audio_id] = emb.to(CPU)
            
            
    def idListToEmbListTensor(self, id_list):
        """
        [
            [ embedding 1],
            [ embedding 2],
            ...
        ]
        """
        result = []
        
        
        
        for audio_id in id_list:
            result.append(self.embs[audio_id])
            
        return torch.stack(result, dim=0)
    
        
        
    def getEER(self, labels, cos_sims):
        labels = labels.to(CPU)
        cos_sims = cos_sims.to(CPU)
        fpr, tpr, _ = metrics.roc_curve(labels, cos_sims, pos_label=1)
        eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
        return eer
    
    def test(self, epoch):
        
        self.model.eval()
        
        self.prepareEmbedding()
        
        sims = []
        labels = []
        
        for audio_id1, audio_id2, label in tqdm(self.loader, desc="testing"):
            
            embs1 = self.idListToEmbListTensor(audio_id1).to(GPU) # 2차원 [id, node_idx]
            embs2 = self.idListToEmbListTensor(audio_id2).to(GPU)
            
            sim = F.cosine_similarity(embs1, embs2, dim = 1).to(CPU) # 1차원 형태
            #print(sim)
            sims.append(sim)
            labels.append(label)
            
        sims = torch.concat(sims, dim = 0)
        labels = torch.concat(labels, dim = 0)
        eer = self.getEER(labels, sims)
        print(f"epoch: {epoch}, EER: {eer}")
        wandb.log({"EER by epoch" : eer})