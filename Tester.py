import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import torch.nn.functional as F

from tqdm import tqdm

from sklearn import metrics
from scipy.optimize import brentq
from scipy.interpolate import interp1d

from datasets import TestDataset

GPU = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
CPU = "cpu"

class Tester:
    
    def __init__(self, model, dataset:TestDataset):
        
        self.model = model
        self.dataset = dataset
        self.loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4)
        
    def prepareEmbedding(self):
        data_dict = self.dataset.getAllFeature()
        self.embs = {}
        
        for audio_id, features in data_dict.items():
            
            self.model(features)
            emb = self.model.getEmbedding()
            emb = torch.mean(emb, dim=0) # 여러개의 feature들의 임베딩 평균
            
            self.embs[audio_id] = emb
            
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
        for audio_id1, audio_id2, label in tqdm(self.loader):
            
            print(self.embs[audio_id1].shape)
            sim = F.cosine_similarity(self.embs[audio_id1], self.embs[audio_id2])
            
            sims.append(sim)
            labels.append(label)
            
        sims = torch.tensor(sims)
        labels = torch.tensor(label)
        eer = self.getEER(labels, sims)
        print(f"epoch: {epoch}, EER: {eer}")