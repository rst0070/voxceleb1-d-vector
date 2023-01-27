from NeuralNetModel import NeuralNetModel
from dataset.TestDataset import TestDataset
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from sklearn import metrics
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import wandb

GPU = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
CPU = "cpu"

class Tester:
    
    def __init__(self, model:NeuralNetModel, test_annotation_path:str, data_dir:str, batch_size:int):
        
        self.model = model
        
        self.dataset = TestDataset(annotations_file=test_annotation_path, audio_dir=data_dir)
        self.dataloader = DataLoader(dataset = self.dataset, batch_size=batch_size, shuffle = True, num_workers=1)
        
    #def findThreshold(self, signal1, siganl2, is_same_speaker):
        
        
    def getEER(self, labels, cos_sims):
        labels = labels.to(CPU)
        cos_sims = cos_sims.to(CPU)
        fpr, tpr, _ = metrics.roc_curve(labels, cos_sims, pos_label=1)
        eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
        return eer
    
    def test(self, epoch):
        
        # 평가모드
        self.model.eval()
        
        sims = []
        labels = []
        for batch_idx, ((signal1, id1, signal2, id2), label) in enumerate(self.dataloader):
            
            self.model(signal1.to(GPU))
            emb1 = self.model.getEmbedding()
            self.model(signal2.to(GPU))
            emb2 = self.model.getEmbedding()
            
            sim = F.cosine_similarity(emb1, emb2, dim=1)
            print("EMBEDDING SIM", sim, label)
            sims.append(sim)
            labels.append(label)
            
        
        sims = torch.concat(sims, dim = 0)
        labels = torch.concat(labels, dim = 0)
        
        eer = self.getEER(labels, sims)
        print(f"epoch: {epoch}, EER: {eer}")
        wandb.log({"EER by epoch" : eer})
            
        
        
        
        
        
        