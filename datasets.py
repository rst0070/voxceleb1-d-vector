import os
import torch
from torch.utils.data import Dataset
import torchaudio
import random
import pandas as pd
import transformer

"""_summary_

"""

NUM_ENROLLED_SPEAKER = 1211
GPU = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
CPU = "cpu"

class TrainDataset(Dataset):
    """_summary_
    train annotation file은 (id_num, id, audio_path) 형태로 구성되어있다. 
    id_num을 1차원 벡터로
    audio path를 feature extracting 해야한다.
    """
    
    def __init__(self, annotations_file, audio_dir):
        super().__init__()
        self.labels = pd.read_csv(annotations_file, delim_whitespace=True)
        self.num_utter = len(self.labels)
        self.cache = []

        for r_idx in range(0, self.num_utter):
            path = audio_dir + '/' + self.labels.iloc[r_idx, 2]
            
            waveform, _ = torchaudio.load(path)
            features = transformer.transform(waveform.to(GPU)).to(CPU)
            
            speaker_num = int(self.labels.iloc[r_idx, 0])
            one_hot_v = torch.zeros(NUM_ENROLLED_SPEAKER, dtype = float)
            one_hot_v[speaker_num - 1] = 1.
            
            self.cache.append((features, one_hot_v))
            
    def __len__(self):
        return self.num_utter
    
    def __getitem__(self, idx):
        return self.cache[idx]
    
class TestDataset(Dataset):
    """_summary_
    직접 발성에 대한 특징을 주는게 아니라 각 오디오파일의 id를 넘겨준다. 
    
    """
    
    def __init__(self, annotations_file, audio_dir):
        super().__init__()
        self.labels = pd.read_csv(annotations_file, delim_whitespace=True)
        self.num_label = len(self.labels)
        self.cache = []
        self.all_feature = {}

        for r_idx in range(0, self.num_label):
            label = int(self.labels.iloc[r_idx, 0])
            # 오디오에 대한 id = path
            id1 = self.labels.iloc[r_idx, 1]
            id2 = self.labels.iloc[r_idx, 2]
            
            if id1 not in self.all_feature:
                path = self.audio_dir + '/' + id1
                wf, _ = torchaudio.load(path)
                self.all_feature[id1] = transformer.transform(wf.to(GPU)).to(CPU)
                
            if id2 not in self.all_feature:
                path = self.audio_dir + '/' + id2
                wf, _ = torchaudio.load(path)
                self.all_feature[id2] = transformer.transform(wf.to(GPU)).to(CPU)
            
            
            self.cache.append((id1, id2, label))
            
    def __len__(self):
        return self.num_label
    
    def __getitem__(self, idx:int):
        """
        """
        return self.cache[idx]
    
    def getAllFeature(self):
        return self.all_feature