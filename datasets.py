import os
import torch
from torch.utils.data import Dataset
import torchaudio
import random
import pandas as pd
import transformer
from tqdm import trange

"""_summary_

"""

NUM_ENROLLED_SPEAKER = 1211
GPU = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
CPU = "cpu"

class TrainDataset(Dataset):
    """_summary_
    train시에는 각 발성에 대한 전체 정보가 아니라 랜덤으로 자른 정보를 전달한다. 
    __getitem__ 호출시에 해당 화자에 대한 특정길이의 waveform을 랜덤으로 전달한다.
    """
    
    def __init__(self, annotations_file, audio_dir):
        
        """
        train annotation file은 (id_num, id, audio_path) 형태로 구성되어있다. 
        id_num을 1차원 벡터로 (one hot vector)
        audio path를 feature extracting 해야한다.
        """
        super().__init__()
        self.labels = pd.read_csv(annotations_file, delim_whitespace=True)
        self.num_utter = len(self.labels)
        self.cache = []

        for r_idx in trange(self.num_utter, desc="loading train data"):
            path = audio_dir + '/' + self.labels.iloc[r_idx, 2]
            
            wf, _ = torchaudio.load(path)
            wf = transformer.resizeWaveform(wf)
            
            speaker_num = int(self.labels.iloc[r_idx, 0])
            one_hot_v = torch.zeros(NUM_ENROLLED_SPEAKER, dtype = float)
            one_hot_v[speaker_num - 1] = 1.
            
            self.cache.append((wf, one_hot_v))
            
    def __len__(self):
        return self.num_utter
    
    def __getitem__(self, idx):
        """
        waveform에서 random으로 자른 부분(입력크기에 맞게)과 answer를 준다. 
        """
        wf, ans = self.cache[idx]
        _, n_fr = wf.shape
        
        start_fr = random.randint(0, n_fr - transformer.NUM_FRAME_PER_INPUT)
        return wf[:, start_fr : start_fr + transformer.NUM_FRAME_PER_INPUT], ans
    
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

        for r_idx in trange(self.num_label, desc="loading test data"):
            label = int(self.labels.iloc[r_idx, 0])
            # 오디오에 대한 id = path
            id1 = self.labels.iloc[r_idx, 1]
            id2 = self.labels.iloc[r_idx, 2]
            
            if id1 not in self.all_feature:
                path = audio_dir + '/' + id1
                wf, _ = torchaudio.load(path)
                wf = wf.to(GPU)
                
                self.all_feature[id1] = transformer.transformWithSplit(wf).to(CPU)
                
                del wf
                
            if id2 not in self.all_feature:
                path = audio_dir + '/' + id2
                wf, _ = torchaudio.load(path)
                wf = wf.to(GPU)
                self.all_feature[id2] = transformer.transformWithSplit(wf).to(CPU)

                del wf
            
            self.cache.append((id1, id2, label))
        
        print(len(self.all_feature))
            
    def __len__(self):
        return self.num_label
    
    def __getitem__(self, idx:int):
        """
        """
        return self.cache[idx]
    
    def getAllFeature(self):
        return self.all_feature