import os
import pandas as pd
from torch.utils.data import Dataset
import dataset.Database as db


class TestDataset(Dataset):
    
    def __init__(self, annotations_file, audio_dir):
        super().__init__()
        self.audio_labels = pd.read_csv(annotations_file, delim_whitespace=True)
        self.audio_dir = audio_dir
    
    def __len__(self):
        return len(self.audio_labels)
    
    def __getitem__(self, idx):
        """_summary_
        (signal1, signal2, True or False) 형태로 반환
        """
        label = self.audio_labels.iloc[idx, 0]
        path1 = os.path.join(self.audio_dir, self.audio_labels.iloc[idx, 1])# id10270
        path2 = os.path.join(self.audio_dir, self.audio_labels.iloc[idx, 2])
        
        signal1 = db.transformWaveform(path1)
        signal2 = db.transformWaveform(path2)
        id1 = self.audio_labels.iloc[idx, 1][0 : 7]
        id2 = self.audio_labels.iloc[idx, 1][0 : 7]
        return (signal1, id1, signal2, id2), label
    
        
        