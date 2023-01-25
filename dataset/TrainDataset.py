"""_summary_
Audio Data를 불러오는 데이터셋을 정의한다.
이때 모든 audio file은 1초가 되도록 만든다.
"""
import os
import pandas as pd
from torch.utils.data import Dataset
import dataset.Database as db



class TrainDataset(Dataset):
    """_summary_
    train annotation file은 (id_num, id, audio_path) 형태로 구성되어있다. 
    id_num을 1차원 벡터로
    audio path를 feature extracting 해야한다.
    """
    
    def __init__(self, annotations_file, audio_dir):
        super().__init__()
        self.audio_labels = pd.read_csv(annotations_file, delim_whitespace=True)
        self.audio_dir = audio_dir
        
    def __len__(self):
        return len(self.audio_labels)
    
        
    
    def __getitem__(self, idx):
        r"""_summary_
        경로를 받아서 transform한 audio tensor, label을 반환한다.  
        label은 one hot vector 방식
        """
        #print(f"in train data set, {idx}")
        
        audio_path = os.path.join(self.audio_dir, self.audio_labels.iloc[idx, 2])
        
        tensor = db.transformWaveform(audio_path)            
        #print(tensor.shape)
        label = int( self.audio_labels.iloc[idx, 0] )

        label = db.speakerNumToTensor(label)
        
        return tensor, label