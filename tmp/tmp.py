import torch
import torchaudio
import os
import os.path

def getInfo(path):
    waveform, sample_rate = torchaudio.load(path)
    _, n_frames_wf = waveform.shape
    
    return n_frames_wf, sample_rate
     
rates = []
max_len = 0
name = None
def find(path):
    if os.path.isdir(path):
        for path2 in os.listdir(path):
            find(os.path.join(path,path2))
        return
        
    if path.endswith('.wav'):
        
        n, s = getInfo(path)
        if s not in rates:
            rates.append(s)
        
        global max_len
        global name
        if n > max_len:
            max_len = n
            name = path
    
        
if __name__ == '__main__':
    torch.multiprocessing.set_start_method("spawn")
    n = find("/data")
    print(max_len, rates, name)  

    