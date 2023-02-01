import torch
import torchaudio
import torchaudio.transforms as T
"""
가장긴 frame 개수: 2318721
sample rate: 16000

4초 단위로 waveform을 자른다면 모든 발성들을 약 38개로 나누면 된다.
"""
GPU = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
CPU = "cpu"



SAMPLE_RATE = 16000
NUM_FRAME_PER_INPUT = 16000 * 4
NUM_FILTER_PER_WINDOW = 64
NUM_WINDOW_PER_INPUT = 401 #NUM_FRAME_PER_INPUT // 160
NUM_SEG_PER_UTTER = 38
""" 각각의 발성들을 NUM_SEG_PER_UTTER개로 나눈다. """

# Mel spectogram을 기본 transformer로 사용한다.
WAVEFORM_TRANSFORMER = T.MelSpectrogram(
    sample_rate = SAMPLE_RATE,
    n_fft= 512,
    win_length = 512,
    hop_length = 160,
    center=True,
    pad_mode="reflect",
    power=2.0,
    onesided=True,
    n_mels = NUM_FILTER_PER_WINDOW,
    mel_scale="htk"
).to(GPU)


def resizeWaveform(waveform:torch.Tensor):
    """_summary_
    waveform의 frame수는 NUM_FRAME_PER_INPUT 이상이어야한다.
    """
    _, n_frames_wf = waveform.shape
    
    # 길이조정 필요없는경우
    if n_frames_wf >= NUM_FRAME_PER_INPUT:
        return waveform
        
    residue = NUM_FRAME_PER_INPUT % n_frames_wf
    tensor_list = []
            
    for i in range(0, NUM_FRAME_PER_INPUT // n_frames_wf):
        tensor_list.append(waveform)
    if residue > 0:
        tensor_list.append(waveform[:, 0:residue])
                
    return torch.cat(tensor_list, 1)


def transform(waveform):
    tensor = WAVEFORM_TRANSFORMER(waveform)
    tensor = torch.squeeze(tensor, dim = 0)
    return tensor

def transformWithSplit(waveform):
    waveform = resizeWaveform(waveform)
    _, n_frames_wf = waveform.shape
    
    start_frs = torch.linspace(start = 0, end = n_frames_wf - NUM_FRAME_PER_INPUT, steps = NUM_SEG_PER_UTTER, dtype = int)
    
    tensor_list = []
    for start in start_frs:
        end = start + NUM_FRAME_PER_INPUT
        tensor = WAVEFORM_TRANSFORMER(waveform[:, start : end])
        tensor_list.append(torch.squeeze(tensor, dim=0))
    
    return torch.stack(tensor_list, dim = 0)