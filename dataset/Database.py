import torch
import torchaudio
import torchaudio.transforms as T

"""
어떻게 전처리 속도를 늘릴 수 있을까..????

데이터 전부를 VRAM에 저장시킬 순 없다. 
따라서 전처리 동작때만 VRAM에 가져오고, 처리후에는 RAM에 저장하는 방식을 사용해 보자.(RAM은 알아서 swap도 해주니까 그냥 저장해도 되겠지?) 
걱정되는 부분은 VRAM과 RAM간의 이동이 얼마나 시간이 필요한지 모르겠다는것. 전처리를 빠르게 하는것이 우선인지 데이터 이동을 빠르게 하는게 우선인지 모르겠다. 
"""


GPU = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
CPU = "cpu"

NUM_ENROLLED_SPEAKER = 1211
NUM_FRAMES = 16000 * 2
#  이부분 수정할 수 도?
SAMPLE_RATE = 16000

MEL_CONFIG = {
    'n_fft' : 1024,
    'win_length' : 1024,
    'hop_length' : 512,
    'n_mels' : 40
}


# Mel spectogram을 기본 transformer로 사용한다.
# 각 파라미터에 대해서 좀더 찾아봐야함
WAVEFORM_TRANSFORMER = T.MelSpectrogram(
    sample_rate = SAMPLE_RATE,
    n_fft= MEL_CONFIG['n_fft'],
    win_length = MEL_CONFIG['win_length'],
    hop_length = MEL_CONFIG['hop_length'],
    center=True,
    pad_mode="reflect",
    power=2.0,
    norm="slaney",
    onesided=True,
    n_mels = MEL_CONFIG['n_mels'],
    mel_scale="htk",
).to(GPU)


AUDIO_CACHE = {}
"""
전처리가 완료된 오디오 tensor를 dictionary 형태로 저장해둔다.
`audio_path:str : Tensor` 로 매칭됨.
"""


LABEL_CACHE = {}
"""
정답 레이블의 one hot vector 형태를 딕셔너리로 저장함.
`speaker_num : tensor` 로 매칭됨
"""


def speakerNumToTensor(speaker:int):
    """_summary_
    speaker의 번호를 one-hot vector 방식으로 변환한다. 이때 tensor를 반환.
    [0., 0., ..., 1.,... , 0.]
    """
    if speaker in LABEL_CACHE:
        return LABEL_CACHE[speaker]
    
    LABEL_CACHE[speaker] = torch.zeros(NUM_ENROLLED_SPEAKER)
    LABEL_CACHE[speaker][speaker-1] = 1.
    return LABEL_CACHE[speaker]


def resizeWaveform(waveform:torch.Tensor):
    """_summary_
    audio data를 정해진 frame 개수에 맞게 변환한다. 
    이때 긴것은 자르고, 짧은것은 이어붙인다.
    """
    _, n_frames_wf = waveform.shape
        
    # 길이 같은 경우
    if n_frames_wf == NUM_FRAMES:
        return waveform
        
    # 길이 짧은 경우
    if n_frames_wf < NUM_FRAMES:
        residue = NUM_FRAMES % n_frames_wf
        tensor_list = []
            
        for i in range(0, NUM_FRAMES // n_frames_wf):
            tensor_list.append(waveform)
        if residue > 0:
            tensor_list.append(waveform[:, 0:residue])
                
        return torch.cat(tensor_list, 1)
     # 길이 긴 경우
    return waveform[:, 0:NUM_FRAMES]

def transformWaveform(audio_path:str) -> torch.Tensor:
    """_summary_
    오디오파일의 경로를 받으면 해당 waveform을 transform까지 해서 반환한다. 
    경로에 대한 캐시를 사용한다.
    Args:
        audio_path (str): 오디오 파일이 있는 경로
        
    """
    if audio_path in AUDIO_CACHE:
        return AUDIO_CACHE[audio_path]
    
    waveform, sample_rate = torchaudio.load(audio_path)
    
    # 연산할때는 VRAM에
    waveform = waveform.to(GPU) 
    waveform = resizeWaveform(waveform) 
    tensor = WAVEFORM_TRANSFORMER(waveform)
    
    # 중간에 쓸데없이 크기가 1인 차원이 존재해서 없애주는것.(squeeze)    
    # 저장할때는 RAM에 
    AUDIO_CACHE[audio_path] = tensor.squeeze().to(CPU)
    # print(AUDIO_CACHE[audio_path].shape, sample_rate)
    return AUDIO_CACHE[audio_path]