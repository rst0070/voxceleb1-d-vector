from email.mime import audio
import torch
import torchaudio
import torchaudio.transforms as T
import random

"""
어떻게 전처리 속도를 늘릴 수 있을까..????

데이터 전부를 VRAM에 저장시킬 순 없다. 
따라서 전처리 동작때만 VRAM에 가져오고, 처리후에는 RAM에 저장하는 방식을 사용해 보자.(RAM은 알아서 swap도 해주니까 그냥 저장해도 되겠지?) 
걱정되는 부분은 VRAM과 RAM간의 이동이 얼마나 시간이 필요한지 모르겠다는것. 전처리를 빠르게 하는것이 우선인지 데이터 이동을 빠르게 하는게 우선인지 모르겠다. 

가장긴 frame: 2318721
sample rate: 16000

frame을 32000으로 해야함
모든 audio를 72개로 나눠서 저장하자

"""


GPU = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
CPU = "cpu"

NUM_ENROLLED_SPEAKER = 1211
NUM_FRAMES = 16000 * 2 # waveform의 길이

NUM_WINDOW = 72
"""spectogram화된 발성을 시간에 대해 NUM_WINDOW 개수만큼 나누어 Tensor로 저장한다."""
LEN_WINDOW = 63
"""spectogram화된 발성을 나누는 시간단위이다."""


SAMPLE_RATE = 16000
"""모든 오디오파일이 16000의 sample rate를 갖는다."""

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
`AUDIO_CACHE[audio_path_str][window_idx]`형태로 접근가능하다.  
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

def resizeAudio(waveform:torch.Tensor):
    """_summary_
    waveform의 frame수는 NUM_FRAMES 이상이어야한다.
    NUM_FRAME , LEN_WINDOW 둘다 2초에 대한 길이이다.  
    신경망이 소리를 처리하는 기준이므로 이 규격을 지켜야함  
    """
    _, n_frames_wf = waveform.shape
    
    # 길이조정 필요없는경우
    if n_frames_wf >= NUM_FRAMES:
        return waveform
        
    residue = NUM_FRAMES % n_frames_wf
    tensor_list = []
            
    for i in range(0, NUM_FRAMES // n_frames_wf):
        tensor_list.append(waveform)
    if residue > 0:
        tensor_list.append(waveform[:, 0:residue])
                
    return torch.cat(tensor_list, 1)

def divideSpectrogram(spectrogram):
    """
    오디오에서 특징을 추출한 spectrogram은 크기가 제각각이다.(최소 LEN_WINDOW이상은 동일)  
    LEN_WINDOW길이만큼 NUM_WINDOW개로 나누어 사용해야한다.(모든 정보를 살리기 위해서) 
    NUM_WINDOW는 가장 긴 오디오파일을 LEN_WINDOW로 나누었을때 몫이다.  
    
    hop length를 정해야함....
    """
    _, len_origin = spectrogram.shape # 시간축 크기
    
    
    

def transformWaveform(audio_path:str) -> torch.Tensor:
    """_summary_
    오디오파일의 경로를 받으면 해당 waveform을 mel spectrogram화 해서 cache에 저장후 반환한다.
    이때 오디오의 길이를 최소 2초로 만들자.
    
    또한 오디오를 72개로 나눌것.(가장 큰 오디오파일이 2초*72임)
    테스트시 이 모든 72개에 대한 임베딩의 평균을 이용할 예정
    
    train시에는 72개중 하나를 random하게 뽑아서 사용할 예정
    """
    if audio_path in AUDIO_CACHE:
        return AUDIO_CACHE[audio_path]
    
    waveform, sample_rate = torchaudio.load(audio_path)
    
    # 연산할때는 VRAM에
    waveform = waveform.to(GPU) 
    waveform = resizeAudio(waveform) 
    tensor = WAVEFORM_TRANSFORMER(waveform)
    #
    
    # 중간에 쓸데없이 크기가 1인 차원이 존재해서 없애주는것.(squeeze)    
    # 저장할때는 RAM에 
    AUDIO_CACHE[audio_path] = tensor.squeeze().to(CPU)
    # print(AUDIO_CACHE[audio_path].shape, sample_rate)
    return AUDIO_CACHE[audio_path]


