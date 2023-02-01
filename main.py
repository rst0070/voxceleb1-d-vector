from datasets import TrainDataset, TestDataset, NUM_ENROLLED_SPEAKER
from transformer import NUM_FILTER_PER_WINDOW, NUM_WINDOW_PER_INPUT
from trainer import Trainer
from tester import Tester
from model import NeuralNet
import torch
import os
import wandb


class Main:
    """
    생성자에서 프로그램이 필요한 데이터를 모두 로드 시킨다. 이를 통해 dataloader에서 많은 worker를 사용할 필요가 없어진다. 
    
    start를 통해서 epoch을 진행한다.
    
    save를 통해 현재 모델의 파라미터들을 저장한다.
    """
    
    def __init__(self, max_epoch, batch_size):
        
        self.max_epoch = max_epoch
        
        GPU = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        CPU = "cpu"

        TRAIN_DIR = "/data/train"
        TEST_DIR = "/data/test"
        TRAIN_LABEL = "train_label.csv"
        TEST_LABEL = "trial_label.csv"
        WANDB_KEY = "7c6c025691da1f01124a2b61a50c7c2932f0fb85"
        os.system(f"wandb login {WANDB_KEY}")
        wandb.init(
            project = "Voxceleb1 D-Vector",
            name = "Adam"
        )

        train_data = TrainDataset(TRAIN_LABEL, TRAIN_DIR)
        test_data = TestDataset(TEST_LABEL, TEST_DIR)
        
        self.model = NeuralNet(NUM_WINDOW_PER_INPUT * NUM_FILTER_PER_WINDOW, NUM_ENROLLED_SPEAKER).to(GPU)
        
        self.trainer = Trainer(self.model, train_data, batch_size)
        self.tester = Tester(self.model, test_data, batch_size)
        
    def start(self):
        print("program started!")
        for epoch in range(1, self.max_epoch + 1):
            print(f"epoch {epoch}")
            self.trainer.train(epoch)
            self.tester.test(epoch)

    def save(self):
        torch.save(self.model, "result.pth")

if __name__ == '__main__':
    torch.multiprocessing.set_start_method("spawn")
    program = Main(max_epoch=100, batch_size=500)
    program.start()
    program.save()
