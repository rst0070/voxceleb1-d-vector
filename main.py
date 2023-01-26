from NeuralNetModel import NeuralNetModel
from Trainer import Trainer
from Tester import Tester
import wandb
import torch
import os


class Main:
    
    def __init__(self, max_epoch, batch_size):
        """
        프로그램이 사용하는 경로들을 이용해 trainer, tester등을 초기화시킨다.  
        간혹 class 밖에 상수를 정의하다 참조하지 못하는 상황이있어 상수를 생성자 내부에 위치시켰다. 
        (main을 import할 수 없으니 상수는 main에서만 사용함)
        """
        
        GPU = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        CPU = "cpu"
        
        print(f"using {GPU} device")
        
        TRAIN_DIR = "/data/train"
        TEST_DIR = "/data/test"
        TRAIN_LABEL = "/app/train_label.csv"
        TEST_LABEL = "/data/trials/trials.txt"
        WANDB_KEY = "7c6c025691da1f01124a2b61a50c7c2932f0fb85"
        
        os.system(f"wandb login {WANDB_KEY}")
        wandb.init(
            project = "Voxceleb1 D-Vector",
            name = "momentum"
        )
        
        self.max_epoch = max_epoch
        self.model = NeuralNetModel().to(GPU)
        
        self.trainer = Trainer(model = self.model, train_annotation_path = TRAIN_LABEL, data_dir = TRAIN_DIR, batch_size = batch_size)
        self.tester = Tester(model = self.model, test_annotation_path = TEST_LABEL, data_dir = TEST_DIR, batch_size = batch_size)
    
    def start(self):
        """
        program 시작.
        train, test 시킨다.
        """
        
        print("program started!")
        for epoch in range(1, self.max_epoch + 1):
            print(f"epoch {epoch}")
            self.trainer.train(epoch)
            self.tester.test(epoch)
            
    def save(self):
        torch.save(self.model, "/result.pth")

if __name__ == '__main__':
    torch.multiprocessing.set_start_method("spawn")
    program = Main(max_epoch=100, batch_size=50)
    program.start()
    program.save()


