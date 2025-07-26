from torch import nn
from torch.optim.optimizer import Optimizer
from torch.utils.data.dataloader import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
from copy import deepcopy
from pathlib import Path
import datetime
import torch

class Trainer:
    def __init__(self,
                 num_epoch:int,
                 model:nn.Module,
                 optimizer:Optimizer,
                 device,
                 train_dataloader:DataLoader,
                 valid_dataloader:DataLoader,
                 save_dir,
                 lr_scheduler = None,
                 patience:int = 1e9,
                 eps:float = 1e-2
                ):
        
        self.num_epoch = num_epoch
        self.device = device
        self.model = model.to(device)
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        
        self.base_save_dir = Path(save_dir)
        self.epoch_train_loss = []
        self.epoch_valid_loss = []
        
        self.patience = patience
        self.eps = eps
        self.best_loss = 1e9
        self.best_model = deepcopy(model).cpu()
        
    def _train_epoch(self):
        self.model.train()
        
        loss_sum = 0
        
        with tqdm(enumerate(self.train_dataloader), total=len(self.train_dataloader), leave=False, ncols=80) as t_pbar:
            for step, batch in t_pbar:
                self.optimizer.zero_grad()
                
                loss = self.model(**batch)
                
                loss.backward()
                self.optimizer.step()
                
                t_pbar.set_description(f"step-{step} t_loss-{loss.item():.4f}")
                loss_sum += loss.item()
                
                # total_norm = 0
                # for p in self.model.parameters():
                #     if p.grad is not None:
                #         total_norm += p.grad.data.norm(2).item()**2
                #     total_norm = total_norm**0.5
                # print(f"Step {step}: grad_norm={total_norm:.4f}")
                
        if self.lr_scheduler != None:
            self.lr_scheduler.step()
                
        avg_loss = loss_sum / len(self.train_dataloader)
        self.epoch_train_loss.append(avg_loss)
        
        return avg_loss
        
    def _valid_epoch(self):
        self.model.eval()
        
        loss_sum = 0
        
        with torch.no_grad():
            with tqdm(enumerate(self.valid_dataloader), total=len(self.valid_dataloader), leave=False, ncols=80) as v_pbar:
                for step, batch in v_pbar:
                    loss = self.model(**batch)
                    
                    v_pbar.set_description(f"step-{step} v_loss-{loss.item():.4f}")
                    loss_sum += loss.item()
                    
        avg_loss = loss_sum / len(self.valid_dataloader)
        self.epoch_valid_loss.append(avg_loss)
        
        return avg_loss
        
    def train(self):
        
        accumulated_patience = 0
        
        with trange(self.num_epoch, ncols=80) as pbar:
            for epoch in pbar:
                t_loss = self._train_epoch()
                
                v_loss = self._valid_epoch()
                
                pbar.set_description(f"epoch-{epoch} t_loss-{t_loss:.4f} v_loss-{v_loss:.4f}")
                
                if self.best_loss - v_loss >= self.eps:
                    accumulated_patience = 0
                    self.best_loss = v_loss
                    self.best_model.load_state_dict(self.model.state_dict())
                else:
                    accumulated_patience += 1
                    
                if accumulated_patience >= self.patience:
                    tqdm.write("触发早停，训练停止")
                    break
        
        print(f"训练结束，最佳验证集损失：{self.best_loss:.4f}")
        self._save_and_plot()
                
    def _create_experiment_dir(self) -> Path:
        """创建新的实验目录"""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        experiment_dir = self.base_save_dir / f"exp_{timestamp}"
        experiment_dir.mkdir(parents=True, exist_ok=True)
        return experiment_dir
                
    def _save_and_plot(self):
        save_dir = self._create_experiment_dir()
        print(f"结果将保存到{save_dir}")
        plt.figure()
        plt.plot(range(len(self.epoch_train_loss)), self.epoch_train_loss, label="Train Loss")
        plt.plot(range(len(self.epoch_valid_loss)), self.epoch_valid_loss, label="Valid Loss")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.title("epoch-loss")
        plt.legend()
        plt.savefig(save_dir / "epoch-loss.png")
        plt.show()
        
        torch.save(self.best_model.state_dict(),save_dir / "model.pth")
                