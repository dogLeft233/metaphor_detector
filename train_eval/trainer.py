from torch.optim.optimizer import Optimizer
from torch.utils.data.dataloader import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
from sklearn.metrics import accuracy_score, f1_score
from copy import deepcopy
from pathlib import Path
from torch import nn
import datetime
import torch
import torch.nn.functional as F
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
                
                if self.lr_scheduler != None:
                    self.lr_scheduler.step()
                
                t_pbar.set_description(f"step-{step} t_loss-{loss.item():.4f}")
                loss_sum += loss.item()
                
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

class TrainerBinary:
    def __init__(self,
                 num_epoch:int,
                 model:nn.Module,
                 optimizer:Optimizer,
                 device,
                 train_dataloader:DataLoader,
                 valid_dataloader:DataLoader,
                 save_dir,
                 test_dataloader:DataLoader = None,
                 exp_dir = None,
                 lr_scheduler = None,
                 patience:int = 1e9,
                 eps:float = 1e-2,
                 log_fn = print
                ):
        
        self.num_epoch = num_epoch
        self.device = device
        self.model = model.to(device)
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.test_dataloader = test_dataloader
        
        self.base_save_dir = Path(save_dir)
        self._exp_dir = exp_dir
        
        self.epoch_train_loss = []
        self.epoch_valid_loss = []
        
        self.patience = patience
        self.eps = eps
        self.best_f1 = -1e9
        self.best_epoch = -1
        self.best_model = deepcopy(model).cpu()
        
        self.log_fn = log_fn
        
    @property
    def exp_dir(self):
        if self._exp_dir == None:
            self._exp_dir =self._create_experiment_dir()
        return self._exp_dir
        
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
                
        if self.lr_scheduler != None:
            self.lr_scheduler.step()
                
        avg_loss = loss_sum / len(self.train_dataloader)
        self.epoch_train_loss.append(avg_loss)
        
        return avg_loss
        
    def _valid_epoch(self, epoch:int):
        self.model.eval()
        
        loss_sum = 0
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            with tqdm(enumerate(self.valid_dataloader), total=len(self.valid_dataloader), leave=False, ncols=80) as v_pbar:
                for step, batch in v_pbar:
                    labels = batch['labels']
                    batch_inputs = {k: v for k, v in batch.items() if k != 'labels'}
                    
                    logits = self.model(**batch_inputs)
                    
                    loss = F.cross_entropy(logits, labels)
                    
                    v_pbar.set_description(f"step-{step} v_loss-{loss.item():.4f}")
                    loss_sum += loss.item()
                    preds = torch.argmax(logits, dim=-1)
                    all_preds.extend(preds.cpu().numpy().tolist())
                    all_labels.extend(labels.cpu().numpy().tolist())
        
        acc = accuracy_score(all_labels, all_preds)
        f1_pos = f1_score(all_labels, all_preds, pos_label=1)
        f1_neg = f1_score(all_labels, all_preds, pos_label=0)
        
        pos_count = sum(1 for label in all_labels if label == 1)
        neg_count = sum(1 for label in all_labels if label == 0)
        total_count = len(all_labels)
        
        f1_avg = (f1_pos * pos_count + f1_neg * neg_count) / total_count
        self.log_fn(f"Valid-epoch: {epoch} acc: {acc:.4f} f1_pos: {f1_pos:.4f} f1_neg: {f1_neg:.4f} f1_avg: {f1_avg:.4f} (pos:{pos_count}, neg:{neg_count})")
        
        avg_loss = loss_sum / len(self.valid_dataloader)
        self.epoch_valid_loss.append(avg_loss)
        
        return f1_avg
    
    def _test_epoch(self, epoch:int):
        self.model.eval()
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            with tqdm(enumerate(self.test_dataloader), total=len(self.test_dataloader), leave=False, ncols=80) as v_pbar:
                for step, batch in v_pbar:
                    labels = batch['labels']
                    batch_inputs = {k: v for k, v in batch.items() if k != 'labels'}
                    
                    logits = self.model(**batch_inputs)

                    preds = torch.argmax(logits, dim=-1)
                    all_preds.extend(preds.cpu().numpy().tolist())
                    all_labels.extend(labels.cpu().numpy().tolist())
        
        acc = accuracy_score(all_labels, all_preds)
        f1_pos = f1_score(all_labels, all_preds, pos_label=1)
        f1_neg = f1_score(all_labels, all_preds, pos_label=0)
        
        pos_count = sum(1 for label in all_labels if label == 1)
        neg_count = sum(1 for label in all_labels if label == 0)
        total_count = len(all_labels)
        
        f1_avg = (f1_pos * pos_count + f1_neg * neg_count) / total_count
        self.log_fn(f"Test-epoch: {epoch} acc: {acc:.4f} f1_pos: {f1_pos:.4f} f1_neg: {f1_neg:.4f} f1_avg: {f1_avg:.4f} (pos:{pos_count}, neg:{neg_count})")
        
        return f1_avg
        
    def train(self):
        
        accumulated_patience = 0
        
        with trange(self.num_epoch, ncols=80) as pbar:
            for epoch in pbar:
                t_loss = self._train_epoch()
                v_f1 = self._valid_epoch(epoch)
                
                if self.test_dataloader != None:
                    self._test_epoch(epoch)
                
                pbar.set_description(f"epoch-{epoch} t_loss-{t_loss:.4f} v_f1-{v_f1:.4f}")
                
                self.log_fn(f"===========================================================")
                
                if  v_f1 - self.best_f1 >= self.eps:
                    accumulated_patience = 0
                    self.best_f1 = v_f1
                    self.best_epoch = epoch
                    self.best_model.load_state_dict(self.model.state_dict())
                else:
                    accumulated_patience += 1
                    
                if accumulated_patience >= self.patience:
                    self.log_fn("early stopping !!!")
                    break
        
        self.log_fn(f"best valid f1：{self.best_f1:.4f} best epoch : {self.best_epoch}")
        self._save_and_plot()
                
    def _create_experiment_dir(self) -> Path:
        """创建新的实验目录"""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        experiment_dir = self.base_save_dir / f"exp_{timestamp}"
        experiment_dir.mkdir(parents=True, exist_ok=True)
        return experiment_dir
                
    def _save_and_plot(self):
            
        self.log_fn(f"结果将保存到{self.exp_dir}")
        
        plt.figure()
        plt.plot(range(len(self.epoch_train_loss)), self.epoch_train_loss, label="Train Loss")
        plt.plot(range(len(self.epoch_valid_loss)), self.epoch_valid_loss, label="Valid Loss")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.title("epoch-loss")
        plt.legend()
        plt.savefig(self.exp_dir / "epoch-loss.png")
        
        torch.save(self.best_model.state_dict(),self.exp_dir / "model.pth")