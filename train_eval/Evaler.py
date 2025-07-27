import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

class Evaler:
    def __init__(self, model, dataloader: DataLoader, device, log_fn=print):
        self.model = model.to(device)
        self.dataloader = dataloader
        self.device = device
        self.log_fn = log_fn

    def evaluate(self):
        self.model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch in tqdm(self.dataloader, desc="Evaluating", ncols=80):
                labels = batch['labels']
                batch_inputs = {k: v for k, v in batch.items() if k != 'labels'}
                
                logits = self.model(**batch_inputs)
                if hasattr(logits, 'logits'):
                    logits = logits.logits
                    
                preds = torch.argmax(logits, dim=-1)
                all_preds.extend(preds.cpu().numpy().tolist())
                all_labels.extend(labels.cpu().numpy().tolist())
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds)
        self.log_fn(f"Accuracy: {acc:.4f}")
        self.log_fn(f"F1 Score: {f1:.4f}")
        return acc, f1
