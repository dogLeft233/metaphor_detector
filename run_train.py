from torch.utils.data.dataloader import DataLoader
from torch.optim.adam import Adam
from torch.optim.adamw import AdamW
from model.CLIPW2V import GPTClassifier, Meteor
from data_processing.MultiMET.MultiMETDatasetTF import EmbeddedDataset, EmbeddedCollator
from train_eval.Trainer import Trainer
from train_eval.Evaler import Evaler
from utils.setting import set_random_seed
import torch

def main():
    set_random_seed(42)
    
    batch_size = 128
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_dataset = EmbeddedDataset("./data/archive/new_train.csv")
    valid_dataset = EmbeddedDataset("./data/archive/new_val.csv")
    collator = EmbeddedCollator(device)
    
    model = GPTClassifier(dropout=0.2)
    
    num_epoch = 200
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collator.collate)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=collator.collate)
    
    prompt_lr = 5e-6
    base_lr = 1e-4
    
    prompt_params    = {"params": model.prompt_embedding,"lr": prompt_lr}
    other_params     = {
        "params": [
            p for n, p in model.named_parameters()
            if n != "prompt_embedding" and p.requires_grad
        ],
        "lr": base_lr
    }

    optimizer = AdamW([ prompt_params, other_params ])
    
    # optimizer = AdamW(model.parameters(), lr=1e-4)
    
    num_schedule_cycle = 6
    lr_schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epoch//num_schedule_cycle, eta_min=5e-6)

    trainer = Trainer(
        num_epoch=num_epoch,
        model=model,
        optimizer=optimizer,
        device=device,
        train_dataloader=train_dataloader,
        valid_dataloader=valid_dataloader,
        lr_scheduler=lr_schedule,
        patience=20,
        eps=1e-3,
        save_dir="./train_log"
    )
    
    trainer.train()
    
    test_dataset = EmbeddedDataset("./data/archive/new_test.csv")
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collator.collate)
    
    evaler = Evaler(model=model, dataloader=test_dataloader, device=device)
    
    evaler.evaluate()

if __name__ == "__main__":
    main()