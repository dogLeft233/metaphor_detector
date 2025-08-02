from torch.utils.data.dataloader import DataLoader
from torch.optim import AdamW
from data_processing.metaphor_dataset import EmbeddedDataset, EmbeddedCollator
from train_eval.trainer import Trainer
from model.clip_w2v_gpt import ContrastiveEmbedding
import torch
from utils.setting import set_random_seed

def main():
    set_random_seed(42)
    
    batch_size = 128

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_dataset = EmbeddedDataset("./data/archive/avg_train.csv")
    valid_dataset = EmbeddedDataset("./data/archive/avg_val.csv")
    collator = EmbeddedCollator(device=device)
    
    model = ContrastiveEmbedding(dropout=0.3)
    
    num_epoch = 200
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collator.collate)
    
    optimizer = AdamW(model.parameters(), lr=1e-4)

    lr_schedule = None

    trainer = Trainer(
        num_epoch=num_epoch,
        model=model,
        optimizer=optimizer,
        device=device,
        train_dataloader=train_dataloader,
        valid_dataloader=DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=collator.collate),
        lr_scheduler=lr_schedule,
        patience=10,
        eps=1e-4,
        save_dir="./train_log"
    )
    
    trainer.train()

if __name__ == "__main__":
    main()