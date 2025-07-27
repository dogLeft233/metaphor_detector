from torch.utils.data.dataloader import DataLoader
from torch.optim import AdamW
from data_processing.flickr_dataset import EmbeddingsDataset, EmbeddingsCollator
from train_eval.trainer import Trainer
from model.clip_w2v_gpt import W2VDisambiguator
import torch
from gensim.models import KeyedVectors
from utils.setting import set_random_seed

def main():
    set_random_seed(42)
    
    batch_size = 16

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_dataset = EmbeddingsDataset("./data/flickr30k/train.csv")
    valid_dataset = EmbeddingsDataset("./data/flickr30k/dev.csv")
    collator = EmbeddingsCollator(device=device)
    
    model = W2VDisambiguator(dropout=0.2)
    
    num_epoch = 200
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collator.collate)
    
    optimizer = AdamW(model.parameters(), lr=1e-4)
    
    # lr_schedule = get_scheduler(
    #     name="linear", optimizer=optimizer, num_warmup_steps=0,
    #     num_training_steps=num_epoch * len(train_dataloader)
    # )
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
        eps=1e-3,
        save_dir="./train_log"
    )
    
    trainer.train()

if __name__ == "__main__":
    main()