from torch.utils.data.dataloader import DataLoader
from torch.optim.adam import Adam
from model.BertW2V import BertW2V
from data_processing.flickr.flickr import FlickrDataset, FlickrCollator
from train_eval.Trainer import Trainer
from transformers import CLIPProcessor, CLIPModel
from model.CLIPW2V import CLIPEncoder
import torch
from gensim.models import KeyedVectors
from utils.setting import set_random_seed

def main():
    set_random_seed(42)
    
    w2v = KeyedVectors.load_word2vec_format("./word2vec/GoogleNews-vectors-negative300.bin", binary=True)
    
    batch_size = 16
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    train_dataset = FlickrDataset("./data/flickr8k/train.csv", images_dir="./data/flickr8k/Images")
    valid_dataset = FlickrDataset("./data/flickr8k/val.csv", images_dir="./data/flickr8k/Images")
    collator = FlickrCollator(processor=processor, w2v=w2v, device=device)
    
    model = CLIPEncoder()
    
    for p in model.clip.parameters():
        p.requires_grad = False
    
    num_epoch = 20
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collator.collate)
    
    optimizer = Adam(model.parameters(), lr=1e-4)
    
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
        patience=3,
        eps=1e-2,
        save_dir="./train_log"
    )
    
    trainer.train()

if __name__ == "__main__":
    main()