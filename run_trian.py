from torch.utils.data.dataloader import DataLoader
from torch.optim.adam import Adam
from model.TransformerClassifier import TransformerClassifier
from model.BertW2V import BertW2V
from data.VUA18.VUA18_Dataset import VUA18_Dataset, VUA18_Collator, VUA18_Collator_Embed
from train_eval.Trainer import Trainer
from transformers import get_scheduler
from transformers import BertTokenizer
import torch
from transformers import BertModel, BertTokenizerFast
from gensim.models import KeyedVectors
from utils.setting import set_random_seed
from model.BertClassifier import BertClassifier

def main():
    set_random_seed(42)
    
    w2v = KeyedVectors.load_word2vec_format("./word2vec/GoogleNews-vectors-negative300.bin", binary=True)
    
    batch_size = 16
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

    train_dataset = VUA18_Dataset("./data/VUA18/train.tsv",tokenizer=tokenizer)
    valid_dataset = VUA18_Dataset("./data/VUA18/dev.tsv",tokenizer=tokenizer)
    collator = VUA18_Collator_Embed(tokenizer, w2v, device=device)
    
    model = BertW2V()
    
    for p in model.bert.parameters():
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
        patience=100000000,
        eps=1e-2,
        save_dir="./train_log"
    )
    
    trainer.train()

if __name__ == "__main__":
    main()