from torch.utils.data.dataloader import DataLoader
from torch.optim.adam import Adam
from torch.optim.adamw import AdamW
from model.clip_w2v_gpt import GPTClassifier, Meteor
from data_processing.metaphor_dataset import EmbeddedDataset, EmbeddedCollator
from train_eval.evaler import Evaler
from utils.setting import set_random_seed
import torch

def main():
    set_random_seed(42)
    
    batch_size = 16
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    test_dataset = EmbeddedDataset("./data/archive/new_test.csv")
    collator = EmbeddedCollator(device)
    
    model = Meteor(img_feature_dim=512)
    
    model.load_state_dict(torch.load("./train_log/exp_2025-07-25_21-31-03/model.pth"))
    
    # model.load_state_dict(torch.load("./train_log/exp_2025-07-25_16-46-37/model.pth"))
    
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collator.collate)
    
    evaler = Evaler(model=model, dataloader=test_dataloader, device=device)
    
    evaler.evaluate()

if __name__ == "__main__":
    main()