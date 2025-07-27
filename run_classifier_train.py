import argparse
import sys
from torch.utils.data.dataloader import DataLoader
from torch.optim.adamw import AdamW
from model.clip_w2v_gpt import GPT2LoRAClassifier
from utils.setting import set_random_seed
import torch
import logging
from data_processing.metaphor_dataset import EmbeddedDataset, EmbeddedCollator
from train_eval.trainer import Trainer
from train_eval.evaler import Evaler
from utils.logger_handler import TqdmLoggingHandler


def parse_args():
    parser = argparse.ArgumentParser(description='训练脚本')
    parser.add_argument('--batch_size', type=int, default=64, help='批次大小')
    parser.add_argument('--num_epoch', type=int, default=200, help='训练轮数')
    parser.add_argument('--lr', type=float, default=5e-4, help='学习率')
    parser.add_argument('--patience', type=int, default=10, help='早停耐心值')
    parser.add_argument('--eps', type=float, default=1e-3, help='早停阈值')
    parser.add_argument('--num_schedule_cycle', type=int, default=5, help='学习率调度周期')
    parser.add_argument('--train_data', type=str, default="./data/archive/avg_train.csv", help='训练数据路径')
    parser.add_argument('--val_data', type=str, default="./data/archive/avg_val.csv", help='验证数据路径')
    parser.add_argument('--test_data', type=str, default="./data/archive/avg_test.csv", help='测试数据路径')
    parser.add_argument('--save_dir', type=str, default="./train_log", help='保存目录')
    return parser.parse_args()


def main():
    args = parse_args()
    set_random_seed(42)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_dataset = EmbeddedDataset(args.train_data)
    valid_dataset = EmbeddedDataset(args.val_data)
    test_dataset = EmbeddedDataset(args.test_data)
    collator = EmbeddedCollator(device)
    
    model = GPT2LoRAClassifier(dropout=0.3, img_feature_dim=768, lora_r=1, lora_alpha=4)
    
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collator.collate)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collator.collate)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collator.collate)
    
    # 为不同参数组设置不同学习率
    prompt_lr = args.lr * 0.1
    lora_lr = args.lr * 0.5
    base_lr = args.lr 
    
    # 分离不同类型的参数
    prompt_params = {"params": model.prompt_embedding, "lr": prompt_lr}
    
    lora_params = {
        "params": [
            p for n, p in model.named_parameters()
            if "lora" in n.lower() and p.requires_grad
        ],
        "lr": lora_lr
    }
    
    other_params = {
        "params": [
            p for n, p in model.named_parameters()
            if n != "prompt_embedding" and "lora" not in n.lower() and p.requires_grad
        ],
        "lr": base_lr
    }
    
    optimizer = AdamW([prompt_params, lora_params, other_params])
    lr_schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epoch//args.num_schedule_cycle, eta_min=5e-6)

    trainer = Trainer(
        num_epoch=args.num_epoch,
        model=model,
        optimizer=optimizer,
        device=device,
        train_dataloader=train_dataloader,
        valid_dataloader=valid_dataloader,
        lr_scheduler=lr_schedule,
        patience=args.patience,
        eps=args.eps,
        test_dataloader=test_dataloader,
        save_dir=args.save_dir,
        log_fn=logging.info
    )
    
    # 设置日志文件，保存在实验目录下
    exp_dir = trainer.exp_dir
    log_path = exp_dir / "train.log"
    logging.basicConfig(
        filename=log_path,
        filemode='w',
        format='%(asctime)s %(levelname)s: %(message)s',
        level=logging.INFO
    )
    
    console = TqdmLoggingHandler()
    console.setLevel(logging.INFO)
    logging.getLogger().addHandler(console)
    logging.info(f"parameters: {vars(args)}")
    
    trainer.train()

if __name__ == "__main__":
    main()