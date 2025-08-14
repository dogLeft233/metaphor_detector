from torch.utils.data.dataloader import DataLoader
from torch.optim import AdamW
from data_processing.metaphor_dataset import MIPEmbeddingDataset, MIPEmbeddingCollator
from train_eval.trainer import Trainer
from model.metaphor_detector import ContrastiveEncoder, ContrastiveEncoder2
from utils.logger_handler import TqdmLoggingHandler
import torch
import logging
import argparse
from utils.setting import set_random_seed
from utils.plot_distance import plot_distance

def parse_args():
    parser = argparse.ArgumentParser(description='训练脚本')
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--intention_coefficient', type=float, default=0.06, help='意图检测loss系数')
    parser.add_argument('--sentiment_coefficient', type=float, default=0.08)
    parser.add_argument('--offensiveness_coefficient', type=float, default=0.12)
    parser.add_argument('--batch_size', type=int, default=128, help='批次大小')
    parser.add_argument('--num_epoch', type=int, default=200, help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--patience', type=int, default=6, help='早停耐心值')
    parser.add_argument('--eps', type=float, default=1e-4, help='早停阈值')
    parser.add_argument('--train_data', type=str, default="./data/archive/avg_train.csv", help='训练数据路径')
    parser.add_argument('--val_data', type=str, default="./data/archive/avg_val.csv", help='验证数据路径')
    parser.add_argument('--save_dir', type=str, default="./train_log", help='保存目录')
    return parser.parse_args()

def main():
    args = parse_args()
    set_random_seed(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_dataset = MIPEmbeddingDataset(args.train_data)
    valid_dataset = MIPEmbeddingDataset(args.val_data)
    collator = MIPEmbeddingCollator(device=device)
    
    model = ContrastiveEncoder2(
                                dropout=args.dropout,
                                intention_coefficient=args.intention_coefficient,
                                sentiment_coefficient=args.sentiment_coefficient,
                                offensiveness_coefficient=args.offensiveness_coefficient
                            )
    
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collator.collate)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collator.collate)
    
    optimizer = AdamW(model.parameters(), lr=args.lr)

    lr_schedule = None

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
        save_dir=args.save_dir
    )
    
    exp_dir = trainer.exp_dir
    log_path = exp_dir / "train.log"
    logging.basicConfig(
        filename=log_path,
        filemode='w',
        format='%(asctime)s %(levelname)s: %(message)s',
        level=logging.INFO
    )
    
    logging.info(f"parameters: {vars(args)}")
    console = TqdmLoggingHandler()
    console.setLevel(logging.INFO)
    logging.getLogger().addHandler(console)
    
    trainer.train()
    
    plot_distance(trainer.best_model, exp_dir , logging.info)

if __name__ == "__main__":
    main()