import argparse
from torch.utils.data.dataloader import DataLoader
from torch.optim.adamw import AdamW
from model.metaphor_detector import MLPclassifier, MLPclassifier2
from utils.setting import set_random_seed
import torch
import logging
from data_processing.metaphor_dataset import MIPEmbeddingDataset, MIPEmbeddingCollator
from train_eval.trainer import TrainerBinary
from utils.logger_handler import TqdmLoggingHandler


def parse_args():
    parser = argparse.ArgumentParser(description='训练脚本')
    parser.add_argument('--batch_size', type=int, default=128, help='批次大小')
    parser.add_argument('--num_epoch', type=int, default=200, help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--encoder_lr', type=float, default=5e-6, help='encoder学习率')
    parser.add_argument('--patience', type=int, default=6, help='早停耐心值')
    parser.add_argument('--eps', type=float, default=1e-3, help='早停阈值')
    parser.add_argument('--num_schedule_cycle', type=int, default=6, help='学习率调度周期')
    parser.add_argument('--train_data', type=str, default="./data/archive/avg_train.csv", help='训练数据路径')
    parser.add_argument('--val_data', type=str, default="./data/archive/avg_val.csv", help='验证数据路径')
    parser.add_argument('--test_data', type=str, default="./data/archive/avg_test.csv", help='测试数据路径')
    parser.add_argument('--save_dir', type=str, default="./train_log", help='保存目录')
    return parser.parse_args()


def main():
    args = parse_args()
    set_random_seed(42)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_dataset = MIPEmbeddingDataset(args.train_data)
    valid_dataset = MIPEmbeddingDataset(args.val_data)
    test_dataset = MIPEmbeddingDataset(args.test_data)
    collator = MIPEmbeddingCollator(device)
    
    model = MLPclassifier(dropout=0.3)
    
    model.encoder.load_state_dict(torch.load("./checkpoint/exp_2025-08-14_14-50-19/model.pth"))
    
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collator.collate)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collator.collate)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collator.collate)
    
    # # 将参数分为两个参数组，设置不同的学习率
    # encoder_params = list(model.encoder.parameters())
    # other_params = []
    
    # # 获取所有参数，排除encoder参数
    # encoder_param_set = set(encoder_params)
    # for param in model.parameters():
    #     if param not in encoder_param_set:
    #         other_params.append(param)
    
    for p in model.encoder.parameters():
        p.requires_grad = False
    
    # optimizer = AdamW([
    #     {'params': encoder_params, 'lr': args.encoder_lr, 'name': 'encoder'},
    #     {'params': other_params, 'lr': args.lr, 'name': 'other'}
    # ])
    
    optimizer = AdamW(model.parameters(), lr= args.lr)
    
    lr_schedule = None

    trainer = TrainerBinary(
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
    logging.info(f"encoder学习率: {args.encoder_lr}")
    logging.info(f"其他参数学习率: {args.lr}")
    
    trainer.train()

if __name__ == "__main__":
    main()