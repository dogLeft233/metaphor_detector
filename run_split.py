import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import argparse
from pathlib import Path
import os

def split_dataset(csv_path: str, train_ratio: float = 0.8, random_state: int = 42, 
                 output_dir: str = "./data/splits"):
    """
    读取CSV文件并划分训练集与验证集
    
    Args:
        csv_path (str): CSV文件路径
        train_ratio (float): 训练集比例，默认为0.8
        random_state (int): 随机种子，默认为42
        output_dir (str): 输出目录，默认为"./data/splits"
    """
    print(f"正在读取CSV文件: {csv_path}")
    
    # 读取CSV文件
    try:
        df = pd.read_csv(csv_path, encoding="utf-8")
        print(f"成功读取数据，共 {len(df)} 行")
        print(f"列名: {list(df.columns)}")
    except Exception as e:
        print(f"读取CSV文件失败: {e}")
        return
    
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 划分训练集和验证集
    train_df, val_df = train_test_split(
        df, 
        train_size=train_ratio, 
        random_state=random_state,
        shuffle=True
    )
    
    print(f"数据集划分完成:")
    print(f"  训练集: {len(train_df)} 行 ({len(train_df)/len(df)*100:.1f}%)")
    print(f"  验证集: {len(val_df)} 行 ({len(val_df)/len(df)*100:.1f}%)")
    
    # 生成输出文件名
    csv_name = Path(csv_path).stem
    train_path = output_path / f"{csv_name}_train.csv"
    val_path = output_path / f"{csv_name}_val.csv"
    
    # 保存划分后的数据集
    train_df.to_csv(train_path, index=False, encoding="utf-8")
    val_df.to_csv(val_path, index=False, encoding="utf-8")
    
    print(f"数据集已保存:")
    print(f"  训练集: {train_path}")
    print(f"  验证集: {val_path}")
    
    # 显示数据集统计信息
    print("\n数据集统计信息:")
    print(f"  原始数据形状: {df.shape}")
    print(f"  训练集形状: {train_df.shape}")
    print(f"  验证集形状: {val_df.shape}")
    
    # 如果有标签列，显示标签分布
    if 'label' in df.columns:
        print("\n标签分布:")
        print("原始数据:")
        print(df['label'].value_counts())
        print("\n训练集:")
        print(train_df['label'].value_counts())
        print("\n验证集:")
        print(val_df['label'].value_counts())
    
    return train_path, val_path

def split_with_stratification(csv_path: str, label_column: str = 'label', 
                            train_ratio: float = 0.8, random_state: int = 42,
                            output_dir: str = "./data/splits"):
    """
    使用分层抽样划分数据集（保持标签分布一致）
    
    Args:
        csv_path (str): CSV文件路径
        label_column (str): 标签列名，默认为'label'
        train_ratio (float): 训练集比例，默认为0.8
        random_state (int): 随机种子，默认为42
        output_dir (str): 输出目录，默认为"./data/splits"
    """
    print(f"正在读取CSV文件: {csv_path}")
    
    # 读取CSV文件
    try:
        df = pd.read_csv(csv_path, encoding="utf-8")
        print(f"成功读取数据，共 {len(df)} 行")
        print(f"列名: {list(df.columns)}")
    except Exception as e:
        print(f"读取CSV文件失败: {e}")
        return
    
    # 检查标签列是否存在
    if label_column not in df.columns:
        print(f"警告: 标签列 '{label_column}' 不存在，使用普通划分")
        return split_dataset(csv_path, train_ratio, random_state, output_dir)
    
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 使用分层抽样划分数据集
    train_df, val_df = train_test_split(
        df, 
        train_size=train_ratio, 
        random_state=random_state,
        shuffle=True,
        stratify=df[label_column]
    )
    
    print(f"分层抽样数据集划分完成:")
    print(f"  训练集: {len(train_df)} 行 ({len(train_df)/len(df)*100:.1f}%)")
    print(f"  验证集: {len(val_df)} 行 ({len(val_df)/len(df)*100:.1f}%)")
    
    # 生成输出文件名
    csv_name = Path(csv_path).stem
    train_path = output_path / f"{csv_name}_train_stratified.csv"
    val_path = output_path / f"{csv_name}_val_stratified.csv"
    
    # 保存划分后的数据集
    train_df.to_csv(train_path, index=False, encoding="utf-8")
    val_df.to_csv(val_path, index=False, encoding="utf-8")
    
    print(f"数据集已保存:")
    print(f"  训练集: {train_path}")
    print(f"  验证集: {val_path}")
    
    # 显示标签分布
    print("\n标签分布:")
    print("原始数据:")
    print(df[label_column].value_counts())
    print("\n训练集:")
    print(train_df[label_column].value_counts())
    print("\n验证集:")
    print(val_df[label_column].value_counts())
    
    return train_path, val_path

def main():
    parser = argparse.ArgumentParser(description="数据集划分工具")
    parser.add_argument("csv_path", help="CSV文件路径")
    parser.add_argument("--train-ratio", type=float, default=0.8, 
                       help="训练集比例 (默认: 0.8)")
    parser.add_argument("--random-state", type=int, default=42,
                       help="随机种子 (默认: 42)")
    parser.add_argument("--output-dir", default="./data/splits",
                       help="输出目录 (默认: ./data/splits)")
    parser.add_argument("--stratify", action="store_true",
                       help="使用分层抽样")
    parser.add_argument("--label-column", default="label",
                       help="标签列名 (默认: label)")
    
    args = parser.parse_args()
    
    # 检查输入文件是否存在
    if not os.path.exists(args.csv_path):
        print(f"错误: 文件 {args.csv_path} 不存在")
        return
    
    print("=" * 60)
    print("数据集划分工具")
    print("=" * 60)
    
    if args.stratify:
        print("使用分层抽样划分数据集...")
        split_with_stratification(
            args.csv_path, 
            args.label_column,
            args.train_ratio, 
            args.random_state, 
            args.output_dir
        )
    else:
        print("使用普通随机划分数据集...")
        split_dataset(
            args.csv_path, 
            args.train_ratio, 
            args.random_state, 
            args.output_dir
        )
    
    print("\n" + "=" * 60)
    print("数据集划分完成")
    print("=" * 60)

if __name__ == "__main__":
    # 如果没有命令行参数，使用默认参数运行
    import sys
    if len(sys.argv) == 1:
        # 默认处理flickr30k数据
        csv_path = "./data/flickr30k/processed_flickr30k_.csv"
        if os.path.exists(csv_path):
            print("使用默认参数处理flickr30k数据...")
            split_dataset(csv_path, 0.8, 42, "./data/splits")
        else:
            print("请提供CSV文件路径作为参数")
            print("用法示例: python run_split.py data.csv --train-ratio 0.8")
    else:
        main()
