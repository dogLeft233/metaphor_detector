from model.metaphor_detector import ContrastiveEncoder, ContrastiveEncoder2
from data_processing.metaphor_dataset import MIPEmbeddingDataset, MIPEmbeddingCollator
import torch
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, classification_report

def plot_distance(contrastiveEncoder, exp_dir:Path, log_fn = print):
    
    log_fn("\n" + "="*50)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    contrastiveEncoder.to(device)
    contrastiveEncoder.eval()

    dataset = MIPEmbeddingDataset("./data/archive/avg_test.csv")
    collator = MIPEmbeddingCollator(device=device)
    dataloader = DataLoader(dataset, batch_size=1,  shuffle=False, collate_fn=collator.collate)

    # 收集distance数据
    positive_distances = []
    negative_distances = []
    all_distances = []
    all_labels = []
    # 收集正例样本信息用于保存表格
    positive_samples_info = []

    log_fn("正在计算distance分布...")
    for i, batch in enumerate(tqdm(dataloader, ncols=80)):
        output = contrastiveEncoder.encode(**batch)
        label = batch["labels"].item()
        distance = output["distance"].item()
        
        all_distances.append(distance)
        all_labels.append(label)
        
        if label == 1:  # 正例（隐喻）
            positive_distances.append(distance)
            # 获取原始数据中的文件名等信息
            original_data = dataset.data.iloc[i]
            positive_samples_info.append({
                'file_name': original_data.get('file_name', f'sample_{i}'),
                'distance': distance,
                'text': original_data.get('text', ''),
                'metaphor_occurrence': original_data.get('metaphor occurrence', 1)
            })
        else:  # 负例（非隐喻）
            negative_distances.append(distance)

    # 转换为numpy数组
    positive_distances = np.array(positive_distances)
    negative_distances = np.array(negative_distances)
    all_distances = np.array(all_distances)
    all_labels = np.array(all_labels)

    log_fn(f"数据统计:")
    log_fn(f"  总样本数: {len(all_distances)}")
    log_fn(f"  正例数量: {len(positive_distances)} ({len(positive_distances)/len(all_distances)*100:.1f}%)")
    log_fn(f"  负例数量: {len(negative_distances)} ({len(negative_distances)/len(all_distances)*100:.1f}%)")
    log_fn(f"  正例distance范围: [{positive_distances.min():.4f}, {positive_distances.max():.4f}]")
    log_fn(f"  负例distance范围: [{negative_distances.min():.4f}, {negative_distances.max():.4f}]")
    log_fn(f"  正例distance均值: {positive_distances.mean():.4f}")
    log_fn(f"  负例distance均值: {negative_distances.mean():.4f}")

    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    # 创建图形
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Distance分布分析', fontsize=16, fontweight='bold')

    # 1. 直方图 - 正例和负例对比
    axes[0, 0].hist(positive_distances, bins=30, alpha=0.7, label='正例(隐喻)', color='red', density=True)
    axes[0, 0].hist(negative_distances, bins=30, alpha=0.7, label='负例(非隐喻)', color='blue', density=True)
    axes[0, 0].set_xlabel('Distance')
    axes[0, 0].set_ylabel('密度')
    axes[0, 0].set_title('正例和负例Distance分布对比')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. 箱线图
    data_to_plot = [positive_distances, negative_distances]
    axes[0, 1].boxplot(data_to_plot, labels=['正例(隐喻)', '负例(非隐喻)'])
    axes[0, 1].set_ylabel('Distance')
    axes[0, 1].set_title('Distance箱线图')
    axes[0, 1].grid(True, alpha=0.3)

    # 3. 核密度估计图
    sns.kdeplot(data=positive_distances, ax=axes[1, 0], label='正例(隐喻)', color='red')
    sns.kdeplot(data=negative_distances, ax=axes[1, 0], label='负例(非隐喻)', color='blue')
    axes[1, 0].set_xlabel('Distance')
    axes[1, 0].set_ylabel('密度')
    axes[1, 0].set_title('Distance核密度估计')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 4. 累积分布图
    sorted_positive = np.sort(positive_distances)
    sorted_negative = np.sort(negative_distances)
    y_positive = np.arange(1, len(sorted_positive) + 1) / len(sorted_positive)
    y_negative = np.arange(1, len(sorted_negative) + 1) / len(sorted_negative)

    axes[1, 1].plot(sorted_positive, y_positive, label='正例(隐喻)', color='red', linewidth=2)
    axes[1, 1].plot(sorted_negative, y_negative, label='负例(非隐喻)', color='blue', linewidth=2)
    axes[1, 1].set_xlabel('Distance')
    axes[1, 1].set_ylabel('累积概率')
    axes[1, 1].set_title('Distance累积分布函数')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    image_save_path = exp_dir / 'distance_distribution.png'
    plt.savefig(image_save_path, dpi=300, bbox_inches='tight')
    # plt.show()

    # 打印详细统计信息
    log_fn("\n详细统计信息:")
    log_fn("=" * 50)
    log_fn(f"正例(隐喻)统计:")
    log_fn(f"  数量: {len(positive_distances)}")
    log_fn(f"  均值: {positive_distances.mean():.4f}")
    log_fn(f"  标准差: {positive_distances.std():.4f}")
    log_fn(f"  中位数: {np.median(positive_distances):.4f}")
    log_fn(f"  最小值: {positive_distances.min():.4f}")
    log_fn(f"  最大值: {positive_distances.max():.4f}")
    log_fn(f"  25%分位数: {np.percentile(positive_distances, 25):.4f}")
    log_fn(f"  75%分位数: {np.percentile(positive_distances, 75):.4f}")

    log_fn(f"\n负例(非隐喻)统计:")
    log_fn(f"  数量: {len(negative_distances)}")
    log_fn(f"  均值: {negative_distances.mean():.4f}")
    log_fn(f"  标准差: {negative_distances.std():.4f}")
    log_fn(f"  中位数: {np.median(negative_distances):.4f}")
    log_fn(f"  最小值: {negative_distances.min():.4f}")
    log_fn(f"  最大值: {negative_distances.max():.4f}")
    log_fn(f"  25%分位数: {np.percentile(negative_distances, 25):.4f}")
    log_fn(f"  75%分位数: {np.percentile(negative_distances, 75):.4f}")

    # 计算分离度指标
    mean_diff = positive_distances.mean() - negative_distances.mean()
    log_fn(f"\n分离度分析:")
    log_fn(f"  均值差: {mean_diff:.4f}")
    log_fn(f"  正例均值: {positive_distances.mean():.4f}")
    log_fn(f"  负例均值: {negative_distances.mean():.4f}")

    # 计算重叠度
    overlap_threshold = (positive_distances.mean() + negative_distances.mean()) / 2
    positive_below_threshold = np.sum(positive_distances < overlap_threshold)
    negative_above_threshold = np.sum(negative_distances > overlap_threshold)
    overlap_rate = (positive_below_threshold + negative_above_threshold) / len(all_distances)
    log_fn(f"  重叠度: {overlap_rate:.4f} ({overlap_rate*100:.1f}%)")

    log_fn(f"\n图形已保存为: {image_save_path}")

    low_distance_positive = [sample for sample in positive_samples_info]

    if low_distance_positive:
        # 按distance升序排列
        low_distance_positive.sort(key=lambda x: x['distance'])
        
        # 创建DataFrame
        df_low_distance = pd.DataFrame(low_distance_positive)
        
        # 保存到CSV文件
        output_file = exp_dir / 'positive_samples.csv'
        df_low_distance.to_csv(output_file, index=False, encoding='utf-8')
        
        log_fn(f"已保存到: {output_file}")
        log_fn(f"Distance范围: [{df_low_distance['distance'].min():.4f}, {df_low_distance['distance'].max():.4f}]")
        
        # 显示前10个样本
        log_fn("\n前10个样本:")
        log_fn(df_low_distance[['file_name', 'distance']].head(10).to_string(index=False))

    # 基于distance阈值的分类评估
    log_fn("\n" + "="*50)
    log_fn("基于Distance阈值的分类评估")
    log_fn("="*50)

    # 设置阈值
    threshold = 0.5
    log_fn(f"分类阈值: {threshold}")
    log_fn(f"Distance >= {threshold}: 预测为正类")
    log_fn(f"Distance < {threshold}: 预测为负类")

    # 生成预测标签
    predictions = (all_distances >= threshold).astype(int)
    true_labels = all_labels

    # 计算准确率
    accuracy = accuracy_score(true_labels, predictions)
    log_fn(f"\n准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")

    # 计算加权F1分数
    weighted_f1 = f1_score(true_labels, predictions, average='weighted')
    log_fn(f"加权F1分数: {weighted_f1:.4f}")

    # 计算各类别的F1分数
    f1_positive = f1_score(true_labels, predictions, pos_label=1)
    f1_negative = f1_score(true_labels, predictions, pos_label=0)
    log_fn(f"正类F1分数: {f1_positive:.4f}")
    log_fn(f"负类F1分数: {f1_negative:.4f}")

    # 计算混淆矩阵
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(true_labels, predictions)
    log_fn(f"\n混淆矩阵:")
    log_fn(f"预测\真实    负类(0)  正类(1)")
    log_fn(f"负类(0)      {cm[0,0]:6d}   {cm[0,1]:6d}")
    log_fn(f"正类(1)      {cm[1,0]:6d}   {cm[1,1]:6d}")

    # 计算精确率和召回率
    from sklearn.metrics import precision_score, recall_score
    precision = precision_score(true_labels, predictions, average='weighted')
    recall = recall_score(true_labels, predictions, average='weighted')
    log_fn(f"\n加权精确率: {precision:.4f}")
    log_fn(f"加权召回率: {recall:.4f}")

    # 详细分类报告
    log_fn(f"\n详细分类报告:")
    log_fn(classification_report(true_labels, predictions, target_names=['负类(非隐喻)', '正类(隐喻)']))

    # 分析不同阈值下的性能
    log_fn(f"\n不同阈值下的性能分析:")
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
    log_fn(f"{'阈值':<8} {'准确率':<10} {'加权F1':<10} {'正类F1':<10} {'负类F1':<10}")
    log_fn("-" * 50)
    best_thresh = -1
    best_f1 = 0.0
    best_acc = 0.0
    for thresh in thresholds:
        preds = (all_distances >= thresh).astype(int)
        acc = accuracy_score(true_labels, preds)
        w_f1 = f1_score(true_labels, preds, average='weighted')
        pos_f1 = f1_score(true_labels, preds, pos_label=1)
        neg_f1 = f1_score(true_labels, preds, pos_label=0)
        log_fn(f"{thresh:<8.1f} {acc:<10.4f} {w_f1:<10.4f} {pos_f1:<10.4f} {neg_f1:<10.4f}")
        
        if w_f1 > best_f1:
            best_thresh = thresh
            best_f1 = w_f1
            best_acc = acc
    
    log_fn("-" * 50)   
    log_fn(f"最佳阈值:{best_thresh:<8.1f} 最佳f1:{best_f1:<10.4f} 准确率:{best_acc:<10.4f}")