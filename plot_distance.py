from model.clip_w2v_gpt import ContrastiveEmbedding, Contrastivegpt
from data_processing.metaphor_dataset import EmbeddedDataset, EmbeddedCollator
import torch
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from collections import defaultdict
import pandas as pd

device = 'cuda' if torch.cuda.is_available() else 'cpu'

contrastiveEncoder = Contrastivegpt(lora_r=2, lora_alpha=4).to(device)
contrastiveEncoder.load_state_dict(torch.load("./checkpoint/exp_2025-08-02_15-32-20/model.pth"))
contrastiveEncoder.eval()

dataset = EmbeddedDataset("./data/archive/avg_test.csv")
collator = EmbeddedCollator(device=device)
dataloader = DataLoader(dataset, batch_size=1,  shuffle=False, collate_fn=collator.collate)

# 收集distance数据
positive_distances = []
negative_distances = []
all_distances = []
all_labels = []
# 收集正例样本信息用于保存表格
positive_samples_info = []

print("正在计算distance分布...")
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

print(f"数据统计:")
print(f"  总样本数: {len(all_distances)}")
print(f"  正例数量: {len(positive_distances)} ({len(positive_distances)/len(all_distances)*100:.1f}%)")
print(f"  负例数量: {len(negative_distances)} ({len(negative_distances)/len(all_distances)*100:.1f}%)")
print(f"  正例distance范围: [{positive_distances.min():.4f}, {positive_distances.max():.4f}]")
print(f"  负例distance范围: [{negative_distances.min():.4f}, {negative_distances.max():.4f}]")
print(f"  正例distance均值: {positive_distances.mean():.4f}")
print(f"  负例distance均值: {negative_distances.mean():.4f}")

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
plt.savefig('./distance_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

# 打印详细统计信息
print("\n详细统计信息:")
print("=" * 50)
print(f"正例(隐喻)统计:")
print(f"  数量: {len(positive_distances)}")
print(f"  均值: {positive_distances.mean():.4f}")
print(f"  标准差: {positive_distances.std():.4f}")
print(f"  中位数: {np.median(positive_distances):.4f}")
print(f"  最小值: {positive_distances.min():.4f}")
print(f"  最大值: {positive_distances.max():.4f}")
print(f"  25%分位数: {np.percentile(positive_distances, 25):.4f}")
print(f"  75%分位数: {np.percentile(positive_distances, 75):.4f}")

print(f"\n负例(非隐喻)统计:")
print(f"  数量: {len(negative_distances)}")
print(f"  均值: {negative_distances.mean():.4f}")
print(f"  标准差: {negative_distances.std():.4f}")
print(f"  中位数: {np.median(negative_distances):.4f}")
print(f"  最小值: {negative_distances.min():.4f}")
print(f"  最大值: {negative_distances.max():.4f}")
print(f"  25%分位数: {np.percentile(negative_distances, 25):.4f}")
print(f"  75%分位数: {np.percentile(negative_distances, 75):.4f}")

# 计算分离度指标
mean_diff = positive_distances.mean() - negative_distances.mean()
print(f"\n分离度分析:")
print(f"  均值差: {mean_diff:.4f}")
print(f"  正例均值: {positive_distances.mean():.4f}")
print(f"  负例均值: {negative_distances.mean():.4f}")

# 计算重叠度
overlap_threshold = (positive_distances.mean() + negative_distances.mean()) / 2
positive_below_threshold = np.sum(positive_distances < overlap_threshold)
negative_above_threshold = np.sum(negative_distances > overlap_threshold)
overlap_rate = (positive_below_threshold + negative_above_threshold) / len(all_distances)
print(f"  重叠度: {overlap_rate:.4f} ({overlap_rate*100:.1f}%)")

print(f"\n图形已保存为: ./distance_distribution.png")

low_distance_positive = [sample for sample in positive_samples_info if sample['distance']]

if low_distance_positive:
    # 按distance升序排列
    low_distance_positive.sort(key=lambda x: x['distance'])
    
    # 创建DataFrame
    df_low_distance = pd.DataFrame(low_distance_positive)
    
    # 保存到CSV文件
    output_file = './low_distance_positive_samples.csv'
    df_low_distance.to_csv(output_file, index=False, encoding='utf-8')
    
    print(f"已保存到: {output_file}")
    print(f"Distance范围: [{df_low_distance['distance'].min():.4f}, {df_low_distance['distance'].max():.4f}]")
    
    # 显示前10个样本
    print("\n前10个样本:")
    print(df_low_distance[['file_name', 'distance']].head(10).to_string(index=False))
    