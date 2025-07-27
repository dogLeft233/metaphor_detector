# import pandas as pd
# from sklearn.model_selection import train_test_split

# # 读取两个文件
# fb_path = './data/MultiMET/embedded_Facebook_pic_solved.tsv'
# tw_path = './data/MultiMET/embedded_Twitter_pic_solved.tsv.tsv'
# df_fb = pd.read_csv(fb_path, sep='\t', encoding='utf-8')
# df_tw = pd.read_csv(tw_path, sep='\t', encoding='utf-8')

# # 合并
# df = pd.concat([df_fb, df_tw], ignore_index=True)

# # 丢弃Metaphor?属性缺失的行
# df = df.dropna(subset=['Metaphor?'])

# print(df.columns)

# # 要去掉的列
# cols_to_drop = [
#     'ID', ' Text  ','is text metaphor?','Category ', 'Target (目标域）', 'Target  Modality ', 'Source  （源域）', 'Source  Modality'
# ]
# df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

# # 划分训练集和验证集（8:2）
# train_df, dev_df = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)

# # 保存
# train_df.to_csv('./data/MultiMET/train.tsv', sep='\t', index=False, encoding='utf-8')
# dev_df.to_csv('./data/MultiMET/dev.tsv', sep='\t', index=False, encoding='utf-8')
# print('已保存train.tsv和dev.tsv')

import pandas as pd

# 读取processed_flickr30k.csv并分成训练集和验证集
from sklearn.model_selection import train_test_split

# 读取处理后的数据
processed_df = pd.read_csv("./data/flickr30k/processed_flickr30k.csv", encoding="utf-8")

# 分成训练集和验证集（8:2比例）
train_df, dev_df = train_test_split(processed_df, test_size=0.2, random_state=42, shuffle=True)

# 保存训练集和验证集
train_df.to_csv("./data/flickr30k/train.csv", index=False, encoding="utf-8")
dev_df.to_csv("./data/flickr30k/dev.csv", index=False, encoding="utf-8")

print(f"训练集大小: {len(train_df)}")
print(f"验证集大小: {len(dev_df)}")
print("已保存train.csv和dev.csv")
