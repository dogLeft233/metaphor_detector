# metaphor_detector

## 运行

1. 由于文件大小的限制。 需要自行下载数据集`flickr30k`和`MET-Meme`数据集、`word2vec`。
2. `calculate_disambiguator_input.py`：用 CLIP 编码`flickr30k`的图像和文本，用`word2vec`编码文本，作为消歧器的输入。
3. `run_split.py`：划分编码后的`flickr30k`数据集
4. `run_disambiguator_train.py`：训练消歧器
5. `calculate_classifier_input.py`:用训练得到的消歧器和 CLIP 编码`MET-Meme`英文的图像和文本，获得分类器的输入。
6. `run_classifier_train.py`:训练分类器，获得最终模型
7. `run_contrast_train.py`:训练一般对比编码器
8. `run_contrastivegpt_train.py`:训练gpt对比编码器
9. `plot_distance.py`:做出正负样本的对比分布图，获得正样本距离升序排序表格
10. `run_classifier_mlp.py`:训练mlp分类头

## 阶段性结果
- 模型效果没有明显提升
    ### 由正样本分布分析模型表现
  - 分类不佳的样本多含有专有的人名或地名
  - 模型对文本较长的样本分类不佳
  - 使用w2v消歧效果可能不足