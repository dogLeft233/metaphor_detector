# metaphor_detector

## 运行

1. 由于文件大小的限制。 需要自行下载数据集`flickr30k`和`MET-Meme`数据集、`word2vec`。
2. `calculate_disambiguator_input.py`：用 CLIP 编码`flickr30k`的图像和文本，用`word2vec`编码文本，作为消歧器的输入。
3. `run_disambiguator_train.py`：训练消歧器
4. `calculate_classifier_input.py`:用训练得到的消歧器和 CLIP 编码`MET-Meme`英文的图像和文本，获得分类器的输入。
5. `run_classifier_train.py`:训练分类器，获得最终模型
6. `flickr30k`需要自行划分训练集与测试集:(

## 目前最佳

- 测试集上： acc: `90.62` f1_pos: `80.92` f1_neg: `93.79` f1_avg: `90.17` (pos:225, neg:575)
- 在`Contrast`分支上达到 acc: `91.50` f1_avg: `91.08`