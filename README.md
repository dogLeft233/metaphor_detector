# metaphor_detector

## 运行

1. 由于文件大小的限制。 需要自行下载数据集`flickr30k`和`MET-Meme`数据集、`word2vec`。
2. `calculate_disambiguator_input.py`：用 CLIP 编码`flickr30k`的图像和文本，用`word2vec`编码文本，作为消歧器的输入。
3. `run_split.py`：划分编码后的`flickr30k`数据集
4. `run_disambiguator_train.py`：训练消歧器
5. `calculate_classifier_input.py`:用训练得到的消歧器和 CLIP 编码`MET-Meme`英文的图像和文本，获得分类器的输入。
6. `run_contrast_train.py`:训练一般对比编码器
7.  `run_classifier_mlp.py`:训练mlp分类头(暂时弃用)

## 阶段性成果

- 不需要`gpt-2`了，大大减少了模型的体量
- 目前最佳：f1:`91.08` acc:`91.50` 
- 这个成果是用对比学习器得到的`distance`属性以`0.3`为阈值得到的。`distance`属性小于`0.3`为非隐喻，否则为正样本。使用神经网络分类头得到的效果暂时没有超过这个阈值分类。
- 模型对长文本的效果仍然很差

## 方法

1. 使用`CLIP`编码原始的图像和文本，获得图像表示$E_i$,文本表示$E_t$
2. 使用图文对齐数据集`flickr30k`训练消歧器$disambiguator$。
   1. $disambiguator$输入输出：输入$E_i$则获得图像消歧编码$D_i$,输入$E_t$则获得文本消歧编码$D_t$
   2. 监督信息：`word2vec`+`TF-IDF`加权编码数据集`flickr30k`中的句子，记为$W_t$。`flickr30k`数据集中一个图片又5段文本描述,图片的表示为这5段文本编码的平均，记为$W_i$。
   3. 损失函数：使用余弦相似度确保消歧后的图像和文本向量接近，同时与`word2vec`+`TF-IDF`编码的监督向量接近，公式为$CosineLoss(D_i, D_t) + CosineLoss(D_i, W_i) + CosineLoss(D_t, W_t)$
3. 对比器:
   1. 对比器将学到一个向量的表示空间，在这个空间里，`CLIP`输出的向量$E_i, E_t$将被结合成上下文表示向量$C$，消歧器输出的$D_i, D_t$将被表示成孤立表示向量$S$。
   2. 对比器的目的是：对隐喻样本，$C$要与$S$互相远离，对非隐喻样本，$C$与$S$互相接近。
   3. 向量的距离仍用余弦距离衡量,记为$d$
   4. 损失函数:$label*max(m-d,0)^2 + (1-label)*d^2$,其中$label$为取值为0或1的标签，m为超参数。
   5. 为了让上下文向量$C$更能表示说话者意图，在`ContrastiveEncoder2`中引入了对说话者`情感`、`目的`、`话语冒犯程度`的分类，在训练中有一点效果，但没有更加严格的验证。
   6. 目前为取阈值`0.3`作分类