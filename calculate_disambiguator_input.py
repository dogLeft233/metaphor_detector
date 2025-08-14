import pandas as pd
from typing import List
import torch
import re
import string
import numpy as np
from gensim.models import KeyedVectors
from typing import List
from pathlib import Path
from tqdm import tqdm
from utils.clip_handler import CLIPHandler
from sklearn.feature_extraction.text import TfidfVectorizer

device = 'cuda' if torch.cuda.is_available() else 'cpu'

#设置flickr 30k路径
csv_path = "./data/flickr30k/flickr_annotations_30k.csv"
image_dir = Path("./data/flickr30k/flickr30k-images")
save_path = "./data/flickr30k/processed_flickr30k.csv"

w2v = KeyedVectors.load_word2vec_format("./word2vec/GoogleNews-vectors-negative300.bin", binary=True)

data = pd.read_csv(csv_path, encoding="utf-8")

punct_re = f'[{re.escape(string.punctuation)}]'
token_pattern = re.compile(r"(?u)\b\w+\b")

def fit_tfidf(corpus: List[str]) -> TfidfVectorizer:
    vec = TfidfVectorizer(
        lowercase=True,
        token_pattern=token_pattern.pattern,
        max_df=0.8,
        min_df=1
    )
    vec.fit(corpus)
    return vec

all_sentences = []
for idx, row in data.iterrows():
    sents = eval(row["raw"])
    sents = [re.sub(punct_re, '', s) for s in sents]
    all_sentences.extend(sents)

vectorizer = fit_tfidf(all_sentences)

def get_sentence_w2v(
    sentence: str,
    vectorizer: TfidfVectorizer,
    w2v_model: KeyedVectors,
    device: str = "cpu"
) -> torch.Tensor:
    words = vectorizer.build_analyzer()(sentence)

    tfidf_row = vectorizer.transform([sentence])

    weights = []
    for w in words:
        idx = vectorizer.vocabulary_.get(w)
        if idx is None:
            weights.append(0.0)
        else:
            weights.append(tfidf_row[0, idx])
    w_arr = np.array(weights, dtype=np.float32)

    if w_arr.sum() == 0:
        w_arr = np.ones_like(w_arr, dtype=np.float32)

    embeds = []
    for w in words:
        if w in w2v_model:
            vec = torch.from_numpy(w2v_model[w]).to(device)
        else:
            vec = torch.zeros(w2v_model.vector_size, device=device)
        embeds.append(vec)
    if not embeds:
        return torch.zeros(w2v_model.vector_size, device=device)

    embeds = torch.stack(embeds, dim=0)
    w_tensor = torch.from_numpy(w_arr).to(device)

    sent_vec = (embeds * w_tensor.unsqueeze(1)).sum(0) / w_tensor.sum()
    return sent_vec

@torch.no_grad()
def process(sentences: List[str], file_name: str, vectorizer=None):
    assert len(sentences) == 5
    sentences = [re.sub(punct_re, '', s) for s in sentences]
    text_embeds = torch.stack([CLIPHandler.process_text(sentence, device=device) for sentence in sentences], dim=0)
    image_path = image_dir / file_name
    image_embeds = CLIPHandler.process_image(image_path, device=device)
    image_embeds = image_embeds.view(-1).cpu().tolist()
    text_embeds = text_embeds.view(-1).cpu().tolist()
    w2v_embeds = torch.stack([
        get_sentence_w2v(s, vectorizer=vectorizer, w2v_model=w2v) for s in sentences
    ], dim=0).view(-1).cpu().tolist()
    return pd.Series({
        "image_embeds": image_embeds,
        "text_embeds": text_embeds,
        "labels": w2v_embeds
    })

processed_data = []
for idx, row in tqdm(data.iterrows(), total=len(data), desc="Processing data"):
    sentences = eval(row["raw"])
    file_name = row["filename"]
    result = process(sentences, file_name, vectorizer=vectorizer)
    processed_data.append(result)

result_df = pd.DataFrame(processed_data)
result_df.to_csv(save_path, encoding="utf-8")