# File: data_builder.py
import json
import numpy as np
import random
from typing import List, Tuple, Dict
from sklearn.model_selection import KFold
from tqdm import tqdm
from utils import flatten_embedding


def load_sequence_dict(path: str) -> Dict[str, str]:
    """Load a JSON file or plain dict mapping index->sequence or sequence->... expected format:
    {"1":"SEQ1", "2":"SEQ2", ...}
    Returns a dict index_str->seq_str
    """
    with open(path, 'r', encoding='utf8') as f:
        d = json.load(f)
    return d


def build_samples_from_seqdict(seqdict: Dict[str, str], positive_ids: List[str]) -> Tuple[List[str], List[int]]:
    """Given seqdict mapping id->sequence, and list of positive ids (as keys of seqdict),
    build lists of sequences and binary labels aligned.
    """
    seqs = []
    labels = []
    for k, seq in seqdict.items():
        seqs.append(seq)
        labels.append(1 if k in set(positive_ids) else 0)
    return seqs, labels


# def similarity_score(sample1: str, sample2: str) -> float:
#     """计算两个样本之间的相似性分数。此处使用简单的字符串匹配方式，
#     实际应用中可以使用更复杂的特征或深度学习模型进行相似度计算。"""
#     # 假设 sample1 和 sample2 是氨基酸序列
#     # 这里只是示意代码，可以使用更复杂的相似性度量方法
#     return sum(1 for a, b in zip(sample1, sample2) if a == b) / max(len(sample1), len(sample2))

def similarity_score(sample1: str, sample2: str) -> float:
    """计算两个样本之间的相似性分数，保留三位小数。"""
    if not sample1 or not sample2:
        return 0.0
    score = sum(1 for a, b in zip(sample1, sample2) if a == b) / max(len(sample1), len(sample2))
    return round(score, 3)


def negative_sampling_strategy(positives: List[str], negatives: List[str], mode: str, seed: int=42):
    """mode: 'a' (all negatives with positives), 'b' (split negatives into 6 non-overlap equal groups -> 6 datasets),
    'c' (one random equal-size negative sample)
    Returns list of datasets, each dataset is (pos_list, neg_list)
    """
    rnd = random.Random(seed)
    datasets = []
    if mode == 'a':
        datasets.append((positives, negatives))
    elif mode == 'b':
        n = len(positives)
        tot_neg = len(negatives)
        # shuffle negatives deterministically
        negs = negatives.copy()
        rnd.shuffle(negs)
        # split into 6 chunks of size n (if available), non-overlap; if insufficient for full cover, last chunk may be smaller
        chunks = []
        idx = 0
        while idx < tot_neg and len(chunks) < 6:
            chunk = negs[idx: idx + n]
            chunks.append(chunk)
            idx += n
        # if not enough to make 6, remaining negatives discarded as per requirement
        for chunk in chunks:
            datasets.append((positives, chunk))
    elif mode == 'c':
        n = len(positives)
        negs = negatives.copy()
        rnd.shuffle(negs)
        datasets.append((positives, negs[:n//3]))  # negs[:n//3])
    elif mode == 'd':
        """新模式：根据相似性筛选负样本"""
        n = len(positives)
        valid_negatives = []

        # 遍历负样本，计算与每个正样本的相似度
        for neg in tqdm(negatives, desc="Filtering negatives"):
            is_valid = True
            for pos in positives:
                # 计算正负样本之间的相似度
                if similarity_score(pos, neg) > 0.20: # 0.26  8695 ; 0.25 1140
                    is_valid = False
                    break
            if is_valid:
                valid_negatives.append(neg)
        print(len(valid_negatives))
        # 从筛选后的负样本中随机选择与正样本数量相等的负样本
        rnd.shuffle(valid_negatives)
        datasets.append((positives, valid_negatives[:n//2]))
    else:
        raise ValueError('mode must be a/b/c/d')
    return datasets


def build_feature_label_list(pos_seqs: List[str], neg_seqs: List[str], emb_all_store, emb_left_store, emb_right_store) -> Tuple[np.ndarray, np.ndarray]:
    """For given positive and negative sequence lists (sequence strings), fetch embeddings and build
    X (num_samples x feature_dim) and y (num_samples)
    Feature format: concat(emb_all, emb_left, emb_right)
    """
    samples = []
    labels = []
    missing = 0
    print(emb_all_store)

    for seq in pos_seqs:
        mid = len(seq)//2
        ea = emb_all_store.get(seq)
        el = emb_left_store.get(seq[0: mid])
        er = emb_right_store.get(seq[mid+1: len(seq)])
        if ea is None or el is None or er is None:
            missing += 1
            continue
        feat = np.concatenate([flatten_embedding(ea), flatten_embedding(el), flatten_embedding(er)])
        samples.append(feat)
        labels.append(1)
    for seq in neg_seqs:
        mid = len(seq) // 2
        ea = emb_all_store.get(seq)
        el = emb_left_store.get(seq[0:mid])
        er = emb_right_store.get(seq[mid+1: len(seq)])
        if ea is None or el is None or er is None:
            missing += 1
            continue
        feat = np.concatenate([flatten_embedding(ea), flatten_embedding(el), flatten_embedding(er)])
        samples.append(feat)
        labels.append(0)
    if missing > 0:
        print(f"Warning: {missing} samples skipped due to missing embeddings")
    X = np.vstack(samples)
    y = np.array(labels, dtype=int)
    return X, y


def make_outer_folds(X: np.ndarray, y: np.ndarray, n_splits: int=5, seed: int=42):
    """Create outer folds: returns list of (trainval_idx, test_idx)
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    out = []
    for train_idx, test_idx in kf.split(X):
        out.append((train_idx, test_idx))
    return out


def make_inner_folds(trainval_idx: np.ndarray, n_splits: int=10, seed: int=42):
    """Given indices for train+val, produce inner k-folds for CV (i.e., split into 10 folds with ratio ~9:1)
    returns list of (train_idx_within, val_idx_within) indices relative to the trainval_idx array
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    out = []
    for tr, va in kf.split(trainval_idx):
        out.append((trainval_idx[tr], trainval_idx[va]))
    return out