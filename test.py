# File: main.py
import argparse
import os
import numpy as np
from data_builder import (
    load_sequence_dict,
    negative_sampling_strategy,
    build_feature_label_list,
    make_outer_folds,
    make_inner_folds
)
from embeddings_loader import load_npz_embeddings, EmbeddingStore
from autogluon_runner import train_and_evaluate_ag
from sklearn.utils import shuffle

def run_experiment(args):
    import json

    # --- load positives (for training + validation) ---
    with open(args.all_pos, "r", encoding="utf8") as f:
        pos_dict = json.load(f)
    positives = list(pos_dict.values())

    # --- load negatives (for training + validation) ---
    with open(args.all_neg, "r", encoding="utf8") as f:
        neg_dict = json.load(f)
    negatives = list(neg_dict.values())

    print(f"Loaded {len(positives)} positives, {len(negatives)} negatives for training/validation")

    # --- load independent test set ---
    with open(args.test_pos, "r", encoding="utf8") as f:
        test_pos_dict = json.load(f)
    test_positives = list(test_pos_dict.values())

    with open(args.test_neg, "r", encoding="utf8") as f:
        test_neg_dict = json.load(f)
    test_negatives = list(test_neg_dict.values())

    print(f"Loaded {len(test_positives)} positives, {len(test_negatives)} negatives for independent testing")

    # --- load embeddings ---
    emb_all_map = load_npz_embeddings(args.emb_all)
    emb_left_map = load_npz_embeddings(args.emb_left)
    emb_right_map = load_npz_embeddings(args.emb_right)
    emb_all = EmbeddingStore(emb_all_map)
    emb_left = EmbeddingStore(emb_left_map)
    emb_right = EmbeddingStore(emb_right_map)

    # --- negative sampling (train/val only) ---
    datasets = negative_sampling_strategy(positives, negatives, args.mode, seed=args.seed)

    # --- build independent test set features ---
    X_test, y_test = build_feature_label_list(test_positives, test_negatives, emb_all, emb_left, emb_right)
    X_test, y_test = shuffle(X_test, y_test, random_state=42)

    all_results = []
    ds_idx = 0

    pos_list, neg_list = datasets[0]

    print(f"Total samples: pos={len(pos_list)}, neg={len(neg_list)}")

    # 构建全数据集特征与标签
    X, y = build_feature_label_list(pos_list, neg_list, emb_all, emb_left, emb_right)

    # ====== 使用全部数据进行 10 折交叉验证 ======
    inner = make_inner_folds(np.arange(len(y)), n_splits=10, seed=args.seed)

    fold_idx = 0
    for train_idx, val_idx in inner:
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        outdir = os.path.join(args.outdir, f'fold{fold_idx}')
        metrics, predictor = train_and_evaluate_ag(
            X_train, y_train, X_val, y_val, X_test, y_test,
            outdir, time_limit=args.time_limit, seed=args.seed
        )
        all_results.append({
            'fold': fold_idx,
            'metrics': metrics
        })
        fold_idx += 1

    # --- save summary ---
    os.makedirs(args.outdir, exist_ok=True)
    with open(os.path.join(args.outdir, 'summary_results.json'), 'w', encoding='utf8') as f:
        json.dump(all_results, f, indent=2)
    print('✅ All experiments done.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # ===============================
    # 训练 + 验证数据（五折交叉验证）
    # ===============================
    parser.add_argument('--all_pos', type=str, default='data_2/all_pos_2.txt',
                        help='positive samples txt/json for training and validation (5-fold CV)')
    parser.add_argument('--all_neg', type=str, default='data_2/all_neg_2.txt',
                        help='negative samples txt/json for training and validation (5-fold CV)')

    # ===============================
    # 独立测试集（independent test set）
    # ===============================
    parser.add_argument('--test_pos', type=str, default='data_2/test_pos_2.txt',
                        help='independent positive samples txt/json for testing')
    parser.add_argument('--test_neg', type=str, default='data_2/test_neg_2.txt',
                        help='independent negative samples txt/json for testing')

    # ===============================
    # 嵌入文件（保持不变）
    # ===============================
    parser.add_argument('--emb_all', type=str, default='data_2/prott5_pooled_fea/merged_full.npz',
                        help='NPZ file path for all-sequence embeddings')
    parser.add_argument('--emb_left', type=str, default='data_2/prott5_pooled_fea/merged_left.npz',
                        help='NPZ file path for left subseq embeddings')
    parser.add_argument('--emb_right', type=str, default='data_2/prott5_pooled_fea/merged_right.npz',
                        help='NPZ file path for right subseq embeddings')

    parser.add_argument('--mode', type=str, choices=['a', 'b', 'c', 'd'], default='c',
                        help='Negative sampling mode')
    parser.add_argument('--outdir', type=str, default='results', help='Output directory')
    parser.add_argument('--time_limit', type=int, default=600, help='Time limit for AutoGluon (seconds)')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    run_experiment(args)
