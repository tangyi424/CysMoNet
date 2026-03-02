# File: main.py
import argparse
import os
import numpy as np
from data_builder import load_sequence_dict, negative_sampling_strategy, build_feature_label_list, make_outer_folds, make_inner_folds
from embeddings_loader import load_npz_embeddings, EmbeddingStore
from autogluon_runner import train_and_evaluate_ag



def run_experiment(args):
    import json

    # --- load positives ---
    with open(args.all_pos, "r", encoding="utf8") as f:
        pos_dict = json.load(f)       # {"1": "...", "2": "..."}
    positives = list(pos_dict.values())

    # --- load negatives ---
    with open(args.all_neg, "r", encoding="utf8") as f:
        neg_dict = json.load(f)
    negatives = list(neg_dict.values())

    print(f"Loaded {len(positives)} positives, {len(negatives)} negatives")

    # --- load embeddings ---
    emb_all_map = load_npz_embeddings(args.emb_all)
    emb_left_map = load_npz_embeddings(args.emb_left)
    emb_right_map = load_npz_embeddings(args.emb_right)
    emb_all = EmbeddingStore(emb_all_map)
    emb_left = EmbeddingStore(emb_left_map)
    emb_right = EmbeddingStore(emb_right_map)

    # --- negative sampling ---
    datasets = negative_sampling_strategy(positives, negatives, args.mode, seed=args.seed)

    all_results = []
    for ds_idx, (pos_list, neg_list) in enumerate(datasets):
        print(f"Dataset {ds_idx}: pos={len(pos_list)} neg={len(neg_list)}")

        # 构造特征和标签
        X, y = build_feature_label_list(pos_list, neg_list, emb_all, emb_left, emb_right)

        # === 1️⃣ 划分独立测试集（只做一次） ===
        outer = make_outer_folds(X, y, n_splits=5, seed=args.seed)
        trainval_idx, test_idx = next(iter(outer))  # 只取第一个fold作为独立测试集
        X_trainval, y_trainval = X[trainval_idx], y[trainval_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        # === 2️⃣ 对训练+验证集进行 10 折交叉验证 ===
        inner = make_outer_folds(X_trainval, y_trainval, n_splits=10, seed=args.seed)

        for fold_idx, (train_idx, val_idx) in enumerate(inner):
            X_train, y_train = X_trainval[train_idx], y_trainval[train_idx]
            X_val, y_val = X_trainval[val_idx], y_trainval[val_idx]

            outdir = os.path.join(args.outdir, f'dataset{ds_idx}_fold{fold_idx}')
            print(f"\n[Dataset {ds_idx}] Fold {fold_idx}: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

            # === 3️⃣ 训练并在独立测试集上评估 ===
            metrics, predictor = train_and_evaluate_ag(
                X_train, y_train, X_val, y_val, X_test, y_test,
                outdir, time_limit=args.time_limit, seed=args.seed
            )

            all_results.append({
                'dataset': ds_idx,
                'fold': fold_idx,
                'metrics': metrics
            })

    with open(os.path.join(args.outdir, 'summary_results.json'), 'w', encoding='utf8') as f:
        json.dump(all_results, f, indent=2)
    print('All experiments done.')




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--all_pos', type=str, default='data_2/all_pos_2.txt', help='positive samples txt/json')
    parser.add_argument('--all_neg', type=str, default='data_2/all_neg_2.txt', help='negative samples txt/json')

    # parser.add_argument('--emb_all', type=str, default='data_2/esm2_fea/merged_full.npz', help='NPZ file path for all-sequence embeddings')
    # parser.add_argument('--emb_left', type=str, default='data_2/esm2_fea/merged_left.npz', help='NPZ file path for left subseq embeddings')
    # parser.add_argument('--emb_right', type=str, default='data_2/esm2_fea/merged_right.npz', help='NPZ file path for right subseq embeddings')

    # parser.add_argument('--emb_all', type=str, default='data_2/blosum_fea/blosum_full.npz',help ='NPZ file path for all-sequence embeddings')
    # parser.add_argument('--emb_left', type=str, default='data_2/blosum_fea/blosum_left.npz',help='NPZ file path for left subseq embeddings')
    # parser.add_argument('--emb_right', type=str, default='data_2/blosum_fea/blosum_right.npz',help='NPZ file path for right subseq embeddings')

    parser.add_argument('--emb_all', type=str, default='data_2/ProtT5+atom/merged_full.npz',help='NPZ file path for all-sequence embeddings')
    parser.add_argument('--emb_left', type=str, default='data_2/ProtT5+atom/merged_left.npz',help='NPZ file path for left subseq embeddings')
    parser.add_argument('--emb_right', type=str, default='data_2/ProtT5+atom/merged_right.npz',help='NPZ file path for right subseq embeddings')


    parser.add_argument('--mode', type=str, choices=['a','b','c','d'], default='a', help='Negative sampling mode')
    parser.add_argument('--outdir', type=str, default='results', help='Output directory')
    parser.add_argument('--time_limit', type=int, default=600, help='Time limit for AutoGluon (seconds)')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    run_experiment(args)