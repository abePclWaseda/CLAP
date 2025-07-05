import laion_clap
import os
import json
import glob
import torch
import numpy as np

if not hasattr(np.random, "integers"):  # NumPy 2.0 対策
    np.random.integers = np.random.randint

# ===== 設定 =====
device = torch.device("cuda:1")
birdsong_dir = "/mnt/work-qnap/yuabe/kaggle/birdsong/train_audio"
class_index_dict_path = "/mnt/kiso-qnap3/yuabe/m1/CLAP/class_labels.json"
batch_size = 64  # ← バッチサイズは適宜調整してください

# ===== モデル読み込み =====
model = laion_clap.CLAP_Module(enable_fusion=False, device=device)
model.load_ckpt()

# ===== クラスラベルの読み込みと逆引き辞書作成 =====
with open(class_index_dict_path) as f:
    label2index = json.load(f)
index2label = {v: k for k, v in label2index.items()}

all_class_names = list(label2index.keys())
all_texts = ["This is a sound of " + name for name in all_class_names]
text_embed_np = model.get_text_embedding(all_texts)
text_embed = torch.tensor(text_embed_np, dtype=torch.float32).to(device)

# === few-shot の設定 ===
k_shot = 5  # 1クラス当たり support サンプル数
rng = np.random.default_rng(42)  # 再現性用

# === 1. support / query を分けて収集 ====================
support_paths = {c: [] for c in all_class_names}
query_paths = []
query_labels = []

for cls in all_class_names:
    cls_dir = os.path.join(birdsong_dir, cls)
    files = glob.glob(os.path.join(cls_dir, "*.mp3"))
    if len(files) < k_shot:
        raise ValueError(f"{cls}: 音声が {k_shot} 本未満です")

    rng.shuffle(files)  # ランダムに並べ替え
    support_paths[cls] = files[:k_shot]  # few-shot 用
    query_files = files[k_shot:]  # 評価用

    query_paths.extend(query_files)
    query_labels.extend([label2index[cls]] * len(query_files))

# === 2. プロトタイプ計算 =================================
prototypes = []
for cls in all_class_names:
    emb_np = model.get_audio_embedding_from_filelist(support_paths[cls])
    proto = torch.tensor(emb_np.mean(axis=0), dtype=torch.float32)  # 平均
    prototypes.append(proto)

prototypes = torch.stack(prototypes).to(device)  # (num_cls, D)

# === 3. クエリ推論（バッチ処理はそのまま） ==============
batch_size = 64
all_ranks = []

with torch.no_grad():
    for i in range(0, len(query_paths), batch_size):
        paths = query_paths[i : i + batch_size]
        truth = query_labels[i : i + batch_size]

        emb_np = model.get_audio_embedding_from_filelist(paths)
        emb = torch.tensor(emb_np, dtype=torch.float32).to(device)

        sim = emb @ prototypes.T  # (B, num_cls)
        ranks = torch.argsort(sim, dim=1, descending=True)

        gt = torch.tensor(truth, device=device).view(-1, 1)
        rank_pos = torch.where(ranks == gt)[1].cpu().numpy()
        all_ranks.append(rank_pos)

        del emb, sim, ranks, gt
        torch.cuda.empty_cache()

preds = np.concatenate(all_ranks)

# === 4. メトリクスも同様に集計 ==========================
# ここは元コードと同じ
# ===== メトリクスの計算 =====
metrics = {
    "mean_rank": preds.mean() + 1,
    "median_rank": np.floor(np.median(preds)) + 1,
}
for k in [1, 5, 10]:
    metrics[f"R@{k}"] = np.mean(preds < k)
metrics["mAP@10"] = np.mean(np.where(preds < 10, 1.0 / (preds + 1), 0.0))

# ===== 結果出力 =====
print("Zeroshot Classification Results:")
for k, v in metrics.items():
    print(f"{k}: {v:.4f}")
