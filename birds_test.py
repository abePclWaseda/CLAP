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

# ===== 音声ファイルと正解ラベルの収集 =====
audio_paths = []
ground_truth_indices = []
for class_name in all_class_names:
    class_dir = os.path.join(birdsong_dir, class_name)
    if not os.path.isdir(class_dir):
        continue
    files = glob.glob(os.path.join(class_dir, "*.mp3"))
    audio_paths.extend(files)
    ground_truth_indices.extend([label2index[class_name]] * len(files))

print(f"Found {len(audio_paths)} audio files.")

# ===== 推論とランキング（バッチ処理） =====
all_preds = []
with torch.no_grad():
    for i in range(0, len(audio_paths), batch_size):
        batch_paths = audio_paths[i : i + batch_size]
        batch_truth = ground_truth_indices[i : i + batch_size]

        # 音声埋め込み取得
        emb_np = model.get_audio_embedding_from_filelist(batch_paths)
        emb = torch.tensor(emb_np, dtype=torch.float32).to(device)

        # 類似度計算 → ランキング
        sim = emb @ text_embed.T  # (batch_size, num_classes)
        ranking = torch.argsort(sim, descending=True, dim=1)

        # 各サンプルの正解クラス順位を取得
        gt = torch.tensor(batch_truth, dtype=torch.long, device=device).view(-1, 1)
        preds = torch.where(ranking == gt)[1].cpu().numpy()  # 0-based rank
        all_preds.append(preds)

        # メモリ開放
        del emb, sim, ranking, gt
        torch.cuda.empty_cache()

# バッチごとの結果をまとめる
preds = np.concatenate(all_preds, axis=0)

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
