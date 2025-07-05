import laion_clap
import os
import json
import glob
import torch
import numpy as np

# ===== 設定 =====
device = torch.device("cuda:1")
birdsong_dir = "/mnt/work-qnap/yuabe/kaggle/birdsong/train_audio"
class_index_dict_path = "/mnt/kiso-qnap3/yuabe/m1/CLAP/class_labels.json"

# ===== モデル読み込み =====
model = laion_clap.CLAP_Module(enable_fusion=False, device=device)
model.load_ckpt()

# ===== クラスラベルの読み込みと逆引き辞書作成 =====
with open(class_index_dict_path) as f:
    label2index = json.load(f)
index2label = {v: k for k, v in label2index.items()}

# ===== クラス名リストとテキスト埋め込み作成 =====
all_class_names = list(label2index.keys())
all_texts = ["This is a sound of " + name for name in all_class_names]
text_embed = model.get_text_embedding(all_texts)
text_embed = torch.tensor(text_embed).to(device)

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

# ===== 推論とランキング =====
with torch.no_grad():
    audio_embed = model.get_audio_embedding_from_filelist(audio_paths)
    audio_embed = torch.tensor(audio_embed).to(device)

    similarity = audio_embed @ text_embed.T  # (N, C)
    ranking = torch.argsort(similarity, descending=True, dim=1)

    ground_truth = torch.tensor(ground_truth_indices).to(device).view(-1, 1)
    preds = torch.where(ranking == ground_truth)[1].cpu().numpy()

# ===== メトリクスの計算 =====
metrics = {
    "mean_rank": preds.mean() + 1,
    "median_rank": np.floor(np.median(preds)) + 1,
}
for k in [1, 5, 10]:
    metrics[f"R@{k}"] = np.mean(preds < k)
metrics["mAP@10"] = np.mean(np.where(preds < 10, 1 / (preds + 1), 0.0))

# ===== 結果出力 =====
print("Zeroshot Classification Results:")
for k, v in metrics.items():
    print(f"{k}: {v:.4f}")
