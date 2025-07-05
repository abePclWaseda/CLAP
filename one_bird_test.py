import numpy as np

# ---- NumPy 互換パッチ（必ず laion_clap より前で）----
if not hasattr(np.random, "integers"):
    np.random.integers = np.random.randint

import laion_clap, torch, json

# ---------- 設定 ----------
device = torch.device("cuda:0") 
audio_file = "/mnt/work-qnap/yuabe/kaggle/birdsong/train_audio/aldfly/XC2628.mp3"
class_index_dict_path = "/mnt/kiso-qnap3/yuabe/m1/CLAP/class_labels.json"

# ---------- モデル ----------
model = laion_clap.CLAP_Module(enable_fusion=False, device=device)
model.load_ckpt()  # 重みをロード

# ---------- ラベルとテキスト埋め込み ----------
with open(class_index_dict_path) as f:
    label2idx = json.load(f)  # {"aldfly": 0, ...}
idx2label = {v: k for k, v in label2idx.items()}

texts = [f"This is a sound of {lbl}" for lbl in label2idx.keys()]
text_emb = torch.tensor(
    model.get_text_embedding(texts), dtype=torch.float32, device=device
)  # (C, D)

# ---------- 音声埋め込み ----------
audio_emb = torch.tensor(
    model.get_audio_embedding_from_filelist([audio_file]),
    dtype=torch.float32,
    device=device,  # (1, D)
)

# ---------- 類似度 & 予測 ----------
sim = (audio_emb @ text_emb.T).squeeze(0)  # (C,)
top_idx = int(torch.argmax(sim))
pred_label = idx2label[top_idx]
score = sim[top_idx].item()

print(f"Pred label : {pred_label}")
print(f"Similarity : {score:.4f}")
