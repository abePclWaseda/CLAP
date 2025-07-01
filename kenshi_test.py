import torch, laion_clap, torchaudio
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity  # pip install scikit-learn
import numpy as np

if not hasattr(np.random, "integers"):  # NumPy 2.0 対策
    np.random.integers = np.random.randint

device = torch.device("cuda:0")

# 1) モデル読み込み ── music_audioset_... を例に
model = laion_clap.CLAP_Module(
    enable_fusion=False,
    amodel="HTSAT-base",
    device=device,
)
model.load_ckpt("/home/yuabe/checkpoints/music_audioset_epoch_15_esc_90.14.pt")
model.eval()

# 2) Lemon の音声ファイル（wav, flac, mp3 など）を用意してパスを書く
flac_path = Path("/mnt/kiso-qnap3/yuabe/m1/CLAP/data/kenshi/Lemon.flac")

text_prompt = "米津玄師"
# 3) 埋め込み計算
with torch.no_grad():
    # ------- Audio side -------
    audio_emb = model.get_audio_embedding_from_filelist(
        [str(flac_path)]
    )  # ← NumPy (1, D)
    audio_emb = torch.as_tensor(audio_emb, device=device)  # Tensor 化
    audio_emb = audio_emb / audio_emb.norm(dim=1, keepdim=True)  # 正規化

    # ------- Text side -------
    text_emb = model.get_text_embedding([text_prompt])  # ← NumPy (1, D)
    text_emb = torch.as_tensor(text_emb, device=device)
    text_emb = text_emb / text_emb.norm(dim=1, keepdim=True)

# ------- Cosine similarity -------
score = float((audio_emb @ text_emb.T).item())
print(f"Similarity(Lemon ↔ 『{text_prompt}』): {score:.4f}")
