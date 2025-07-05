import os
import json

# 音声ディレクトリのパス
base_path = "/mnt/work-qnap/yuabe/kaggle/birdsong/train_audio"

# サブディレクトリ一覧を取得し、ソート
class_names = sorted(
    [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
)

# クラス名にインデックスを付与
label_map = {name: idx for idx, name in enumerate(class_names)}

# JSONファイルに保存（例: class_labels.json）
with open("class_labels.json", "w") as f:
    json.dump(label_map, f, indent=2)

# 結果の表示（確認用）
# print(json.dumps(label_map, indent=2))
