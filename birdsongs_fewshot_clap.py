"""Few‑shot birdsong classification with CLAP (prototype method).

Usage:
    python birdsongs_fewshot_clap.py --audio_dir /path/to/train_audio \
                                      --labels_json /path/to/class_labels.json \
                                      --device cuda:0 --k_shot 5 --batch 64
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import laion_clap

if not hasattr(np.random, "integers"):  # NumPy 2.0 対策
    np.random.integers = np.random.randint

###############################################################################
# Utility functions
###############################################################################


def seed_everything(seed: int = 42) -> None:
    """Set RNG seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def l2_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-12) -> torch.Tensor:
    """Row‑wise L2 normalisation."""
    return x / (x.norm(dim=dim, keepdim=True).clamp(min=eps))


###############################################################################
# Data handling
###############################################################################


def load_class_labels(path: Path) -> Tuple[Dict[str, int], Dict[int, str]]:
    with path.open() as f:
        label2idx: Dict[str, int] = json.load(f)
    idx2label = {v: k for k, v in label2idx.items()}
    return label2idx, idx2label


def split_support_query(
    audio_root: Path,
    classes: Sequence[str],
    k_shot: int,
    seed: int = 42,
) -> Tuple[Dict[str, List[Path]], List[Path], List[int]]:
    """Return support paths per class and flat query paths/labels."""
    rng = np.random.default_rng(seed)
    support: Dict[str, List[Path]] = {c: [] for c in classes}
    query_paths: List[Path] = []
    query_labels: List[int] = []

    for cls in classes:
        files = list((audio_root / cls).glob("*.mp3"))
        if len(files) <= k_shot:
            print(
                f"[warn] {cls}: only {len(files)} files (<= k_shot); skipping queries."
            )
            support[cls] = files  # still use as support (could be < k_shot)
            continue

        rng.shuffle(files)
        support[cls] = files[:k_shot]
        for p in files[k_shot:]:
            query_paths.append(p)
            query_labels.append(classes.index(cls))
    return support, query_paths, query_labels


###############################################################################
# Embedding helpers
###############################################################################


def embed_paths(
    model: laion_clap.CLAP_Module,
    paths: Sequence[Path],
    device: torch.device,
    batch_size: int = 128,
) -> torch.Tensor:
    """Embed a list of audio paths and return a (N, D) tensor on CPU."""
    embeddings: List[torch.Tensor] = []
    for i in range(0, len(paths), batch_size):
        batch_paths = [str(p) for p in paths[i : i + batch_size]]
        emb_np = model.get_audio_embedding_from_filelist(batch_paths)
        emb = torch.tensor(emb_np, dtype=torch.float32)
        embeddings.append(emb)
    return torch.cat(embeddings, dim=0)


###############################################################################
# Metric calculation
###############################################################################


def calc_metrics(ranks: np.ndarray) -> Dict[str, float]:
    metrics: Dict[str, float] = {
        "mean_rank": ranks.mean() + 1,
        "median_rank": np.median(ranks) + 1,
    }
    for k in (1, 5, 10):
        metrics[f"R@{k}"] = float((ranks < k).mean())
    metrics["mAP@10"] = float(np.where(ranks < 10, 1.0 / (ranks + 1), 0.0).mean())
    return metrics


###############################################################################
# Main pipeline
###############################################################################


def main(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    device = torch.device(args.device)

    # Model -------------------------------------------------------------------
    model = laion_clap.CLAP_Module(enable_fusion=False, device=device)
    model.load_ckpt()

    # Labels ------------------------------------------------------------------
    label2idx, idx2label = load_class_labels(Path(args.labels_json))
    class_names = list(label2idx.keys())

    # Few‑shot split -----------------------------------------------------------
    support, query_paths, query_labels = split_support_query(
        Path(args.audio_dir), class_names, args.k_shot, args.seed
    )

    if not query_paths:
        sys.exit("No query samples found. Reduce k_shot or check data directory.")

    # -------------------------------------------------------------------------
    # 1. Compute prototypes (support)
    # -------------------------------------------------------------------------
    all_support_paths = [p for paths in support.values() for p in paths]
    sup_emb = embed_paths(model, all_support_paths, device, args.batch)
    sup_emb = l2_normalize(sup_emb)

    # Map each slice to its class prototype
    ptr = 0
    prototypes: List[torch.Tensor] = []
    for cls in class_names:
        n = len(support[cls])
        proto = sup_emb[ptr : ptr + n].mean(0)
        prototypes.append(proto)
        ptr += n
    prototypes = torch.stack(prototypes).to(device)  # (C, D)

    # -------------------------------------------------------------------------
    # 2. Evaluate queries
    # -------------------------------------------------------------------------
    ranks_all: List[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(query_paths), args.batch):
            paths = query_paths[i : i + args.batch]
            truth = torch.tensor(query_labels[i : i + args.batch], device=device).view(
                -1, 1
            )

            q_emb = embed_paths(model, paths, device, args.batch).to(device)
            q_emb = l2_normalize(q_emb)

            sim = q_emb @ prototypes.T  # (B, C)
            ranked = torch.argsort(sim, dim=1, descending=True)
            rank_pos = torch.where(ranked == truth)[1].cpu().numpy()
            ranks_all.append(rank_pos)

    ranks = np.concatenate(ranks_all)
    metrics = calc_metrics(ranks)

    print("\nFew‑shot Classification Results (k = {}):".format(args.k_shot))
    for k, v in metrics.items():
        print(f"{k:>11}: {v:.4f}")


###############################################################################
# CLI
###############################################################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Few‑shot birdsong classifier (CLAP prototype)"
    )
    parser.add_argument(
        "--audio_dir", required=True, help="Root directory of <class>/<audio>.mp3 files"
    )
    parser.add_argument("--labels_json", required=True, help="class_labels.json path")
    parser.add_argument(
        "--device", default="cuda:0", help="PyTorch device (default: cuda:0)"
    )
    parser.add_argument(
        "--k_shot", type=int, default=5, help="Number of support samples per class"
    )
    parser.add_argument("--batch", type=int, default=64, help="Embedding batch size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    main(args)
