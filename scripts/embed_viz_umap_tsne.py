#!/usr/bin/env python3
"""
embed_viz_umap_tsne.py
Run UMAP and/or t-SNE on feature vectors (or extract features from images)
and save scatter plots colored by labels.

Usage examples:

# If you already have features+labels:
python scripts/embed_viz_umap_tsne.py --features features.npy --labels labels.npy --outdir results/emb_viz --method both

# If you want to extract features from dataset using resnet18:
PYTHONPATH="$(pwd)" python scripts/embed_viz_umap_tsne.py \
  --data-dir data/FireRisk --dataset firerisk --batch-size 64 \
  --method both --outdir results/firerisk_emb --extract-with resnet18

"""
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import torchvision as tv
from sklearn.manifold import TSNE

# Attempt to import umap; give helpful error if missing
try:
    import umap
except Exception as e:
    umap = None

# --- label grouping mapping (adjust to your dataset labels) ---
HEAD_CLASSES = ["Very_Low", "Non-burnable"]
MEDIUM_CLASSES = ["Low", "Moderate"]
TAIL_CLASSES = ["High", "Very_High", "Water"]

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--features", type=str, default=None,
                   help="Path to saved feature vectors (.npy). If provided, labels must be provided too.")
    p.add_argument("--labels", type=str, default=None,
                   help="Path to saved labels (numpy .npy or a CSV with one label per line).")
    p.add_argument("--data-dir", type=str, default="data",
                   help="Data root (used if extracting features from images).")
    p.add_argument("--dataset", type=str, default="firerisk",
                   help="Dataset key registered in data.REGISTRY (used for extraction).")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--method", choices=["umap","tsne","both"], default="both")
    p.add_argument("--outdir", type=str, default="results/emb_viz")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--perplexity", type=float, default=30.0, help="t-SNE perplexity")
    p.add_argument("--umap-n-neighbors", type=int, default=15)
    p.add_argument("--umap-min-dist", type=float, default=0.1)
    p.add_argument("--extract-with", choices=["resnet18","resnet50"], default="resnet18",
                   help="Backbone used to extract features if --features not provided.")
    p.add_argument("--max-samples", type=int, default=None,
                   help="If set, sample at most this many datapoints (useful to speed t-SNE).")
    return p.parse_args()

def load_labels(path):
    if path.endswith(".npy"):
        return np.load(path)
    else:
        # assume simple CSV with one label per line
        return np.loadtxt(path, dtype=str, delimiter="\n")

def build_loader_from_registry(dataset_key, data_dir, batch_size):
    # Import registry (requires running from project root or PYTHONPATH set)
    from data import REGISTRY as DATA_REG
    if dataset_key not in DATA_REG:
        raise KeyError(f"Dataset '{dataset_key}' not found in data.REGISTRY keys: {list(DATA_REG.keys())}")
    train_ds, test_ds = DATA_REG[dataset_key](data_dir)
    # We'll use the test set for visualization
    loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return loader

def extract_features(loader, device, backbone_name="resnet18", max_samples=None):
    # build backbone
    if backbone_name == "resnet18":
        model = tv.models.resnet18(pretrained=True)
        feat_dim = 512
    else:
        model = tv.models.resnet50(pretrained=True)
        feat_dim = 2048

    # remove final fc and get features from avgpool
    model = model.to(device)
    model.eval()

    features = []
    labels = []

    def hook_fn(module, input, output):
        # output shape (N, C, 1, 1) for avgpool; flatten later
        features.append(output.detach().cpu().clone())

    # register forward hook on avgpool
    h = model.avgpool.register_forward_hook(hook_fn)

    with torch.no_grad():
        seen = 0
        for imgs, labs in loader:
            imgs = imgs.to(device)
            _ = model(imgs)  # hook will capture avgpool outputs
            # after forward, features list has appended a tensor for this batch
            # labs is a tensor -> collect
            if isinstance(labs, torch.Tensor):
                labels.append(labs.cpu().numpy())
            else:
                # some datasets return labels as scalars or lists
                labels.append(np.array(labs))
            seen += imgs.size(0)
            if max_samples and seen >= max_samples:
                break

    h.remove()

    # concatenate and flatten pooled features
    feats_t = torch.cat(features, dim=0)  # shape (N, C, 1, 1)
    feats_t = feats_t.view(feats_t.size(0), -1).numpy()
    labels = np.concatenate(labels, axis=0)
    return feats_t, labels

def subsample_if_needed(X, y, max_samples=None, random_seed=42):
    if (max_samples is None) or (len(X) <= max_samples):
        return X, y
    rng = np.random.RandomState(random_seed)
    idx = rng.choice(len(X), size=max_samples, replace=False)
    return X[idx], y[idx]

def plot_2d(X2, y, outpath, title="embedding", label_names=None):
    plt.figure(figsize=(9,7))
    unique_labels, inv = np.unique(y, return_inverse=True)
    # color by label index
    sc = plt.scatter(X2[:,0], X2[:,1], c=inv, s=8, alpha=0.8)
    # legend with label names
    handles = []
    for i, lbl in enumerate(unique_labels):
        handles.append(plt.Line2D([0],[0], marker='o', color='w', markerfacecolor=sc.cmap(sc.norm(i)), markersize=6))
    # if label names provided map them else use label text
    if label_names is not None:
        legend_labels = [label_names.get(str(lbl), str(lbl)) for lbl in unique_labels]
    else:
        legend_labels = [str(lbl) for lbl in unique_labels]
    plt.legend(handles, legend_labels, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()
    print(f"Saved: {outpath}")

def build_label_name_map(unique_labels):
    # map labels into HEAD/MEDIUM/TAIL explicit strings if possible
    mapping = {}
    for lbl in unique_labels:
        s = str(lbl)
        if s in HEAD_CLASSES:
            mapping[s] = f"{s} (head)"
        elif s in MEDIUM_CLASSES:
            mapping[s] = f"{s} (medium)"
        elif s in TAIL_CLASSES:
            mapping[s] = f"{s} (tail)"
        else:
            mapping[s] = s
    return mapping

def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # Load features+labels OR extract features
    if args.features:
        print("Loading features from", args.features)
        X = np.load(args.features)
        if args.labels is None:
            raise ValueError("If --features is provided you must supply --labels")
        y = load_labels(args.labels)
    else:
        print("Extracting features using dataset registry. Make sure PYTHONPATH includes project root.")
        loader = build_loader_from_registry(args.dataset, args.data_dir, args.batch_size)
        X, y = extract_features(loader, device=args.device, backbone_name=args.extract_with, max_samples=args.max_samples)
        print("Extracted features shape:", X.shape, "labels shape:", y.shape)

    # optional subsample for speed
    if args.max_samples:
        X, y = subsample_if_needed(X, y, max_samples=args.max_samples)
        print("After subsample shape:", X.shape)

    # Build label -> display name mapping
    unique_labels = np.unique(y).astype(str)
    label_name_map = build_label_name_map(unique_labels)

    # Run UMAP
    if args.method in ("umap","both"):
        if umap is None:
            print("UMAP not installed. Install with: pip install umap-learn")
        else:
            print("Running UMAP...")
            reducer = umap.UMAP(n_neighbors=args.umap_n_neighbors, min_dist=args.umap_min_dist, n_components=2, random_state=42)
            X_umap = reducer.fit_transform(X)
            plot_2d(X_umap, y, os.path.join(args.outdir, "umap_plot.png"), title="UMAP projection", label_names=label_name_map)

    # Run t-SNE
    if args.method in ("tsne","both"):
        print("Running t-SNE (this can be slow)...")
        tsne = TSNE(n_components=2, perplexity=args.perplexity, init="pca", random_state=42, n_jobs=8)
        X_tsne = tsne.fit_transform(X)
        plot_2d(X_tsne, y, os.path.join(args.outdir, "tsne_plot.png"), title=f"t-SNE (perp={args.perplexity})", label_names=label_name_map)

    print("Done.")

if __name__ == "__main__":
    main()
