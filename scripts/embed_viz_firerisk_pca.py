#!/usr/bin/env python3
"""
embed_viz_firerisk_pca.py

Usage (from project root):
  python scripts/embed_viz_firerisk_pca.py \
    --ckpt runs/firerisk_resnet_cifar_classifier/best.pth \
    --data-dir /path/to/firerisk_root \
    --batch-size 64 \
    --outdir results/firerisk_pca

Notes:
 - Run from project root or set PYTHONPATH=. so imports resolve.
 - Script assumes model.forward(x, return_feats=True) returns (logits, pooled_feats).
"""

import os
import argparse
import json
from collections import defaultdict

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# -------------------------
# User taxonomy - edit if needed
# These strings should match dataset.class names (or be normalized by the script).
head_classes   = ["Very_Low", "Non-burnable"]
medium_classes = ["Low", "Moderate"]
tail_classes   = ["High", "Very_High", "Water"]
# -------------------------

# Model import hints (edit if your module paths are different)
USE_DIRECT_IMPORT = True
DIRECT_IMPORT_MODULE = "models.resnet_cifar_classifier"
DIRECT_CLASS_NAME = "ResNetClassifier"
DIRECT_BLOCK_MODULE = "models.resnet_cifar"
DIRECT_BLOCK_NAME = "BasicBlock"


def normalize_name(s):
    """Normalize dataset class name and taxonomy items for tolerant matching:
       lower-case, replace spaces and hyphens with underscore, strip.
    """
    return s.strip().lower().replace(" ", "_").replace("-", "_")


def build_dataloader_from_registry(name, data_dir, batch_size):
    # Uses your data registry (data/__init__.py must define REGISTRY)
    from data import REGISTRY as DATA_REG
    if name not in DATA_REG:
        raise KeyError(f"Dataset '{name}' not in data.REGISTRY. Available: {list(DATA_REG.keys())}")
    train_ds, test_ds = DATA_REG[name](data_dir, image_size=224, to_rgb=False, imagenet_norm=True)
    from torch.utils.data import DataLoader
    return DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4), test_ds


def load_model_auto(ckpt_path, device, num_classes):
    model = None
    if USE_DIRECT_IMPORT:
        try:
            mod = __import__(DIRECT_IMPORT_MODULE, fromlist=[DIRECT_CLASS_NAME])
            ModelClass = getattr(mod, DIRECT_CLASS_NAME)
            try:
                bmod = __import__(DIRECT_BLOCK_MODULE, fromlist=[DIRECT_BLOCK_NAME])
                BlockClass = getattr(bmod, DIRECT_BLOCK_NAME)
            except Exception:
                BlockClass = None

            ctor_kwargs = dict(
                block=BlockClass,
                num_blocks=[5, 5, 5],
                num_classes=num_classes,
                scale=1,
                groups=1,
                nc=[16, 32, 64],
                drop=0.0,
            )
            model = ModelClass(**ctor_kwargs)
            print("Instantiated model via direct import.")
        except Exception as e:
            print("Direct import failed:", e)
            model = None

    if model is None:
        try:
            from models import REGISTRY as MODEL_REG
            # try a sensible default key
            if "resnet18" in MODEL_REG:
                model = MODEL_REG["resnet18"](num_classes=num_classes)
                print("Instantiated model via models.REGISTRY['resnet18']")
            else:
                first_key = next(iter(MODEL_REG.keys()))
                model = MODEL_REG[first_key](num_classes=num_classes)
                print(f"Instantiated model via models.REGISTRY['{first_key}']")
        except Exception as e:
            raise RuntimeError("Failed to instantiate model via direct import or registry. Edit load_model_auto(). Error: " + str(e))

    # load checkpoint
    state = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state_dict = state["state_dict"]
    elif isinstance(state, dict) and "model_state" in state:
        state_dict = state["model_state"]
    else:
        state_dict = state

    new_state = {}
    for k, v in state_dict.items():
        new_k = k[len("module."):] if k.startswith("module.") else k
        new_state[new_k] = v

    try:
        model.load_state_dict(new_state, strict=False)
    except Exception as e:
        print("Warning: load_state_dict(strict=False) raised:", e)

    model.to(device)
    model.eval()
    return model


def extract_features(model, dataloader, device):
    all_feats = []
    all_labels = []
    with torch.no_grad():
        for xb, yb in dataloader:
            xb = xb.to(device)
            out = model(xb, return_feats=True)
            if isinstance(out, tuple) and len(out) == 2:
                logits, pooled = out
            else:
                raise RuntimeError("Expected model(x, return_feats=True) -> (logits, pooled_feats)")
            all_feats.append(pooled.cpu().numpy())
            all_labels.append(yb.numpy())
    feats = np.concatenate(all_feats, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    return feats, labels


def plot_two_views(X_pca2, labels, test_ds, outdir):
    # Prepare mappings
    class_names = getattr(test_ds, "classes", None)
    if class_names is None:
        # fallback: use string of indices
        unique_idxs = np.unique(labels)
        class_names = [str(i) for i in unique_idxs]

    # normalized maps
    norm_idx2name = {i: normalize_name(class_names[i]) for i in range(len(class_names))}
    # reverse: normalized name -> original display name
    normname2display = {normalize_name(n): n for n in class_names}

    # taxonomy normalized
    head_norm = set(normalize_name(x) for x in head_classes)
    medium_norm = set(normalize_name(x) for x in medium_classes)
    tail_norm = set(normalize_name(x) for x in tail_classes)

    # idx -> group label
    idx2group = {}
    for i, normname in norm_idx2name.items():
        if normname in head_norm:
            idx2group[i] = "Head"
        elif normname in medium_norm:
            idx2group[i] = "Medium"
        elif normname in tail_norm:
            idx2group[i] = "Tail"
        else:
            idx2group[i] = "Unknown"

    # nice display names for legend: "Very_Low (Head)"
    idx2display = {i: f"{normname2display.get(norm_idx2name[i], norm_idx2name[i])} ({idx2group[i]})" for i in idx2group}

    # colors
    n_classes = len(class_names)
    cmap_classes = plt.get_cmap("tab20" if n_classes > 10 else "tab10")
    class_colors = [cmap_classes(i % cmap_classes.N) for i in range(n_classes)]
    group_palette = {"Head": (0.2, 0.6, 0.2), "Medium": (0.2, 0.4, 0.8), "Tail": (0.8, 0.2, 0.2), "Unknown": (0.6, 0.6, 0.6)}

    # class centroids
    class_centroids = {}
    for cls_idx in range(n_classes):
        mask = (labels == cls_idx)
        if mask.sum() == 0:
            continue
        class_centroids[cls_idx] = X_pca2[mask].mean(axis=0)

    # group points, centroids, counts
    group_points = defaultdict(list)
    for i, lab in enumerate(labels):
        grp = idx2group.get(lab, "Unknown")
        group_points[grp].append(X_pca2[i])
    group_centroids = {g: np.vstack(pts).mean(axis=0) for g, pts in group_points.items() if len(pts) > 0}
    group_counts = {g: len(pts) for g, pts in group_points.items()}

    # plotting
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    # Left: per-class colored points
    ax = axs[0]
    for cls_idx in range(n_classes):
        mask = (labels == cls_idx)
        if mask.sum() == 0:
            continue
        ax.scatter(X_pca2[mask, 0], X_pca2[mask, 1],
                   s=12, alpha=0.75, label=idx2display[cls_idx], color=class_colors[cls_idx])
    ax.set_title("PCA (per-class) — classes labelled with group")
    ax.axis("off")
    ax.legend(loc="best", markerscale=2, fontsize="small", ncol=1)

    # Right: grouped view with faint class dots and colored group overlays + centroids
    ax = axs[1]
    # faint class dots
    for cls_idx in range(n_classes):
        mask = (labels == cls_idx)
        if mask.sum() == 0:
            continue
        ax.scatter(X_pca2[mask, 0], X_pca2[mask, 1],
                   s=8, alpha=0.20, color=class_colors[cls_idx], linewidths=0)

    # overlay group points and annotate centroids
    for grp, pts in group_points.items():
        if len(pts) == 0:
            continue
        pts_arr = np.vstack(pts)
        ax.scatter(pts_arr[:, 0], pts_arr[:, 1], s=18, alpha=0.45, color=group_palette.get(grp, (0.5, 0.5, 0.5)), label=f"{grp} (n={group_counts.get(grp,0)})")
    for grp, cxy in group_centroids.items():
        ax.scatter([cxy[0]], [cxy[1]], s=180, marker="X", color=group_palette.get(grp, (0.5,0.5,0.5)), edgecolors="k")
        ax.text(cxy[0], cxy[1], f"  {grp}\n  n={group_counts.get(grp,0)}", fontsize=10, fontweight="bold",
                verticalalignment="center", horizontalalignment="left", bbox=dict(alpha=0.0))

    ax.set_title("PCA grouped view (Head / Medium / Tail)")
    ax.axis("off")
    ax.legend(loc="best", markerscale=1.8, fontsize="small", ncol=1)

    plt.tight_layout()
    out_png = os.path.join(outdir, "firerisk_pca_perclass_and_group.png")
    plt.savefig(out_png, dpi=200)
    plt.close(fig)

    # Save metadata
    meta = {
        "class_names": class_names,
        "idx2display": {str(k): v for k, v in idx2display.items()},
        "idx2group": {str(k): v for k, v in idx2group.items()},
        "group_counts": group_counts,
    }
    with open(os.path.join(outdir, "pca_meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)

    print("Saved combined PCA plot and metadata to:", outdir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="path to checkpoint .pth")
    parser.add_argument("--data-dir", required=True, help="root folder containing train/ and val/")
    parser.add_argument("--dataset", default="firerisk", help="registry key (default: firerisk)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--outdir", default="results/firerisk_pca")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.outdir, exist_ok=True)

    print("Building dataloader from registry...")
    dataloader, test_ds = build_dataloader_from_registry(args.dataset, args.data_dir, args.batch_size)
    try:
        num_classes = len(test_ds.classes)
    except Exception:
        try:
            num_classes = len(np.unique(test_ds.targets))
        except Exception:
            num_classes = 10
    print("Detected num_classes =", num_classes)

    print("Loading model and checkpoint...")
    model = load_model_auto(args.ckpt, device, num_classes)

    print("Extracting pooled features...")
    feats, labels = extract_features(model, dataloader, device)
    print("Features shape:", feats.shape)
    np.save(os.path.join(args.outdir, "features.npy"), feats)
    np.save(os.path.join(args.outdir, "labels.npy"), labels)

    print("Running PCA -> 2D ...")
    pca2 = PCA(n_components=2)
    X_pca2 = pca2.fit_transform(feats)
    np.save(os.path.join(args.outdir, "pca2.npy"), X_pca2)

    





        # --- after features/labels saved ---
    print("Features shape:", feats.shape)
    np.save(os.path.join(args.outdir, "features.npy"), feats)
    np.save(os.path.join(args.outdir, "labels.npy"), labels)

    # Diagnostic (helpful if you saw only a few labels)
    print("labels dtype:", labels.dtype, "shape:", labels.shape)
    uniq, counts = np.unique(labels, return_counts=True)
    print("unique labels (values):", uniq)
    print("counts per unique label:", dict(zip(uniq, counts)))
    print("test_ds.classes:", getattr(test_ds, "classes", None))

    # 1) PCA -> 2D (existing)
    print("Running PCA -> 2D ...")
    pca2 = PCA(n_components=2)
    X_pca2 = pca2.fit_transform(feats)
    np.save(os.path.join(args.outdir, "pca2.npy"), X_pca2)
    print("Plotting per-class + grouped PCA views ...")
    plot_two_views(X_pca2, labels, test_ds, args.outdir)

    # Helper reused by UMAP/t-SNE for plotting (you can move it above main if you prefer)
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA as SKPCA
    try:
        import umap
    except Exception:
        umap = None

    def prepare_mapped_names(labels, test_ds):
        labels = np.asarray(labels)
        uniq = np.unique(labels)
        if np.issubdtype(labels.dtype, np.integer) and getattr(test_ds, "classes", None) is not None:
            mapped_names = [ test_ds.classes[int(u)] for u in uniq ]
        else:
            mapped_names = [ str(u) for u in uniq ]
        label_to_idx = {v:i for i,v in enumerate(uniq)}
        plot_idx = np.array([label_to_idx[v] for v in labels])
        return mapped_names, uniq, plot_idx

    def plot_embedding_two_views(X2, labels, test_ds, outdir, title_prefix="embedding"):
        mapped_names, uniq, plot_idx = prepare_mapped_names(labels, test_ds)
        head_norm = set(normalize_name(x) for x in head_classes)
        medium_norm = set(normalize_name(x) for x in medium_classes)
        tail_norm = set(normalize_name(x) for x in tail_classes)

        idx2group = {}
        for i, nm in enumerate(mapped_names):
            nn = normalize_name(nm)
            if nn in head_norm:
                idx2group[i] = "Head"
            elif nn in medium_norm:
                idx2group[i] = "Medium"
            elif nn in tail_norm:
                idx2group[i] = "Tail"
            else:
                idx2group[i] = "Unknown"
        idx2display = {i: f"{mapped_names[i]} ({idx2group[i]})" for i in range(len(mapped_names))}

        n_classes = len(mapped_names)
        cmap_classes = plt.get_cmap("tab20" if n_classes > 10 else "tab10")
        class_colors = [cmap_classes(i % cmap_classes.N) for i in range(n_classes)]
        group_palette = {"Head": (0.2, 0.6, 0.2), "Medium": (0.2, 0.4, 0.8), "Tail": (0.8, 0.2, 0.2), "Unknown": (0.6, 0.6, 0.6)}

        from collections import defaultdict
        group_points = defaultdict(list)
        for i, pi in enumerate(plot_idx):
            grp = idx2group.get(pi, "Unknown")
            group_points[grp].append(X2[i])
        group_centroids = {g: np.vstack(pts).mean(axis=0) for g, pts in group_points.items() if len(pts) > 0}
        group_counts = {g: len(pts) for g, pts in group_points.items()}

        fig, axs = plt.subplots(1, 2, figsize=(14, 6))
        ax = axs[0]
        for cls_idx in range(n_classes):
            mask = (plot_idx == cls_idx)
            if mask.sum() == 0:
                continue
            ax.scatter(X2[mask, 0], X2[mask, 1], s=12, alpha=0.75, label=idx2display[cls_idx], color=class_colors[cls_idx])
        ax.set_title(f"{title_prefix} (per-class) — classes labelled with group")
        ax.axis("off")
        ax.legend(loc="best", markerscale=2, fontsize="small", ncol=1)

        ax = axs[1]
        for cls_idx in range(n_classes):
            mask = (plot_idx == cls_idx)
            if mask.sum() == 0:
                continue
            ax.scatter(X2[mask, 0], X2[mask, 1], s=8, alpha=0.20, color=class_colors[cls_idx], linewidths=0)

        for grp, pts in group_points.items():
            if len(pts) == 0:
                continue
            pts_arr = np.vstack(pts)
            ax.scatter(pts_arr[:, 0], pts_arr[:, 1], s=18, alpha=0.45, color=group_palette.get(grp, (0.5,0.5,0.5)), label=f"{grp} (n={group_counts.get(grp,0)})")
        for grp, cxy in group_centroids.items():
            ax.scatter([cxy[0]], [cxy[1]], s=180, marker="X", color=group_palette.get(grp, (0.5,0.5,0.5)), edgecolors="k")
            ax.text(cxy[0], cxy[1], f"  {grp}\n  n={group_counts.get(grp,0)}", fontsize=10, fontweight="bold", verticalalignment="center", horizontalalignment="left", bbox=dict(alpha=0.0))

        ax.set_title(f"{title_prefix} grouped view (Head / Medium / Tail)")
        ax.axis("off")
        ax.legend(loc="best", markerscale=1.8, fontsize="small", ncol=1)

        plt.tight_layout()
        out_png = os.path.join(outdir, f"{title_prefix.lower().replace(' ','_')}_perclass_and_group.png")
        plt.savefig(out_png, dpi=200)
        plt.close(fig)

        meta = {
            "mapped_names": mapped_names,
            "idx2display": {str(k): v for k, v in idx2display.items()},
            "idx2group": {str(k): v for k, v in idx2group.items()},
            "group_counts": group_counts,
        }
        with open(os.path.join(outdir, f"{title_prefix.lower().replace(' ','_')}_meta.json"), "w") as fh:
            json.dump(meta, fh, indent=2)
        print("Saved:", out_png)

    # --- UMAP ---
    if umap is None:
        print("UMAP not installed; skipping UMAP. Install with: pip install umap-learn")
    else:
        print("Running UMAP...")
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
        X_umap = reducer.fit_transform(feats)
        np.save(os.path.join(args.outdir, "umap.npy"), X_umap)
        plot_embedding_two_views(X_umap, labels, test_ds, args.outdir, title_prefix="UMAP")
        print("UMAP done. Saved embedding and metadata to:", args.outdir)

    # --- t-SNE (with optional PCA pre-reduction for speed) ---
    X_for_tsne = feats
    if X_for_tsne.shape[1] > 50:
        print("Reducing dims with PCA(50) before t-SNE...")
        p50 = SKPCA(n_components=50, random_state=42)
        X_for_tsne = p50.fit_transform(X_for_tsne)

    print("Running t-SNE (this can be slow)...")
    tsne = TSNE(n_components=2, perplexity=30.0, init="pca", random_state=42, n_jobs=8)
    X_tsne = tsne.fit_transform(X_for_tsne)
    np.save(os.path.join(args.outdir, "tsne.npy"), X_tsne)
    plot_embedding_two_views(X_tsne, labels, test_ds, args.outdir, title_prefix="t-SNE")
    print("t-SNE done. Saved embedding and metadata to:", args.outdir)

    print("Done.")


    print("Plotting per-class + grouped PCA views ...")
    plot_two_views(X_pca2, labels, test_ds, args.outdir)

    print("Done.")

if __name__ == "__main__":
    main()
