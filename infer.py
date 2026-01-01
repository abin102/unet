import argparse
import os
import csv
import torch
import torch.nn as nn

from types import SimpleNamespace

from utils.config_utils import load_cfg
from utils.data_utils import make_dataloaders
from utils.builders import build_model_from_cfg, build_loss
from utils.checkpoint_utils import load_checkpoint
from utils.eval_utils import evaluate
from data import REGISTRY as DATA_REG


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = load_cfg(args.cfg)
    os.makedirs(args.out, exist_ok=True)

    # -------------------------
    # Device
    # -------------------------
    gpu_ids = cfg.get("system", {}).get("gpu_ids", [0])
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------------
    # Dataset (VAL as test)
    # -------------------------
    data_args = cfg.get("data_args", {}) or {}
    _, val_set = DATA_REG[cfg["dataset"]](cfg["data_dir"], **data_args)

    # make_dataloaders REQUIRES train_set != None
    train_loader, val_loader = make_dataloaders(
        val_set,   # dummy train_set
        val_set,
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
        pin_memory=cfg["pin_memory"],
        persistent_workers=cfg["persistent_workers"],
        shuffle=False,
    )

    # -------------------------
    # Model
    # -------------------------
    model_cfg = {"name": cfg["model"], "params": cfg.get("model_args", {})}
    model = build_model_from_cfg(model_cfg, device=device)

    if len(gpu_ids) > 1 and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    load_checkpoint(args.ckpt, model, optimizer=None, scheduler=None, trainer=None)
    model.eval()

    # -------------------------
    # Loss (needed for val_loss)
    # -------------------------
    loss_fn = build_loss(
        loss_name=cfg["loss"],
        loss_args=cfg.get("loss_args", {}),
        device=device,
    )

    # -------------------------
    # Minimal trainer-like object
    # -------------------------
    fake_trainer = SimpleNamespace(
        model=model,
        loss_fn=loss_fn,
        device=device,
        cfg=cfg,
        progress_bar=True,
        tb_writer=None,   # disable tensorboard
    )

    # -------------------------
    # Run evaluation (YOUR EXACT METRICS)
    # -------------------------
    val_loss, main_score, stats = evaluate(fake_trainer, val_loader, epoch=0)

    # -------------------------
    # Write CSV (training-compatible)
    # -------------------------
    csv_path = os.path.join(args.out, "metrics_repar_unet.csv")

    header = ["epoch", "train_loss", "train_acc(%)", "val_loss"]
    header += cfg.get("metrics", [])

    row = [
        0,              # epoch
        "",             # train_loss (N/A)
        "",             # train_acc (N/A)
        f"{val_loss:.4f}",
    ]

    for m in cfg.get("metrics", []):
        row.append(f"{stats.get(m, ''):.4f}" if m in stats else "")

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerow(row)

    print("\n✅ Clean inference complete")
    print(f"📄 Metrics saved to: {csv_path}")


if __name__ == "__main__":
    main()
