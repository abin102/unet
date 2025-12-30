from utils.logging_utils import init_logger_and_tb
import sys
import torch
import torch.nn as nn  # <--- CHANGE 1: Needed for DataParallel
import argparse, os
from utils.config_utils import load_cfg
from utils.data_utils import make_dataloaders
from utils.builders import build_model_from_cfg, get_optim, get_scheduler
from utils.data_stats import save_dataset_stats
from data import REGISTRY as DATA_REG
from models import REGISTRY as MODEL_REG
from callbacks import Checkpoint, CSVLogger, ModelLogger
from trainer import Trainer
from callbacks.logging import LoggingCallback
from utils.freeze_utils import apply_freeze
from utils.checkpoint_utils import load_checkpoint
from utils.wandb_utils import init_wandb
from utils.builders import build_loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--over", nargs="*", default=[])
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()

    cfg = load_cfg(args.cfg, overrides=args.over)
    seed = cfg.get("seed", 42)
    out_dir = f'{cfg["output_dir"]}/{cfg["exp_name"]}-{__import__("time").strftime("%Y%m%d-%H%M%S")}'
    os.makedirs(out_dir, exist_ok=True)

    # ====================================================
    # CHANGE 2: GPU Setup from Config
    # ====================================================
    # Read gpu_ids from config (default to [0] if missing)
    gpu_ids = cfg.get("system", {}).get("gpu_ids", [0])
    
    # Force PyTorch to only see the requested GPUs
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
    
    # ----------------------------------------------------

    # 1. Logging Setup
    log_cfg = cfg.get("logging", {}) or {}
    tb_logdir = os.path.join(out_dir, "tb") if log_cfg.get("tensorboard", False) else None
    if tb_logdir: os.makedirs(tb_logdir, exist_ok=True)

    logger, tb_writer = init_logger_and_tb(
        debug=log_cfg.get("debug", False),
        debug_dir=out_dir,
        tb_logdir=tb_logdir,
        log_level=log_cfg.get("level", "INFO"),
    )

    wandb, use_wandb = init_wandb(cfg, out_dir)

    # 2. Seeding
    import random, numpy as _np
    random.seed(seed)
    _np.random.seed(seed)
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.backends.cudnn.deterministic = True

    # Check device availability after setting CUDA_VISIBLE_DEVICES
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"🚀 Active GPUs: {gpu_ids} (Visible count: {torch.cuda.device_count()})")
    else:
        device = torch.device("cpu")
        logger.info("⚠️ No GPU detected, using CPU.")
    
    logger.info("Output Directory: {}", out_dir)

    cb = LoggingCallback(tb_logdir=tb_logdir, use_wandb=use_wandb)

    # 3. Data
    data_args = cfg.get("data_args", {}) or {}
    # Assuming your ct_dataset returns (image, mask)
    train_set, val_set = DATA_REG[cfg["dataset"]](cfg["data_dir"], **data_args)
    
    # Optional: Save stats
    try:
        save_dataset_stats(out_dir, train_set, val_set, cfg=cfg)
    except:
        pass # Segmentation stats might differ in structure, ignore if fails

    train_loader, val_loader = make_dataloaders(train_set, val_set, batch_size=cfg["batch_size"], 
                                                num_workers=cfg["num_workers"], pin_memory=cfg["pin_memory"], 
                                                persistent_workers=cfg["persistent_workers"])

    # 4. Model
    model_cfg = {"name": cfg["model"], "params": cfg.get("model_args", {})}
    model = build_model_from_cfg(model_cfg, device=device)
    logger.info("Model '{}' built.", cfg["model"])

    # ====================================================
    # CHANGE 3: Wrap Model with DataParallel
    # ====================================================
    # This must happen BEFORE optimizer creation so optimizer sees all params
    if len(gpu_ids) > 1 and torch.cuda.device_count() > 1:
        logger.info(f"⚡ Enabling DataParallel on {len(gpu_ids)} GPUs!")
        model = nn.DataParallel(model)
    # ----------------------------------------------------

    # 5. Loss & Optimizer
    # Direct CrossEntropyLoss for segmentation
    loss_fn = build_loss(
        loss_name=cfg["loss"],
        loss_args=cfg.get("loss_args", {}),
        device=device
    )
    
    optim = get_optim(cfg["optim"], [p for p in model.parameters() if p.requires_grad], **(cfg.get("optim_args", {}) or {}))
    sched = get_scheduler(cfg["scheduler"], optim, **(cfg.get("scheduler_args", {}) or {}))

    # 6. Checkpoint Loading
    start_epoch = 1
    if args.resume:
        # Note: Our updated load_checkpoint handles the model wrapper keys automatically
        ck_info = load_checkpoint(args.resume, model, optimizer=optim, scheduler=sched, trainer=None)
        start_epoch = int(ck_info.get("start_epoch", 1))
        logger.info(f"[resume] loaded checkpoint from {args.resume}, resuming at epoch {start_epoch}")

    # 7. Callbacks
    cbs = [
        CSVLogger(out_dir), 
        # Note: We monitor the metric defined in config (e.g. dice_infection)
        Checkpoint(out_dir, monitor=cfg["early_stop"]["monitor"], mode=cfg["early_stop"]["mode"]), 
        # mdl_logger removed or simplified as input_size inference is tricky for segmentation
    ]
    all_cbs = cbs + [cb]

    # 8. Trainer
    trainer = Trainer(model=model, loss_fn=loss_fn, optimizer=optim, scheduler=sched, 
                      amp=cfg["amp"], grad_clip=cfg["grad_clip"], device=device, callbacks=all_cbs, 
                      tb_logdir=tb_logdir, progress_bar=cfg.get("progress_bar", True), cfg=cfg,
                      logger=logger, tb_writer=tb_writer)
    
    if args.resume: 
        load_checkpoint(args.resume, model, optimizer=optim, scheduler=sched, trainer=trainer)

    # 9. Run
    trainer.fit(train_loader, val_loader, epochs=cfg["epochs"], start_epoch=start_epoch)
    trainer.evaluate(val_loader)

if __name__ == "__main__":
    main()