import sys
import os
import argparse
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# Utils
from utils.logging_utils import init_logger_and_tb  # Uses the modified function above
from utils.config_utils import load_cfg
from utils.data_utils import make_dataloaders
from utils.builders import build_model_from_cfg, get_optim, get_scheduler
from utils.data_stats import save_dataset_stats
from data import REGISTRY as DATA_REG
from models import REGISTRY as MODEL_REG
from callbacks import Checkpoint, CSVLogger
from trainer import Trainer
from callbacks.logging import LoggingCallback
from utils.checkpoint_utils import load_checkpoint
from utils.wandb_utils import init_wandb
from utils.builders import build_loss

def main():
    # ====================================================
    # DDP SETUP
    # ====================================================
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    is_ddp = local_rank != -1

    if is_ddp:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        local_rank = 0
    
    # Master is Rank 0 (in DDP) or the only process (in Single GPU)
    is_master = (not is_ddp) or (dist.get_rank() == 0)

    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--over", nargs="*", default=[])
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()

    cfg = load_cfg(args.cfg, overrides=args.over)
    seed = cfg.get("seed", 42)
    
    # ====================================================
    # DIRS & LOGGING
    # ====================================================
    out_dir = f'{cfg["output_dir"]}/{cfg["exp_name"]}-{__import__("time").strftime("%Y%m%d-%H%M%S")}'
    
    # 1. Create Directory (Master Only)
    if is_master:
        os.makedirs(out_dir, exist_ok=True)
    
    # 2. Sync: Wait for directory to be created before other ranks try to write logs
    if is_ddp:
        dist.barrier()

    # 3. Init Logging (ALL Ranks)
    # We pass 'rank' and 'is_master' so utils can route console/files correctly
    log_cfg = cfg.get("logging", {}) or {}
    tb_logdir = os.path.join(out_dir, "tb") if (log_cfg.get("tensorboard", False) and is_master) else None
    
    logger, tb_writer = init_logger_and_tb(
        debug=log_cfg.get("debug", False),
        debug_dir=out_dir,
        tb_logdir=tb_logdir,
        log_level=log_cfg.get("level", "INFO"),
        rank=local_rank,         # <--- PASS RANK
        is_master=is_master      # <--- PASS MASTER FLAG
    )

    # 4. WandB (Master Only)
    wandb_run, use_wandb = None, False
    if is_master:
        wandb_run, use_wandb = init_wandb(cfg, out_dir)
        if use_wandb: logger.info("WandB initialized.")

    # ====================================================
    # TRAINING SETUP
    # ====================================================
    # Seeding
    import random, numpy as _np
    random.seed(seed)
    _np.random.seed(seed)
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.backends.cudnn.deterministic = True
    logger.debug(f"Random seed set to {seed}")

    # Data
    data_args = cfg.get("data_args", {}) or {}
    logger.info(f"Loading dataset: {cfg['dataset']}...")
    train_set, val_set = DATA_REG[cfg["dataset"]](cfg["data_dir"], **data_args)
    logger.info(f"Train Size: {len(train_set)}, Val Size: {len(val_set)}")
    
    if is_master:
        try:
            save_dataset_stats(out_dir, train_set, val_set, cfg=cfg)
        except Exception as e:
            logger.warning(f"Could not save stats: {e}")

    # DDP Samplers
    if is_ddp:
        train_sampler = DistributedSampler(train_set, shuffle=True)
        val_sampler = DistributedSampler(val_set, shuffle=False) 
        shuffle_train = False
    else:
        train_sampler = None
        val_sampler = None
        shuffle_train = True

    train_loader = torch.utils.data.DataLoader(
        train_set, 
        batch_size=cfg["batch_size"], 
        sampler=train_sampler,
        shuffle=shuffle_train,
        num_workers=cfg["num_workers"],
        pin_memory=cfg["pin_memory"],
        persistent_workers=cfg["persistent_workers"]
    )
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=cfg["batch_size"], sampler=val_sampler, shuffle=False,
        num_workers=cfg["num_workers"], pin_memory=cfg["pin_memory"]
    )

    # Model
    model_cfg = {"name": cfg["model"], "params": cfg.get("model_args", {})}
    model = build_model_from_cfg(model_cfg, device=device)
    logger.info(f"Model '{cfg['model']}' built.")

    # DDP Wrapping
    if is_ddp:
        logger.debug("Converting to SyncBatchNorm...")
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
        logger.success("DDP Model Wrapped.")

    # Optimizer & Loss
    loss_fn = build_loss(cfg["loss"], cfg.get("loss_args", {}), device=device)
    optim = get_optim(cfg["optim"], [p for p in model.parameters() if p.requires_grad], **(cfg.get("optim_args", {}) or {}))
    sched = get_scheduler(cfg["scheduler"], optim, **(cfg.get("scheduler_args", {}) or {}))

    # Resume
    start_epoch = 1
    if args.resume:
        logger.info(f"Resuming from: {args.resume}")
        ck_info = load_checkpoint(args.resume, model, optimizer=optim, scheduler=sched, trainer=None)
        start_epoch = int(ck_info.get("start_epoch", 1))

    # Callbacks (Master Only)
    all_cbs = []
    if is_master:
        cbs = [
            CSVLogger(out_dir), 
            Checkpoint(out_dir, monitor=cfg["early_stop"]["monitor"], mode=cfg["early_stop"]["mode"]), 
        ]
        all_cbs = cbs + [LoggingCallback(tb_logdir=tb_logdir, use_wandb=use_wandb)]

    # Trainer
    logger.info("Initializing Trainer...")
    trainer = Trainer(model=model, loss_fn=loss_fn, optimizer=optim, scheduler=sched, 
                      amp=cfg["amp"], grad_clip=cfg["grad_clip"], device=device, callbacks=all_cbs, 
                      tb_logdir=tb_logdir, progress_bar=(cfg.get("progress_bar", True) and is_master), 
                      cfg=cfg, logger=logger, tb_writer=tb_writer)
    
    # Run
    trainer.fit(train_loader, val_loader, epochs=cfg["epochs"], start_epoch=start_epoch, 
                train_sampler=train_sampler if is_ddp else None)

    if is_master:
        logger.info("Starting Final Evaluation...")
        trainer.evaluate(val_loader)

    if is_ddp:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
    # torchrun --nproc_per_node=2 train.py --cfg configs/repar_unet_ct_segmentation.yaml
