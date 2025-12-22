# train.py (slim)
from utils.logging_utils import init_logger_and_tb

import sys
import sched
import torch
import argparse, os
from utils.config_utils import load_cfg
from utils.data_utils import make_dataloaders, infer_input_size_from_loader
from utils.builders import build_model_from_cfg, build_loss_and_splits, get_optim, get_scheduler
from utils.inspect_utils import tensor_stats
from utils.data_stats import save_dataset_stats
from data import REGISTRY as DATA_REG
from models import REGISTRY as MODEL_REG
from callbacks import Checkpoint, CSVLogger, ModelLogger
from trainer import Trainer
from callbacks.logging import LoggingCallback
from utils.freeze_utils import apply_freeze
from utils.debug_utils import print_trainable_summary, print_optimizer_summary
from utils.checkpoint_utils import load_checkpoint
from utils.wandb_utils import init_wandb, watch_model
from torchsummary import summary
from utils.onnx_file_creator import export_model_to_onnx

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--over", nargs="*", default=[])
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()

    cfg = load_cfg(args.cfg, overrides=args.over)
    # enforce reproducible seeds early — set *before* creating dataset/dataloaders
    seed = cfg.get("seed", None)
    out_dir = f'{cfg["output_dir"]}/{cfg["exp_name"]}-{__import__("time").strftime("%Y%m%d-%H%M%S")}'
    os.makedirs(out_dir, exist_ok=True)

    # initialising the logger 
    log_cfg = cfg.get("logging", {}) or {}

    tb_logdir = None
    if log_cfg.get("tensorboard", False):
        tb_logdir = os.path.join(out_dir, "tb")
        os.makedirs(tb_logdir, exist_ok=True)

    logger, tb_writer = init_logger_and_tb(
        debug=log_cfg.get("debug", False),
        debug_dir=out_dir,
        tb_logdir=tb_logdir,
        log_level=log_cfg.get("level", "INFO"),
        rotation=log_cfg.get("rotation", "50 MB"),
        retention=log_cfg.get("retention", "14 days"),
        compression=log_cfg.get("compression", "zip"),
    )
    logger.info("Logger initialized from config. Logs saved in {}", out_dir)

    use_wandb = bool(cfg.get("use_wandb", False))
    wandb_cfg = cfg.get("wandb", {}) or {}

    wandb, use_wandb = init_wandb(cfg, out_dir)


    if seed is not None:
        import random, numpy as _np
        random.seed(seed)
        _np.random.seed(seed)
        import torch
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
        # make cudnn deterministic (may slow training slightly but matches repo behavior)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        logger.info(f"[seed] fixed random seed = {seed}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Logger initialized from config. Logs saved in {}", out_dir)
    
    
    # prepare per-run TensorBoard and attention dirs    #attn_dir = os.path.join(out_dir, "attn")+
    # create them so SummaryWriter / file writes will succeed        
    # os.makedirs(attn_dir, exist_ok=True)

    # create the logging callback with per-run dirs
    cb = LoggingCallback(tb_logdir=tb_logdir, use_wandb=cfg.get("use_wandb", False), save_attention_dir=None)

    # Data
    data_args = cfg.get("data_args", {}) or {}
    train_set, val_set = DATA_REG[cfg["dataset"]](cfg["data_dir"], **data_args, 
                                                  image_size=cfg.get("image_size",224), 
                                                  to_rgb=cfg.get("to_rgb",False), 
                                                  imagenet_norm=cfg.get("imagenet_norm",True),
                                                  mean=cfg.get("mean",None), std=cfg.get("std",None))
    stats = save_dataset_stats(out_dir, train_set, val_set, cfg=cfg)
    logger.info("Dataset stats saved to:", os.path.join(out_dir, "dataset_stats.json"))

    train_loader, val_loader = make_dataloaders(train_set, val_set, batch_size=cfg["batch_size"], 
                                                num_workers=cfg["num_workers"], pin_memory=cfg["pin_memory"], 
                                                persistent_workers=cfg["persistent_workers"])

    # Build model
    model_cfg = {"name": cfg["model"], "params": cfg.get("model_args", {})}
    model = build_model_from_cfg(model_cfg, device=device)
    logger.info("Model '{}' built and moved to device: {}", cfg["model"], device)

    # summarize via callback + determine input_size
    # TODO : check the logic here and refactor if needed
    inferred_input_size = infer_input_size_from_loader(train_loader)
    mdl_logger = ModelLogger(out_dir, input_size=inferred_input_size, 
                             save_trace=cfg.get("model_logger",{}).get("save_trace",False), 
                             enabled=cfg.get("model_logger",{}).get("enabled",True))
    mdl_logger.on_train_begin(type("T", (), {"model": model, "out_dir": out_dir, "cfg": cfg}))

    # freeze logic (same as before) ...
    freeze_info = apply_freeze(model, cfg, verbose=True)
    trainer_backbone_frozen = freeze_info["trainer_backbone_frozen"]

    logger.info("Backbone frozen: {}", trainer_backbone_frozen)

    # build loss/optim/sched using functions from builders.py
    loss_fn, class_splits = build_loss_and_splits(cfg, train_set)
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    # create final optimizer & scheduler (use current trainable params)
    optim = get_optim(cfg["optim"], [p for p in model.parameters() if p.requires_grad], **(cfg.get("optim_args", {}) or {}))
    sched = get_scheduler(cfg["scheduler"], optim, **(cfg.get("scheduler_args", {}) or {}))


    # load checkpoint into the *final* optimizer/scheduler (and model)
    start_epoch = 1
    if args.resume:
        ck_info = load_checkpoint(args.resume, model, optimizer=optim, scheduler=sched, trainer=None)
        # load_checkpoint returns a dict containing 'start_epoch' and 'raw' (the raw ckpt dict).
        start_epoch = int(ck_info.get("start_epoch", 1))
        logger.info(f"[resume] loaded checkpoint from {args.resume}, resuming at epoch {start_epoch}")

    # print_trainable_summary(model)
    # print_optimizer_summary(optim, model=model)


    # resume, callbacks, trainer creation (same)
    # cbs = [CSVLogger(out_dir), Checkpoint(out_dir, monitor=cfg["early_stop"]["monitor"], mode=cfg["early_stop"]["mode"]), mdl_logger]
    cbs = [CSVLogger(out_dir), Checkpoint(out_dir, monitor=cfg["early_stop"]["monitor"], mode=cfg["early_stop"]["mode"]), mdl_logger]
    all_cbs = cbs + [cb]

    # getting the model structure in ONNX format
    # export_model_to_onnx(
    #     model,
    #     onnx_path=os.path.join(out_dir, "model.onnx"),
    #     input_size=224,
    #     device=device,
    # )
    
    # sys.exit(0)


    trainer = Trainer(model=model, loss_fn=loss_fn, optimizer=optim, scheduler=sched, 
                      amp=cfg["amp"], grad_clip=cfg["grad_clip"], device=device, callbacks=all_cbs, 
                      tb_logdir=None, progress_bar=cfg.get("progress_bar", True), class_splits=class_splits, cfg=cfg,
                      logger=logger, tb_writer=tb_writer)
    if args.resume: 
        load_checkpoint(args.resume, model, optimizer=optim, scheduler=sched, trainer=trainer)
    trainer._backbone_frozen = trainer_backbone_frozen
    if use_wandb:
        dummy_input = torch.randn(1, 3, 32, 32).cuda()
        summary_str = str(summary(model, input_size=(3, 32, 32)))
        wandb.run.log({"model_summary": wandb.Html("<pre>" + summary_str + "</pre>")})

        # watch_model(wandb, model, log="gradients", log_freq=100, log_graph=False)

    trainer.fit(train_loader, val_loader, epochs=cfg["epochs"], start_epoch=start_epoch)
    trainer.evaluate(val_loader)

if __name__ == "__main__":
    main()
