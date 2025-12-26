import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Any
from loguru import logger
from tqdm import tqdm

def calculate_segmentation_metrics(preds, targets, num_classes, epsilon=1e-6):
    """
    Generic Metric Calculator.
    Produces keys: dice_class_0, dice_class_1, ...
    """
    results = {}
    
    # 1. Global Pixel Accuracy
    correct_pixels = (preds == targets).sum().item()
    total_pixels = targets.numel()
    results["acc_pixel"] = correct_pixels / max(total_pixels, 1)

    # 2. MAE
    mae = torch.abs(preds.float() - targets.float()).mean().item()
    results["mae"] = mae

    # 3. Per-Class Metrics
    dice_sum = 0.0
    sen_sum = 0.0
    spec_sum = 0.0

    for cls in range(num_classes):
        p_flat = (preds == cls).float().view(-1)
        t_flat = (targets == cls).float().view(-1)
        
        tp = (p_flat * t_flat).sum()
        fp = (p_flat * (1 - t_flat)).sum()
        fn = ((1 - p_flat) * t_flat).sum()
        tn = ((1 - p_flat) * (1 - t_flat)).sum()
        
        dice = (2. * tp + epsilon) / (2. * tp + fp + fn + epsilon)
        sens = (tp + epsilon) / (tp + fn + epsilon)
        spec = (tn + epsilon) / (tn + fp + epsilon)
        
        # --- GENERIC NAMING ---
        # No more "lung" or "infection" hardcoding.
        results[f"dice_class_{cls}"] = dice.item()
        results[f"sen_class_{cls}"] = sens.item()
        results[f"spec_class_{cls}"] = spec.item()
        
        dice_sum += dice.item()
        sen_sum += sens.item()
        spec_sum += spec.item()

    # 4. Macro Averages
    results["dice_macro"] = dice_sum / num_classes
    results["sen_macro"]  = sen_sum / num_classes
    results["spec_macro"] = spec_sum / num_classes

    return results

@torch.no_grad()
def evaluate(trainer, loader, epoch=-1) -> Tuple[float, float, Dict[str, Any]]:
    model = trainer.model
    was_training = model.training
    model.eval()

    device = trainer.device
    requested_metrics = set(trainer.cfg.get("metrics", []))
    monitor_key = trainer.cfg.get("early_stop", {}).get("monitor", "val_loss").replace("val/", "")
    
    # Dynamic Class Count
    n_classes = trainer.cfg.get("model_args", {}).get("n_classes", None)

    total_loss = 0.0
    total_batches = 0
    metric_sums = {} 
    
    iterator = tqdm(loader, desc=f"Validating Epoch {epoch}", leave=False) if getattr(trainer, "progress_bar", True) else loader

    for batch in iterator:
        try:
            x = batch[0].to(device, non_blocking=True)
            y = batch[1].to(device, non_blocking=True)
            if y.ndim == 4: y = y.squeeze(1)

            # Forward
            try: outputs = model(x)
            except TypeError: outputs = model(x, current_epoch=epoch)

            if isinstance(outputs, dict): logits = outputs.get('logits', outputs.get('out'))
            elif isinstance(outputs, tuple): logits = outputs[0]
            else: logits = outputs
            
            # Loss
            loss = trainer.loss_fn(logits, y)
            
            bs = x.size(0)
            total_loss += loss.item() * bs
            total_batches += bs
            
            # Infer classes if needed
            if n_classes is None:
                n_classes = logits.size(1)

            preds = logits.argmax(dim=1)
            batch_raw_metrics = calculate_segmentation_metrics(preds, y, num_classes=n_classes)
            
            for k, v in batch_raw_metrics.items():
                metric_sums[k] = metric_sums.get(k, 0.0) + (v * bs)

        except Exception as e:
            logger.exception(f"Evaluation failed at batch {total_batches}: {e}")
            raise

    if total_batches == 0: return 0.0, 0.0, {}

    avg_loss = total_loss / total_batches
    full_avg_metrics = {k: v / total_batches for k, v in metric_sums.items()}
    full_avg_metrics["val_loss"] = avg_loss

    # Filter
    final_stats = {"val_loss": avg_loss}
    for k in requested_metrics:
        if k in full_avg_metrics:
            final_stats[k] = full_avg_metrics[k]
    
    if monitor_key in full_avg_metrics and monitor_key not in final_stats:
        final_stats[monitor_key] = full_avg_metrics[monitor_key]

    main_score = final_stats.get(monitor_key, avg_loss)

    # Logging
    log_parts = [f"Loss={avg_loss:.4f}"]
    for k in sorted(requested_metrics):
        if k in final_stats:
            log_parts.append(f"{k}={final_stats[k]:.4f}")
            
    logger.info(f"Eval Epoch {epoch}: " + " | ".join(log_parts))

    if getattr(trainer, "tb_writer", None):
        for k, v in final_stats.items():
            trainer.tb_writer.add_scalar(f"val/{k}", v, epoch)

    if was_training:
        model.train()

    return avg_loss, main_score, final_stats