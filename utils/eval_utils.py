# utils/eval_utils.py
import torch
import numpy as np
from typing import Tuple, Dict, Any
from utils.outputs import normalize_outputs
from loguru import logger


@torch.no_grad()
def evaluate(trainer, loader, epoch=-1) -> Tuple[float, float, Dict[str, Any]]:
    """
    Evaluate model on a validation or test loader.
    Logs detailed information about progress, device, metrics, and stability.
    Now supports Difficulty Bucket analysis (Hard/Medium/Easy).

    Returns:
        val_loss (float)
        val_acc (float)
        stats (dict): head/medium/tail accuracy AND hard/medium/easy accuracy
    """
    model = trainer.model
    was_training = model.training
    model.eval()

    logger.debug(f"Starting evaluation (model set to eval mode). training={was_training}")
    device = trainer.device
    logger.debug(f"Evaluation using device={device}")

    total = correct = 0
    loss_sum = 0.0
    num_classes = None
    class_counts = None
    class_correct = None

    # --- NEW: Difficulty Tracking Containers ---
    all_difficulty_labels = [] # Stores 0, 1, 2
    all_correct_bools = []     # Stores True/False for every image
    has_difficulty_data = False
    # -------------------------------------------

    # Changed from 'for step, (x, y)' to 'for step, batch' to handle variable length
    for step, batch in enumerate(loader):
        try:
            # Unpack safely based on batch size
            x = batch[0].to(device, non_blocking=True)
            y = batch[1].to(device, non_blocking=True)
            
            # Check for optional difficulty label (3rd element)
            current_diff = None
            if len(batch) >= 3:
                current_diff = batch[2] # Keep on CPU usually, or move to device if needed later
                has_difficulty_data = True

            try:
                outputs = model(x, current_epoch=epoch)
            except TypeError:
                outputs = model(x)
            
            # W = None # (Assuming normalize_outputs logic commented out as in your snippet)

            if isinstance(outputs, dict):
                logits = outputs['logits']
            elif isinstance(outputs, tuple) and len(outputs) == 3:
                logits = outputs[0] + outputs[1] + outputs[2]
            else:
                logits = outputs
            
            # logger.debug("the logits to compute prediction is {}", logits.shape)
            
            try:
                res = trainer.loss_fn(logits, y)
            except TypeError:
                # Fallback if loss_fn args are weird
                res = trainer.loss_fn(logits, y)
            
            loss = res

            # NaN detection
            if not torch.isfinite(logits).all():
                logger.error(f"Non-finite logits detected during evaluation at batch {step}.")
                # ... (Existing crash debug logic) ...
                raise RuntimeError("Non-finite logits detected during evaluation")

            preds = logits.argmax(1)
            y_idx = y.argmax(dim=1) if y.dim() > 1 else y.view(-1)

            # --- Standard Accuracy Tracking ---
            bs = x.size(0)
            loss_sum += float(loss.item()) * bs
            
            # Calculate boolean correctness for this batch
            batch_correct_mask = (preds == y_idx)
            batch_correct_count = int(batch_correct_mask.sum().item())
            
            correct += batch_correct_count
            total += bs

            # --- NEW: Accumulate Difficulty Data if present ---
            if current_diff is not None:
                # Store boolean correctness (True/False) and the difficulty label
                all_correct_bools.append(batch_correct_mask.cpu())
                all_difficulty_labels.append(current_diff.cpu())

            # --- Existing Class Tracking (HMT) ---
            if num_classes is None:
                num_classes = logits.size(1)
                class_counts = torch.zeros(num_classes, device=device)
                class_correct = torch.zeros(num_classes, device=device)
                logger.debug(f"Initialized class tracking: num_classes={num_classes}")

            for c in range(num_classes):
                mask = (y_idx == c)
                cnt = int(mask.sum().item())
                if cnt > 0:
                    class_counts[c] += cnt
                    class_correct[c] += int((preds[mask] == c).sum().item())

            if step % 50 == 0 and step > 0:
                logger.debug(f"Eval progress: step={step} samples={total} partial_acc={correct / max(total, 1):.4f}")

        except Exception as e:
            logger.exception(f"Evaluation failed at batch {step}: {e}")
            raise

    val_loss = loss_sum / total
    val_acc = correct / total
    logger.info(f"Evaluation completed: samples={total} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

    stats = {}

    # --- 1. Compute Head/Medium/Tail Stats (Existing) ---
    if num_classes is not None:
        eps = 1e-12
        per_class_acc = (class_correct.float() / (class_counts.float() + eps))

        def _mean_for(indices):
            if not indices: return None
            import torch as _torch
            idx = _torch.tensor([i for i in indices if 0 <= i < num_classes], device=device, dtype=_torch.long)
            if idx.numel() == 0: return None
            valid_mask = (class_counts[idx] > 0)
            if valid_mask.sum() == 0: return None
            return per_class_acc[idx][valid_mask].mean().item()

        splits = getattr(trainer, "class_splits", None) or {}
        stats["head_acc"] = _mean_for(splits.get("head", []))
        stats["medium_acc"] = _mean_for(splits.get("medium", []))
        stats["tail_acc"] = _mean_for(splits.get("tail", []))
    else:
        logger.warning("num_classes not determined; skipping class-level stats computation.")

    # --- 2. NEW: Compute Difficulty Bucket Stats ---
    # Only runs if the dataset provided difficulty labels
    # logger.debug(f"Has difficulty data: {has_difficulty_data}, collected {len(all_difficulty_labels)} entries.")
    if has_difficulty_data and len(all_difficulty_labels) > 0:
        logger.debug("Computing difficulty bucket stats from accumulated data.")
        # Concatenate lists into single tensors
        flat_diffs = torch.cat(all_difficulty_labels)    # [N]
        flat_correct = torch.cat(all_correct_bools)      # [N]

        # 0=Hard, 1=Medium, 2=Easy (Must match your dataset mapping)
        mapping = {0: "hard_acc", 1: "medium_acc", 2: "easy_acc"}

        for bucket_id, stat_key in mapping.items():
            mask = (flat_diffs == bucket_id)
            if mask.sum() > 0:
                # Mean of booleans gives accuracy
                acc = flat_correct[mask].float().mean().item()
                stats[stat_key] = acc
            else:
                stats[stat_key] = 0.0 # Or None, depending on preference
        
        logger.info(f"Difficulty Stats: Hard={stats.get('hard_acc',0):.3f}, Med={stats.get('medium_acc',0):.3f}, Easy={stats.get('easy_acc',0):.3f}")
    
    # restore mode
    if was_training:
        model.train()
        logger.debug("Restored model to train() mode after evaluation.")

    return val_loss, val_acc, stats