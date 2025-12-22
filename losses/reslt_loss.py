import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

class ResLTLoss(nn.Module):
    def __init__(self, num_classes, head_classes, medium_classes, tail_classes,
                 class_to_idx=None, beta=0.5):
        super().__init__()
        self.num_classes = num_classes
        self.beta = beta
        self.ce = nn.CrossEntropyLoss()

        if class_to_idx is not None:
            head_classes   = [class_to_idx[c] for c in head_classes]
            medium_classes = [class_to_idx[c] for c in medium_classes]
            tail_classes   = [class_to_idx[c] for c in tail_classes]
            logger.debug("head_classes indices: {}", head_classes)
            logger.debug("medium_classes indices: {}", medium_classes)
            logger.debug("tail_classes indices: {}", tail_classes)
            
        # store lists (validate too)
        self.head_classes = list(head_classes)
        self.medium_classes = list(medium_classes)
        self.tail_classes = list(tail_classes)

        # canonical masks as python lists (computed once)
        self._all_classes = sorted(set(self.head_classes + self.medium_classes + self.tail_classes))
        self._medium_tail  = sorted(set(self.medium_classes + self.tail_classes))
        self._tail_only    = list(self.tail_classes)


        logger.debug("all_classes: {}", self._all_classes)
        logger.debug("medium_tail: {}", self._medium_tail)
        logger.debug("tail_only: {}", self._tail_only)



    def forward(self, outputs, targets):
        """
        Robust wrapper around the original ResLTLoss forward.
        When non-finite values are detected, saves a debug dump and raises a descriptive error.
        """
        # Accept either tuple/list of 3 tensors or a single tensor (fallback)
        if not (isinstance(outputs, (tuple, list)) and len(outputs) == 3):
            if torch.is_tensor(outputs):
                return self.ce(outputs, targets)
            raise ValueError(f"Expected outputs tuple of length 3 (H,M,T). Got {type(outputs)} len={len(outputs) if hasattr(outputs,'__len__') else 'N/A'}")

        logitH, logitM, logitT = outputs
        device = targets.device
        batch_size = targets.size(0)

        # quick sanity
        assert targets.dim() == 1, f"targets must be 1-D class indices, got {targets.shape}"
        assert logitH.size(0) == batch_size and logitM.size(0) == batch_size and logitT.size(0) == batch_size, "batch dim mismatch"

        logger.debug("the targets shape is {}", targets.shape)
        logger.debug("the logitH shape is {}", logitH.shape)
        logger.debug("the logitM shape is {}", logitM.shape)
        logger.debug("the logitT shape is {}", logitT.shape)

        # ---------------------------
        # Build per-sample boolean selectors (N,) — EXACTLY the script's behavior
        # ---------------------------
        # Preferred: use torch.isin if available (fast, avoids one-hot)
        
        idx_all = torch.tensor(self._all_classes, device=device)
        idx_med = torch.tensor(self._medium_tail, device=device)
        idx_tail = torch.tensor(self._tail_only, device=device)
        labelH_sel = torch.isin(targets, idx_all)
        labelM_sel = torch.isin(targets, idx_med)
        labelT_sel = torch.isin(targets, idx_tail)

        logger.debug("label selectors shapes: {} {} {}", labelH_sel.shape, labelM_sel.shape, labelT_sel.shape)
        logger.debug("label selector true counts: H={} M={} T={}",
                    int(labelH_sel.sum().item()), int(labelM_sel.sum().item()), int(labelT_sel.sum().item()))

        # ---------------------------
        # keep the old stats & NaN checks for logits
        # ---------------------------
        def stats(t):
            return {
                "shape": tuple(t.shape),
                "dtype": str(t.dtype),
                "min": float(torch.min(t).detach().cpu()) if t.numel() else None,
                "max": float(torch.max(t).detach().cpu()) if t.numel() else None,
                "mean": float(torch.mean(t).detach().cpu()) if t.is_floating_point() and t.numel() else None,
                "has_nan": bool(torch.isnan(t).any().item()) if t.numel() else False,
                "has_inf": bool(torch.isinf(t).any().item()) if t.numel() else False,
            }

        bad = []
        for name, logits in (("H", logitH), ("M", logitM), ("T", logitT)):
            s = stats(logits)
            if s["has_nan"] or s["has_inf"]:
                bad.append((name, s))
        if bad:
            dbg = {
                "bad_branches": bad,
                "targets_shape": tuple(targets.shape),
                "targets_sample": targets.detach().cpu()[:64],
                "logitH_sample_stats": stats(logitH),
                "logitM_sample_stats": stats(logitM),
                "logitT_sample_stats": stats(logitT),
                "loss_beta": self.beta,
            }
            try:
                param_summ = {}
                for i, (n, p) in enumerate(self.named_parameters() if hasattr(self,'named_parameters') else []):
                    if i >= 8:
                        break
                    try:
                        pa = p.detach().cpu()
                        param_summ[n] = {"shape": tuple(pa.shape), "min": float(torch.min(pa)), "max": float(torch.max(pa))}
                    except Exception:
                        param_summ[n] = "err"
                dbg["param_summary"] = param_summ
            except Exception:
                dbg["param_summary"] = "could not gather params"

            torch.save(dbg, "debug/reslt_loss_nan_debug.pth")
            raise RuntimeError(f"Non-finite logits detected in ResLTLoss branches. Saved debug/reslt_loss_nan_debug.pth. Bad branches: {bad}")

        # ---------------------------
        # ice_loss expects a (N,) boolean selector now
        # ---------------------------
        def ice_loss(logits, sample_mask):

            ce_per_sample = F.cross_entropy(logits, targets, reduction="none")  # (N,)
            sel = ce_per_sample[sample_mask]                                    # select scalar CE values
            if not torch.isfinite(sel).all():
                torch.save({"sel": sel.detach().cpu(), "logits": logits.detach().cpu(), "targets": targets.detach().cpu()},
                        "debug/ice_nan.pth")
                raise RuntimeError("Non-finite CE values in ice_loss selection")
            return sel.sum()

        # ---------------------------
        # compute ice loss exactly like the training script (per-sample CE with group selectors)
        # ---------------------------
        loss_ice_H = ice_loss(logitH, labelH_sel)
        loss_ice_M = ice_loss(logitM, labelM_sel)
        loss_ice_T = ice_loss(logitT, labelT_sel)

        # If you want the original script's normalization (divide by total selected samples), uncomment:
        # total_selected = max(1, int(labelH_sel.sum().item() + labelM_sel.sum().item() + labelT_sel.sum().item()))
        # loss_ice = (loss_ice_H + loss_ice_M + loss_ice_T) / total_selected

        # Otherwise keep sum of branch means (matches your prior implementation style)
        loss_ice = (loss_ice_H + loss_ice_M + loss_ice_T)/(labelH_sel.sum() + labelM_sel.sum() + labelT_sel.sum())

        logger.debug("loss_ice components: H={} M={} T={}", float(loss_ice_H.detach().cpu()), float(loss_ice_M.detach().cpu()), float(loss_ice_T.detach().cpu()))
        logger.debug("label selector sample preview (first 16): H={} M={} T={}",
                    labelH_sel[:16].detach().cpu().tolist(), labelM_sel[:16].detach().cpu().tolist(), labelT_sel[:16].detach().cpu().tolist())

        # --------------------------- 
        # final steps (unchanged)
        # ---------------------------
        logit_sum = logitH + logitM + logitT
        if not torch.isfinite(logit_sum).all():
            torch.save({"logit_sum": logit_sum.detach().cpu(), "logitH": logitH.detach().cpu(), "logitM": logitM.detach().cpu(), "logitT": logitT.detach().cpu()},
                    "debug/logit_sum_nan.pth")
            raise RuntimeError("Non-finite values in logit_sum (after addition)")

        loss_fce = self.ce(logit_sum, targets)
        logger.debug("target shape for calculating fce_loss: {}", targets.shape)

        loss = self.beta * loss_ice + (1.0 - self.beta) * loss_fce
        logger.debug("beta value for final loss computation: {}", self.beta)
        if not torch.isfinite(loss):
            torch.save({"loss": float(loss.detach().cpu()), "loss_ice": float(loss_ice.detach().cpu()) if torch.is_tensor(loss_ice) else loss_ice, "loss_fce": float(loss_fce.detach().cpu())}, "debug/final_loss_nan.pth")
            raise RuntimeError(f"Final loss is non-finite: {loss}")
        return loss
