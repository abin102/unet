# scripts/debug_inspect_crash.py  (fixed)

import sys, pathlib
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]  # one level up (dnn_template/)
sys.path.insert(0, str(PROJECT_ROOT))

import torch, os, sys, yaml, traceback
from collections import OrderedDict


from losses import get_loss



# run from project root (dnn_template/)
CANDIDATES = [
    "debug/crash_final.pth",
    "debug/crash_nan_grad.pth",
    "debug/crash_saved_step.pth",
    "debug/crash_debug.pth",
]

ckpt_path = None
for p in CANDIDATES:
    if os.path.exists(p):
        ckpt_path = p
        break

if ckpt_path is None:
    print("No debug checkpoint found in debug/*.pth. Looked for:", CANDIDATES)
    sys.exit(1)

print("Loading checkpoint:", ckpt_path)
d = torch.load(ckpt_path, map_location="cpu")
print("keys in checkpoint:", list(d.keys()))

# print saved exception/traceback if present
if "exc" in d:
    print("\n=== saved exception (exc) ===")
    print(d.get("exc"))
if "traceback" in d:
    print("\n=== saved traceback (traceback) ===")
    print(d.get("traceback"))

# helpful prints if present
if "inputs" in d or "batch" in d or "inputs" in d:
    b = d.get("batch", d.get("inputs", None))
    print("\nFound saved batch/inputs keys:", list(b.keys()) if isinstance(b, dict) else type(b))
    imgs = None
    targets = None
    if isinstance(b, dict):
        imgs = b.get("images", b.get("input", None))
        targets = b.get("targets", b.get("labels", None))
    else:
        # unknown format - try common keys in checkpoint root
        imgs = d.get("inputs", None)
        targets = d.get("targets", None)

    if imgs is not None:
        try:
            print("images dtype/shape/min/max/mean:",
                  getattr(imgs, "dtype", None), getattr(imgs, "shape", None),
                  float(torch.min(imgs)), float(torch.max(imgs)),
                  float(torch.mean(imgs)) if imgs.is_floating_point() else None)
        except Exception as e:
            print("Could not print image stats:", e)
    if targets is not None:
        try:
            print("targets dtype/shape:", getattr(targets, "dtype", None), getattr(targets, "shape", None))
            if torch.is_tensor(targets):
                vals, counts = torch.unique(targets, return_counts=True)
                print("target unique counts:", {int(v): int(c) for v, c in zip(vals, counts)})
        except Exception as e:
            print("Could not print target stats:", e)

# try to print model_state keys & safe stats
state_key = None
for key in ("model", "model_state", "model_state_dict", "model_state_dict"):
    if key in d:
        state_key = key
        break

if state_key is not None:
    ms = d[state_key]
    if isinstance(ms, dict):
        names = list(ms.keys())[:200]
        print(f"\nModel state has {len(ms)} keys. First {len(names)} keys:")
        for n in names:
            arr = ms[n]
            # Only print stats for tensors
            if isinstance(arr, torch.Tensor):
                a = arr.detach().cpu()
                try:
                    # safe min/max on numeric tensors
                    amin = float(torch.min(a))
                    amax = float(torch.max(a))
                except Exception:
                    amin, amax = None, None
                # mean only for floating tensors
                amean = float(torch.mean(a)) if a.is_floating_point() else None
                print(f"  {n}: shape={tuple(a.shape)}, dtype={a.dtype}, min={amin}, max={amax}, mean={amean}")
            else:
                print(f"  {n}: type={type(arr)}")
    else:
        print("Model state format unrecognized:", type(ms))

# Attempt single-step repro if cfg & model registries available
cfg = None
if "cfg" in d:
    cfg = d["cfg"]
else:
    fallback = "configs/firerisk_resltresnet32.yaml"
    if os.path.exists(fallback):
        cfg = yaml.safe_load(open(fallback, "r"))
        print("\nLoaded cfg from", fallback)
    else:
        print("\nNo cfg found inside checkpoint and no fallback YAML. Will not attempt single-step repro.")

if cfg:
    try:
        from data import REGISTRY as DATA_REG
        from models import REGISTRY as MODEL_REG
        from losses import REGISTRY as LOSS_REG
        print("\nImported project registries.")
        model = MODEL_REG[cfg["model"]](**cfg.get("model_args", {}))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        print("Constructed model and moved to", device)
        loss_args = cfg.get("loss_args", {}) or {}
        loss_fn = get_loss(cfg["loss"], num_classes=cfg["model_args"]["num_classes"], **loss_args)
        print("Constructed loss using get_loss()")


        if "inputs" in d or "batch" in d or "inputs" in d:
            b = d.get("batch", d.get("inputs", None))
            if isinstance(b, dict):
                imgs = b.get("images", b.get("input", None))
                targs = b.get("targets", b.get("labels", None))
            else:
                imgs = d.get("inputs", None)
                targs = d.get("targets", None)

            if imgs is None:
                print("No images found in saved batch; attempting to sample from dataset.")
                imgs = None
            else:
                imgs = imgs.to(device)
                if targs is not None and torch.is_tensor(targs):
                    targs = targs.to(device)
        else:
            imgs = None
            targs = None

        if imgs is None:
            # sample a small batch from dataset
            train_set, _ = DATA_REG[cfg["dataset"]](cfg["data_dir"],
                                                    image_size=cfg.get("image_size", 224),
                                                    to_rgb=cfg.get("to_rgb", False),
                                                    imagenet_norm=cfg.get("imagenet_norm", True))
            from torch.utils.data import DataLoader
            dl = DataLoader(train_set, batch_size=min(8, cfg.get("batch_size", 8)), shuffle=True)
            imgs, targs = next(iter(dl))
            imgs = imgs.to(device)
            if torch.is_tensor(targs):
                targs = targs.to(device)
            print("Sampled one batch from dataset for reproduction.")

        # build loss if available
                # Build loss in the same way train.py does (use get_loss helper if available).
        loss_fn = None
        loss_name = cfg.get("loss", "ce").lower()
        loss_args = cfg.get("loss_args", {}) or {}

        # prefer the get_loss helper (registered factories expect num_classes, head/medium/tail, beta)
        try:
            from losses import get_loss
            loss_fn = get_loss(
                loss_name,
                num_classes=cfg["model_args"]["num_classes"],
                head_classes=loss_args.get("head_classes"),
                medium_classes=loss_args.get("medium_classes"),
                tail_classes=loss_args.get("tail_classes"),
                beta=loss_args.get("beta", 0.5),
            )
            print("Constructed loss using get_loss()")
        except Exception:
            # fallback: try registry dict if available (may require positional args)
            try:
                LOSS_REG = globals().get("LOSS_REG", None)
                if LOSS_REG and loss_name in LOSS_REG:
                    # try to call with num_classes if factory expects it
                    try:
                        loss_fn = LOSS_REG[loss_name](
                            num_classes=cfg["model_args"]["num_classes"],
                            head_classes=loss_args.get("head_classes"),
                            medium_classes=loss_args.get("medium_classes"),
                            tail_classes=loss_args.get("tail_classes"),
                            beta=loss_args.get("beta", 0.5),
                        )
                    except TypeError:
                        # last resort: pass raw kwargs (may fail for factories that require positional args)
                        loss_fn = LOSS_REG[loss_name](**loss_args)
                    print("Constructed loss using LOSS_REG fallback.")
                else:
                    # direct import fallback for ResLTLoss
                    from losses.reslt_loss import ResLTLoss
                    loss_fn = ResLTLoss(
                        num_classes=cfg["model_args"]["num_classes"],
                        head_classes=loss_args.get("head_classes"),
                        medium_classes=loss_args.get("medium_classes"),
                        tail_classes=loss_args.get("tail_classes"),
                        beta=loss_args.get("beta", 0.5),
                    )
                    print("Constructed ResLTLoss fallback.")
            except Exception as ex:
                print("Could not construct loss (final fallback):", ex)
                loss_fn = None

        # single-forward/backward with anomaly detection
        model.train()
        torch.autograd.set_detect_anomaly(True)
        optim = torch.optim.Adam(model.parameters(), lr=1e-5)
        try:
            out = model(imgs)

            # handle models that return (logits, aux) or similar tuples
            if isinstance(out, tuple) or isinstance(out, list):
                print("\nForward OK. model returned a tuple/list of length", len(out))
                # try to print stats for each element
                for i, o in enumerate(out):
                    if isinstance(o, torch.Tensor):
                        print(f"  out[{i}]: shape={tuple(o.shape)}, dtype={o.dtype}, min={float(o.min()):.6f}, max={float(o.max()):.6f}, mean={float(o.mean()):.6f}")
                    else:
                        print(f"  out[{i}]: type={type(o)}")
                # assume the primary logits are the first element (common pattern)
                logits = out[0] if len(out) > 0 and isinstance(out[0], torch.Tensor) else out
            else:
                logits = out
                print("\nForward OK. logits shape:", tuple(logits.shape), getattr(logits, "dtype", None))

            # compute loss — prefer the loss function signature that accepts logits (not the whole tuple)
            try:
                loss = loss_fn(logits, targs)
            except Exception as ex:
                print("Loss function failed when called with logits only; trying full model output. Exception:", ex)
                loss = loss_fn(out, targs)

            print("Loss value (detached):", float(loss.detach()))
            loss.backward()

            print("Backward completed (no exception raised during this reproduction).")
        except Exception:
            print("\nException during single-step reproduction (will print stack):")
            traceback.print_exc()

        # inspect grads
        bad = False
        for name, p in model.named_parameters():
            if p.grad is None:
                continue
            g = p.grad.detach().cpu()
            has_nan = torch.isnan(g).any().item()
            has_inf = torch.isinf(g).any().item()
            if has_nan or has_inf:
                print(f"\nBAD GRAD for {name}: has_nan={has_nan}, has_inf={has_inf}, grad_stats min/max/mean:",
                      float(torch.min(g)), float(torch.max(g)),
                      float(torch.mean(g)) if g.is_floating_point() else None)
                # print a few grads for quick inspection
                flat = g.view(-1)
                print("First 40 grad values:", flat[:40].tolist())
                bad = True
                break
        if not bad:
            print("\nNo NaN/Inf found in gradients after single-step backward attempt.")

    except Exception as ex:
        print("Exception while trying to rebuild/run model:", ex)
        traceback.print_exc()
else:
    print("\nNo cfg available; printed checkpoint info and exiting.")
