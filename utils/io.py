import os, yaml, torch
from datetime import datetime

def timestamp():
    return datetime.now().strftime("%Y%m%d-%H%M%S")

def ensure_dir(p): 
    os.makedirs(p, exist_ok=True)

def save_config(cfg, out_dir):
    ensure_dir(out_dir)
    with open(os.path.join(out_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(cfg, f)

def save_state(path, state): 
    torch.save(state, path)

# ---------- NEW: load checkpoint ----------
def load_checkpoint(path, model, device="cuda"):
    ckpt = torch.load(path, map_location=device)

    # Handle common cases
    if "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    elif "model" in ckpt:
        state_dict = ckpt["model"]
    elif "model_state_dict" in ckpt:   # your case
        state_dict = ckpt["model_state_dict"]
    else:
        state_dict = ckpt  # assume it's already a state_dict

    # Strip "module." if saved with DataParallel
    new_state_dict = {}
    for k, v in state_dict.items():
        new_k = k[7:] if k.startswith("module.") else k
        new_state_dict[new_k] = v

    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    print(f"✅ Loaded checkpoint from {path}")
    if missing:
        print("Missing keys:", missing)
    if unexpected:
        print("Unexpected keys:", unexpected)

    return model


