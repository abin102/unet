# ----------------------------------------------------------
# Quick sanity check that all helper modules import cleanly
# Run this from the project root:
#     python test_imports.py
# ----------------------------------------------------------

print("🧩 Testing helper imports...")

# make sure project root is importable
import sys, pathlib
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# ---- core utils ----
print("\n[1/4] utils.config_utils")
from utils.config_utils import load_cfg, _coerce_numbers
print("   ✓ loaded successfully")

print("\n[2/4] utils.data_utils")
from utils.data_utils import make_dataloaders, infer_input_size_from_loader
print("   ✓ loaded successfully")

print("\n[3/4] utils.builders")
from utils.builders import build_model_from_cfg, build_loss_and_splits, get_optim, get_scheduler
print("   ✓ loaded successfully")

print("\n[4/4] utils.inspect_utils")
from utils.inspect_utils import tensor_stats
print("   ✓ loaded successfully")

# ---- callbacks ----
print("\n[callbacks]")
from callbacks.model_logger import ModelLogger
print("   ✓ ModelLogger loaded")

# ---- optional smoke test for a model registry ----
try:
    from models import REGISTRY as MODEL_REG
    print(f"   ✓ models.REGISTRY contains: {list(MODEL_REG.keys())[:5]}")
except Exception as e:
    print("   ⚠️  models import failed:", e)

# ---- run a small function test (no heavy ops) ----
cfg_test = {"optim": "sgd", "optim_args": {"lr": 0.01}}
opt = get_optim(cfg_test["optim"], [__import__('torch').nn.Parameter(__import__('torch').randn(1))], **cfg_test["optim_args"])
print("   ✓ Optimizer test OK:", type(opt).__name__)

print("\n✅ All imports OK")
