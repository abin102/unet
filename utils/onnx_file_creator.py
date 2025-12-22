import torch
import torch.nn as nn


class _OnnxLogitWrapper(nn.Module):
    """Thin wrapper used ONLY for export. It does NOT change the actual model."""
    def __init__(self, model, output_key="logits"):
        super().__init__()
        self.model = model
        self.output_key = output_key

    def forward(self, x):
        out = self.model(x)
        # If model returns a dict, pull out the tensor with logits
        if isinstance(out, dict):
            if self.output_key not in out:
                raise RuntimeError(f"Expected key '{self.output_key}' in model output dict, "
                                   f"got keys: {list(out.keys())}")
            return out[self.output_key]
        # Otherwise assume the model itself already returns a tensor of logits
        return out


def export_model_to_onnx(
    model,
    onnx_path="model.onnx",
    input_size=None,
    device="cpu",
    output_key="logits",
):
    # 1) unwrap & move to device
    model_cpu = model
    try:
        if hasattr(model, "module"):
            model_cpu = model.module
        model_cpu = model_cpu.to(device)
        model_cpu.eval()
    except Exception as e:
        print("Warning: could not move model to device for export:", e)

    # 2) infer input size
    if input_size is None:
        input_size = 224  # fallback
    dummy = torch.randn(1, 3, input_size, input_size, device=device)

    # 3) wrap model so ONNX sees only logits (tensor), NOT the dict
    wrapped = _OnnxLogitWrapper(model_cpu, output_key=output_key)

    try:
        with torch.no_grad():
            torch.onnx.export(
                wrapped,
                dummy,
                onnx_path,
                export_params=True,
                opset_version=13,          # 13 is safer these days
                do_constant_folding=True,
                input_names=["input"],
                output_names=[output_key], # keep name meaningful
                dynamic_axes={"input": {0: "batch"}, output_key: {0: "batch"}},
                verbose=False,
            )
        print(f"ONNX export succeeded -> {onnx_path}")
    except Exception as e:
        print("ONNX export failed:", e)
        # TorchScript fallback on the same wrapped module
        try:
            traced = torch.jit.trace(wrapped, dummy)
            traced.save(onnx_path.replace(".onnx", ".pt"))
            print("Saved TorchScript trace as fallback:", onnx_path.replace(".onnx", ".pt"))
        except Exception as ee:
            print("TorchScript fallback failed:", ee)
