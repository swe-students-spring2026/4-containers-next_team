"""Model loading and prediction for realtime inference."""

from pathlib import Path
from typing import Tuple

import numpy as np
import torch

from model import Net
from src_config import CLASSES, MODEL_PATH


def resolve_model_path() -> Path:
    """Resolve model path across run locations."""
    configured = Path(MODEL_PATH)
    here = Path(__file__).resolve().parent

    candidates = [
        configured if configured.is_absolute() else Path.cwd() / configured,
        here / configured,
        here / "data" / "processed" / "sign_language_model.pth",
    ]

    for c in candidates:
        if c.resolve().exists():
            return c.resolve()

    return (here / "data" / "processed" / "sign_language_model.pth").resolve()


def load_model(device: torch.device) -> Net:
    """Load and return the trained model on the selected device."""
    path = resolve_model_path()

    if not path.exists():
        raise FileNotFoundError(f"Model not found at {path}")

    model = Net().to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model


def pixels_to_tensor(gray_28: np.ndarray, device: torch.device) -> torch.Tensor:
    """Convert 28x28 image to normalized model input vectors."""
    tensor = torch.from_numpy(gray_28).float().unsqueeze(0).unsqueeze(0) / 255.0
    tensor = (tensor - 0.5) / 0.5
    return tensor.to(device)


def predict(
    model: Net, gray_28: np.ndarray, device: torch.device
) -> Tuple[int, str, float]:
    """Run model inference and return class index, label, and confidence."""
    with torch.no_grad():
        logits = model(pixels_to_tensor(gray_28, device))
        probs = torch.softmax(logits, dim=1)
        conf, pred_idx = torch.max(probs, dim=1)

    idx = int(pred_idx.item())
    label = CLASSES[idx] if 0 <= idx < len(CLASSES) else str(idx)

    return idx, label, float(conf.item())
