from pathlib import Path
import json
import random
import cv2
import numpy as np


def read_image(path, gray=False):
    im = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8),
                      cv2.IMREAD_GRAYSCALE if gray else cv2.IMREAD_COLOR)
    if im is None:
        raise FileNotFoundError(path)
    return im


def write_image(path, im):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, data = cv2.imencode(path.suffix, im)
    if not ok:
        raise RuntimeError(f"Cannot encode {path}")
    data.tofile(str(path))


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def input_channels(gray):
    """Identical local contrast normalization for synthetic and real data."""
    x = gray.astype(np.float32)
    if x.max() > 1.5:
        x /= 255.0
    bg = cv2.GaussianBlur(x, (0, 0), 12)
    residual = x - bg
    scale = max(float(np.std(residual)), 0.008)
    local = np.clip(residual / (4 * scale), -1, 1)
    fine = x - cv2.GaussianBlur(x, (0, 0), 2)
    fine = np.clip(fine / (4 * scale), -1, 1)
    return np.stack([x, local, fine]).astype(np.float32)


def starts(length, size, stride):
    if length <= size:
        return [0]
    return sorted(set(list(range(0, length-size+1, stride)) + [length-size]))
